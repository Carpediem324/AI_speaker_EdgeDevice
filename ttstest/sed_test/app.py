import numpy as np
import pyaudio
import csv
import time
import datetime
import threading
import queue
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import paho.mqtt.client as mqtt
import json
import io
import base64
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env에서 MQTT 설정 가져오기
MQTT_BROKER = os.getenv("BROKER_ADDRESS", "localhost")
MQTT_PORT = int(os.getenv("BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "audio/detection")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# 상수 정의
MODEL_SAMPLE_RATE = 16000  # YAMNet 모델은 16kHz 샘플 레이트를 사용
DEVICE_SAMPLE_RATE = 48000  # 대부분의 장치가 지원하는 샘플 레이트
CHUNK_SIZE = 2048  # 처리할 오디오 청크 크기
CHANNELS = 1
INPUT_SIZE = 15600  # YAMNet 모델의 입력 크기 (0.975초 분량의 오디오)

print(f"MQTT 설정 로드됨: 브로커={MQTT_BROKER}, 포트={MQTT_PORT}, 토픽={MQTT_TOPIC}")
print(f"신뢰도 임계값: {CONFIDENCE_THRESHOLD}")

# MQTT 클라이언트 설정 (Paho MQTT v2.0)
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
print("Paho MQTT v2.0 클라이언트 초기화 성공")

# 인증이 필요한 경우 사용자 이름과 비밀번호 설정
if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    print(f"MQTT 인증 설정됨: 사용자={MQTT_USERNAME}")

# YAMNet 클래스 맵 로딩
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 스킵
    for row in reader:
        class_names.append(row[2])

print(f"클래스 총 개수: {len(class_names)}")

# TFLite 모델 로드
interpreter = tflite.Interpreter(model_path='yamnet.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

# 모델 입력 형태 확인
print("\n모델 입력 세부 정보:")
for detail in input_details:
    print(f"- 이름: {detail['name']}")
    print(f"- 형태: {detail['shape']}")
    print(f"- 타입: {detail['dtype']}")

# MQTT 콜백 함수 (VERSION2 방식)
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT 브로커에 연결됨, 결과 코드: {reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"메시지 ID {mid}가 성공적으로 발행됨, 결과 코드: {reason_code}")

def on_disconnect(client, userdata, reason_code, properties=None):
    print(f"MQTT 브로커에서 연결 해제됨, 결과 코드: {reason_code}")

# MQTT 클라이언트 콜백 설정
mqtt_client.on_connect = on_connect
mqtt_client.on_publish = on_publish
mqtt_client.on_disconnect = on_disconnect

# 오디오 버퍼를 처리하는 클래스
class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        # 정확히 모델 입력 크기에 맞는 버퍼 생성
        self.buffer = np.zeros(INPUT_SIZE, dtype=np.float32)
        # 원본 오디오 데이터를 저장할 버퍼 (PANNs로 전송용)
        self.raw_buffer = np.zeros(INPUT_SIZE * (DEVICE_SAMPLE_RATE // MODEL_SAMPLE_RATE), dtype=np.int16)
        # 1초 길이의 오디오 버퍼 (약 16000 샘플, 모델 샘플레이트 기준)
        self.one_second_buffer_size = MODEL_SAMPLE_RATE
        # 1초 길이의 원본 오디오 버퍼 (약 44100 샘플, 장치 샘플레이트 기준)
        self.one_second_raw_buffer_size = DEVICE_SAMPLE_RATE
        # 마지막 처리 시간 저장 (1초 간격 처리용)
        self.last_process_time = time.time()
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.running = True
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def add_audio_data(self, audio_data):
        self.audio_queue.put(audio_data)
    
    def _process_audio(self):
        while self.running:
            try:
                # 큐에서 오디오 데이터 가져오기
                audio_chunk = self.audio_queue.get(timeout=1)
                
                # 원본 오디오 데이터 저장 (PANNs로 전송용)
                chunk_size = len(audio_chunk)
                self.raw_buffer = np.roll(self.raw_buffer, -chunk_size)
                self.raw_buffer[-chunk_size:] = audio_chunk
                
                # int16을 float32로 정규화 (-1 ~ 1 사이)
                normalized_chunk = audio_chunk.astype(np.float32) / 32768.0
                
                # 리샘플링 (장치 샘플 레이트 -> 모델 샘플 레이트)
                resampled_chunk = signal.resample(
                    normalized_chunk, 
                    int(len(normalized_chunk) * MODEL_SAMPLE_RATE / DEVICE_SAMPLE_RATE)
                )
                
                # 버퍼 업데이트 (가장 오래된 데이터 제거하고 새 데이터 추가)
                chunk_size = len(resampled_chunk)
                if chunk_size > 0:
                    # 버퍼를 롤링하여 새 데이터를 위한 공간 확보
                    self.buffer = np.roll(self.buffer, -chunk_size)
                    # 새 데이터 추가
                    self.buffer[-chunk_size:] = resampled_chunk
                
                # 1초 간격으로 처리
                current_time = time.time()
                if current_time - self.last_process_time >= 1.0 and np.count_nonzero(self.buffer) > INPUT_SIZE / 2:
                    # 입력 데이터 형태 확인
                    expected_shape = input_details[0]['shape']
                    
                    # 버퍼 크기가 정확히 요구 크기와 같아야 함
                    if self.buffer.dtype != np.float32:
                        self.buffer = self.buffer.astype(np.float32)
                    
                    # 모델에 오디오 데이터 입력
                    interpreter.set_tensor(input_details[0]['index'], self.buffer)
                    interpreter.invoke()
                    
                    # 모델 출력 가져오기
                    scores = interpreter.get_tensor(output_details[0]['index'])
                    
                    # YAMNet 출력은 [1, 521] 형태입니다. 첫 번째 배치의 결과만 사용합니다.
                    predictions = scores[0]  # 첫 번째 배치의 결과
                    
                    # 점수가 가장 높은 5개의 인덱스 찾기
                    top_indices = np.argsort(predictions)[-5:][::-1]
                    
                    # 가장 높은 점수
                    max_score = predictions[top_indices[0]]
                    max_class = class_names[top_indices[0]] if top_indices[0] < len(class_names) else "Unknown"
                    
                    print("\n---- 감지된 소리 ----")
                    for i, idx in enumerate(top_indices):
                        # 인덱스가 class_names 범위 내에 있는지 확인
                        if idx < len(class_names):
                            print(f"{i+1}. {class_names[idx]}: {predictions[idx]:.3f}")
                        else:
                            print(f"{i+1}. 알 수 없는 클래스({idx}): {predictions[idx]:.3f}")
                    print("--------------------")
                    
                    # 임계값 이상이면 MQTT로 오디오 데이터 전송
                    if max_score >= CONFIDENCE_THRESHOLD:
                        print(f"임계값({CONFIDENCE_THRESHOLD}) 이상 감지: {max_class} ({max_score:.3f})")
                        print("PANNs 모델로 오디오 데이터 전송 중...")
                        
                        # 오디오 데이터를 Base64로 인코딩
                        audio_bytes = io.BytesIO()
                        np.save(audio_bytes, self.raw_buffer)
                        audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode('utf-8')
                        
                        # MQTT로 전송할 JSON 데이터
                        message = {
                            "timestamp": time.time(),
                            "class": max_class,
                            "confidence": float(max_score),
                            "audio_data": audio_base64,
                            "sample_rate": DEVICE_SAMPLE_RATE
                        }
                        
                        # MQTT로 전송
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
                    
                    # 처리 시간 업데이트
                    self.last_process_time = current_time
                    print(f"1초 데이터 처리 완료: {datetime.datetime.now().strftime('%H:%M:%S')}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"오류 발생: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def stop(self):
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)

# 메인 함수
def main():
    print("YAMNet TensorFlow Lite 오디오 인식 시작...")
    print("Ctrl+C를 눌러 종료하세요.")
    
    # MQTT 연결
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()  # MQTT 백그라운드 루프 시작
        print(f"MQTT 브로커({MQTT_BROKER}:{MQTT_PORT})에 연결 시도 중...")
    except Exception as e:
        print(f"MQTT 연결 실패: {e}")
        print("MQTT 없이 계속합니다.")
    
    # 오디오 처리기 초기화
    processor = AudioProcessor()
    
    # PyAudio 초기화
    audio = pyaudio.PyAudio()
    
    # 사용 가능한 오디오 장치 출력 및 지원 샘플 레이트 확인
    print("\n사용 가능한 오디오 입력 장치:")
    device_index = None  # 기본 장치
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"장치 {i}: {device_info['name']}")
            print(f"  - 기본 샘플 레이트: {device_info['defaultSampleRate']}Hz")
            # USB 오디오 장치를 찾아 자동 선택 (선택 사항)
            if 'usb' in device_info['name'].lower():
                device_index = i
                print(f"  - USB 장치 자동 선택됨!")
    
    # 사용할 장치 인덱스 선택
    if device_index is None:
        print("\n기본 오디오 장치를 사용합니다.")
    else:
        print(f"\n장치 {device_index}를 사용합니다.")
    
    try:
        # 오디오 스트림 열기
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=DEVICE_SAMPLE_RATE,  # 장치가 지원하는 샘플 레이트 사용
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("\n마이크 입력 대기 중...")
        print(f"장치 샘플 레이트: {DEVICE_SAMPLE_RATE}Hz (YAMNet 모델 요구: {MODEL_SAMPLE_RATE}Hz)")
        print(f"YAMNet 입력 크기: {INPUT_SIZE} 샘플 (약 {INPUT_SIZE/MODEL_SAMPLE_RATE:.2f}초)")
        print(f"신뢰도 임계값: {CONFIDENCE_THRESHOLD}")
        print("실시간 리샘플링이 적용됩니다.")
        
        while True:
            # 오디오 데이터 읽기
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # bytes를 numpy 배열로 변환
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 오디오 프로세서에 데이터 전달
            processor.add_audio_data(audio_np)
            
            # 잠시 대기 (CPU 사용량 감소)
            time.sleep(0.092)
            
    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        audio.terminate()
        processor.stop()
        mqtt_client.loop_stop()  # MQTT 루프 중지
        mqtt_client.disconnect()  # MQTT 연결 종료
        print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()