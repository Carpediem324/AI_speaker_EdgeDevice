import numpy as np
import pyaudio
import time
import csv
import os
import sys
import gc
import threading
import signal
import json
import base64
import io
import paho.mqtt.client as mqtt
from tflite_runtime.interpreter import Interpreter
from scipy import signal as scipy_signal
from collections import deque
from dotenv import load_dotenv
from multiprocessing import shared_memory
# .env 파일 로드
load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/sed_v1/.env')
# signal_shm
SHM_NAME = "signal_shm"
shm = shared_memory.SharedMemory(name=SHM_NAME)
def read_signal():
    signal_bytes = bytes(shm.buf[:16])
    signal = signal_bytes.split(b'\x00')[0].decode('utf-8')
    return signal
# 전역 변수
running = True
last_activity_time = time.time()

# MQTT 설정 로드
DEVICE_ID = os.getenv("DEVICE_ID",str(time.time()))
MQTT_BROKER = os.getenv("BROKER_ADDRESS", "localhost")
MQTT_PORT = int(os.getenv("BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "audio/detection")
SED_CLIENT=os.getenv("SED_CLIENT","SED_CLIENT_"+DEVICE_ID)
MODEL_PATH = os.getenv("MODEL_PATH","yamnet.tflite")
CLASS_MAP_PATH=os.getenv("CLASS_MAP_PATH","yamnet_class_map.csv")
CUSTOM_MODEL_PATH=os.getenv("CUSTOM_MODEL_PATH","yamnet_multilabel_model.tflite")
# 커스텀 클래스 라벨 및 threshold 설정
custom_class_labels = [
    "Alarm_bell_ringing", "Speech", "Dog", "Cat", "Dishes", "Frying",
    "Electric_shaver_toothbrush", "Blender", "Running_water", "Vacuum_cleaner", "Falling"
]
alarm_class_names = [
    "Alarm",
    "Alarm clock",
    "Siren",
    "Civil defense siren",
    "Fire alarm",
    "Smoke detector, smoke alarm",
    "Vehicle horn, car horn, honking",
    "Air horn, truck horn",
    "Foghorn",
    "Bell",
    "Church bell",
    "Jingle bell",
    "Bicycle bell",
    "Fire engine, fire truck (siren)",
    "Ambulance (siren)",
    "Police car (siren)"
]
speech_class_names = [
    "Speech",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Speech synthesizer",
    "Hubbub, speech noise, speech babble",
    "Chatter"
]
water_class_names = [
    "Water",
    "Waterfall",
    "Stream",
    "Boat, Water vehicle",
    "Splash, splatter",
    "Water tap, faucet",
    "Bathtub (filling or washing)",
    "Toilet flush",
]
# 각 라벨별 threshold 값 로드
custom_thresholds = {}
for label in custom_class_labels:
    threshold_key = f"THRESHOLD_{label}"
    custom_thresholds[label] = float(os.getenv(threshold_key, 0.5))  # 기본값 0.5


print("로드된 라벨별 threshold 값:")
for label, threshold in custom_thresholds.items():
    print(f"  - {label}: {threshold}")

# MQTT 클라이언트 설정
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,client_id=SED_CLIENT)
print("Paho MQTT v2.0 클라이언트 초기화 성공")

if MQTT_USERNAME and MQTT_PASSWORD:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    print(f"MQTT 인증 설정됨: 사용자={MQTT_USERNAME}")
# 인증이 필요한 경우 사용자 이름과 비밀번호 설정


# MQTT 콜백 함수
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

# 시그널 핸들러
def signal_handler(sig, frame):
    global running
    print("\n[INFO] 프로그램 종료 신호를 받았습니다.", flush=True)
    running = False

# 디버깅 메시지 출력 함수
def log_debug(message):
    print(f"[DEBUG] {message}", flush=True)

# 파일 존재 확인 함수
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] 파일을 찾을 수 없음: {file_path}", flush=True)
        return False
    return True

# 모델 로드 함수 (예외 처리 추가)
def load_model(model_path):
    try:
        log_debug(f"모델 로드 시도: {model_path}")
        interpreter = Interpreter(model_path)
        log_debug("모델 로드 성공")
        return interpreter
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

custom_class_labels = [
    "Alarm_bell_ringing", "Speech", "Dog", "Cat", "Dishes", "Frying",
    "Electric_shaver_toothbrush", "Blender", "Running_water", "Vacuum_cleaner", "Falling"
]

# 클래스 맵 로드 함수
def class_names_from_csv(csv_text):
    """YAMNet 클래스 맵 CSV에서 클래스 이름을 로드합니다."""
    class_names = []
    for row in csv.reader([x for x in csv_text.split('\n') if x]):
        try:
            class_names.append(row[2])
            #print(row[2])
        except IndexError:
            continue
    return class_names[1:]  # 첫 번째 행(헤더)를 제외한 모든 행 반환

# 정확한 크기로 리샘플링하는 함수
def resample_audio_exact(audio_data, orig_sr, target_sr, target_length):
    """
    오디오 데이터를 원본 샘플 레이트에서 목표 샘플 레이트로 변환하고,
    정확히 target_length 길이의 결과를 반환합니다.
    """
    try:
        # 리샘플링 수행
        resampled_data = scipy_signal.resample(audio_data, target_length)
        
        # 정확한 길이 확인
        if len(resampled_data) != target_length:
            log_debug(f"경고: 리샘플링 후 길이가 다릅니다. 예상: {target_length}, 실제: {len(resampled_data)}")
            # 길이 조정 (부족하면 0으로 채우고, 넘치면 자름)
            if len(resampled_data) < target_length:
                pad_length = target_length - len(resampled_data)
                resampled_data = np.pad(resampled_data, (0, pad_length), 'constant')
            else:
                resampled_data = resampled_data[:target_length]
        
        return resampled_data
    except Exception as e:
        print(f"[ERROR] 리샘플링 실패: {e}", flush=True)
        return np.zeros(target_length, dtype=np.float32)  # 오류 시 무음 데이터 반환

# MQTT로 오디오 데이터 전송 함수
def send_audio_to_mqtt(sound_class, confidence, raw_audio_buffer, sample_rate):
    """MQTT를 통해 오디오 데이터를 전송합니다."""
    try:
        # 오디오 데이터를 Base64로 인코딩
        audio_bytes = io.BytesIO()
        np.save(audio_bytes, raw_audio_buffer)
        audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode('utf-8')
        
        # MQTT로 전송할 JSON 데이터
        message = {
            "timestamp": time.time(),
            "class": sound_class,
            "confidence": float(confidence),
            "audio_data": audio_base64,
            "sample_rate": sample_rate
        }
        
        # MQTT로 전송
        result = mqtt_client.publish(MQTT_TOPIC, json.dumps(message))
        log_debug(f"MQTT 메시지 전송 결과: {result.rc}, 소리 클래스: {sound_class} ({confidence:.3f})")
    except Exception as e:
        print(f"[ERROR] MQTT 전송 실패: {e}", flush=True)
        import traceback
        traceback.print_exc()

def main():
    global running
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 모델 파일 경로
    model_path = MODEL_PATH
    class_map_path = CLASS_MAP_PATH
    custom_model_path = CUSTOM_MODEL_PATH

    # 파일 존재 확인
    required_files = [model_path, class_map_path, custom_model_path]
    for file_path in required_files:
        if not check_file_exists(file_path):
            print(f"[ERROR] 필수 파일 누락: {file_path}", flush=True)
            return

    # TFLite 모델 로드
    interpreter = load_model(model_path)
    if interpreter is None:
        return
    
    # 메모리 할당 명시적으로 수행
    interpreter.allocate_tensors()

    # 모델 입력 및 출력 세부 정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 인덱스 가져오기
    waveform_input_index = input_details[0]['index']
    scores_output_index = output_details[0]['index']
    embeddings_output_index = output_details[1]['index'] if len(output_details) > 1 else None
    spectrogram_output_index = output_details[2]['index'] if len(output_details) > 2 else None

    # 모델 입력 형태 확인
    input_shape = input_details[0]['shape']

    # 커스텀 모델 로드
    custom_interpreter = load_model(custom_model_path)
    if custom_interpreter is None:
        return
    
    custom_interpreter.allocate_tensors()

    custom_input_index = custom_interpreter.get_input_details()[0]['index']
    custom_output_index = custom_interpreter.get_output_details()[0]['index']

    # MQTT 연결
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()  # MQTT 백그라운드 루프 시작
        print(f"MQTT 브로커({MQTT_BROKER}:{MQTT_PORT})에 연결 시도 중...")
    except Exception as e:
        print(f"MQTT 연결 실패: {e}")
        print("MQTT 없이 계속합니다.")

    def predict_custom_model(embedding_vector):
        """
        평균 임베딩 벡터 (1024,)를 입력으로 커스텀 모델에 추론 요청
        """
        try:
            # 입력 데이터 복사본 만들기 (원본 배열 수정 방지)
            embedding_copy = np.copy(embedding_vector)
            embedding_input = np.expand_dims(embedding_copy, axis=0).astype(np.float32)  # (1, 1024)
            
            custom_interpreter.set_tensor(custom_input_index, embedding_input)
            custom_interpreter.invoke()
            output = custom_interpreter.get_tensor(custom_output_index)
            
            # 복사본 생성 후 반환
            return np.copy(output[0])  # (11,)
        except Exception as e:
            print(f"[ERROR] 커스텀 모델 예측 실패: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return np.zeros(len(custom_class_labels), dtype=np.float32)

    # 클래스 이름 로드
    try:
        with open(class_map_path, 'r') as f:
            class_names = class_names_from_csv(f.read())
        log_debug(f"클래스 맵 로드 성공: {len(class_names)} 클래스")
    except FileNotFoundError:
        log_debug(f"경고: 클래스 맵 파일({class_map_path})을 찾을 수 없습니다. 기본 클래스 이름을 생성합니다.")
        # 기본 클래스 이름 생성 (YAMNet은 일반적으로 521개 클래스 사용)
        class_names = [f"Class_{i}" for i in range(521)]
    except Exception as e:
        print(f"[ERROR] 클래스 맵 로드 중 오류: {e}", flush=True)
        class_names = [f"Class_{i}" for i in range(521)]

    # 오디오 설정
    INPUT_SAMPLE_RATE = 48000  # 입력 샘플 레이트 (48kHz)
    TARGET_SAMPLE_RATE = 16000  # YAMNet 요구사항 (16kHz)

    # 청크 및 윈도우 크기 설정 (1초 단위로 작업)
    CHUNK_SIZE_SECONDS = 1  # 1초 청크
    WINDOW_SIZE_SECONDS = 3  # 3초 윈도우

    # 샘플 수 계산
    INPUT_CHUNK_SIZE = int(INPUT_SAMPLE_RATE * CHUNK_SIZE_SECONDS)  # 1초에 해당하는 48kHz 샘플 수
    TARGET_CHUNK_SIZE = int(TARGET_SAMPLE_RATE * CHUNK_SIZE_SECONDS)  # 1초에 해당하는 16kHz 샘플 수
    TARGET_WINDOW_SIZE = int(TARGET_SAMPLE_RATE * WINDOW_SIZE_SECONDS)  # 3초에 해당하는 16kHz 샘플 수

    log_debug(f"입력 청크 크기: {INPUT_CHUNK_SIZE} 샘플 ({CHUNK_SIZE_SECONDS}초)")
    log_debug(f"다운샘플링 후 청크 크기: {TARGET_CHUNK_SIZE} 샘플")
    log_debug(f"윈도우 크기: {TARGET_WINDOW_SIZE} 샘플 ({WINDOW_SIZE_SECONDS}초)")

    # 작은 청크 설정 (1024 샘플)
    SMALL_CHUNK_SIZE = 2048
    SMALL_CHUNKS_PER_SECOND = INPUT_SAMPLE_RATE // SMALL_CHUNK_SIZE
    log_debug(f"작은 청크 크기: {SMALL_CHUNK_SIZE} 샘플, 초당 {SMALL_CHUNKS_PER_SECOND}개 청크")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    def process_audio_data(waveform):
        """TFLite 모델로 오디오 데이터를 처리합니다."""
        try:
            # 입력 텐서 크기 조정 (필요 시)
            if len(input_shape) == 1:  # 1D 입력 예상
                interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=False)
                interpreter.allocate_tensors()
            
            # 입력 데이터 복사본 만들기 (원본 배열 수정 방지)
            waveform_copy = np.copy(waveform)
            
            # 입력 텐서에 데이터 설정
            interpreter.set_tensor(waveform_input_index, waveform_copy)
            
            # 추론 실행
            interpreter.invoke()
            
            # 결과 가져오기 (복사본 생성)
            scores = np.copy(interpreter.get_tensor(scores_output_index))
            
            # 임베딩과 스펙트로그램도 가져오기 (가능한 경우)
            embeddings = None
            if embeddings_output_index is not None:
                embeddings = np.copy(interpreter.get_tensor(embeddings_output_index))
            
            spectrogram = None
            if spectrogram_output_index is not None:
                spectrogram = np.copy(interpreter.get_tensor(spectrogram_output_index))
            
            return scores, embeddings, spectrogram
        except Exception as e:
            print(f"[ERROR] 오디오 처리 실패: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 기본 반환값 (빈 데이터)
            dummy_scores = np.zeros((521,), dtype=np.float32)
            dummy_embeddings = np.zeros((1024,), dtype=np.float32) if embeddings_output_index is not None else None
            dummy_spectrogram = None
            return dummy_scores, dummy_embeddings, dummy_spectrogram

    def check_audio_signal(audio_data):
        """오디오 신호의 품질을 확인합니다."""
        try:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            peak = np.max(np.abs(audio_data))
            # 너무 많은 로그가 생성되지 않도록 간소화
            print(f"RMS: {rms:.6f}, Peak: {peak:.6f}", end="\r", flush=True)
            return rms > 0.001  # 최소 오디오 레벨
        except Exception as e:
            print(f"[ERROR] 오디오 신호 확인 실패: {e}", flush=True)
            return False

    # 오디오 스트림과 PyAudio 객체를 None으로 초기화
    stream = None
    p = None
    
    try:
        # 오디오 스트림 설정
        p = pyaudio.PyAudio()
        device_index = None
        print("사용 가능한 입력 장치:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"{i}: {device_info['name']} (입력 채널: {device_info['maxInputChannels']})")

        # dsnoop_capture 장치 자동 선택
        device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0 and 'dsnoop_capture' in device_info['name']:
                device_index = i
                print(f"dsnoop_capture 장치 선택됨: {i}: {device_info['name']}")
                print(f"  - 기본 샘플 레이트: {device_info['defaultSampleRate']}Hz")
                break
        
        # 장치를 찾지 못한 경우
        if device_index is None:
            print("dsnoop_capture 장치를 찾을 수 없습니다. 수동으로 선택해주세요.")
            device_index = int(input("사용할 입력 장치 번호를 입력하세요: "))
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,  # 48kHz로 설정
            input=True,
            input_device_index=device_index,
            frames_per_buffer=SMALL_CHUNK_SIZE  # 작은 청크 사용
        )

        log_debug(f"실시간 소리 분류 시작...")
        log_debug(f"입력: {INPUT_SAMPLE_RATE}Hz, 목표: {TARGET_SAMPLE_RATE}Hz")
        log_debug(f"슬라이딩 윈도우: 3초 윈도우, 1초마다 업데이트")
        log_debug("Ctrl+C로 종료하세요.")

        # 3초 윈도우 초기화 (16kHz 샘플링 레이트 기준)
        # 시작 시 모든 샘플을 0으로 초기화
        window_buffer = np.zeros(TARGET_WINDOW_SIZE, dtype=np.float32)
        
        # 원본 오디오 데이터를 저장할 버퍼 (MQTT 전송용)
        raw_window_buffer = np.zeros(INPUT_SAMPLE_RATE * WINDOW_SIZE_SECONDS, dtype=np.int16)
        
        log_debug("3초 윈도우 초기화 완료")
        
        # 1초를 위한 임시 저장소
        chunk_buffer = []
        raw_chunk_buffer = []
        chunk_samples_collected = 0
        chunks_needed = SMALL_CHUNKS_PER_SECOND  # 1초를 위해 필요한 작은 청크 수
        
        # 모델 처리 시간 모니터링
        last_process_time = time.time()
        
        # 주기적인 메모리 정리 시간
        last_gc_time = time.time()
        GC_INTERVAL = 60  # 60초마다 가비지 컬렉션 실행
        
        # 무한 루프 보호 장치
        loop_count = 0
        MAX_LOOPS = 10000000  # 10만 번 루프 후 강제 종료
        
        while running and loop_count < MAX_LOOPS:
            
            # 주기적인 가비지 컬렉션
            if time.time() - last_gc_time > GC_INTERVAL:
                log_debug("주기적인 메모리 정리 수행")
                gc.collect()
                last_gc_time = time.time()
            
            try:
                # 작은 청크 단위로 오디오 데이터 읽기
                audio_data = stream.read(SMALL_CHUNK_SIZE, exception_on_overflow=False)
                small_chunk_int16 = np.frombuffer(audio_data, dtype=np.int16)
                small_chunk = small_chunk_int16.astype(np.float32) / 32768.0
                
                # 작은 청크를 임시 버퍼에 추가
                chunk_buffer.append(small_chunk)
                chunk_samples_collected += len(small_chunk)
                
                # 1초 분량의 데이터가 모이면 처리
                if len(chunk_buffer) >= chunks_needed:
                    log_debug(f"1초 데이터 처리 시작 - 청크 버퍼 크기: {len(chunk_buffer)}")
                    loop_count += 1
                    print(f"loop count : {loop_count}")
                    
                    # 1초 데이터 조합 (복사본 생성)
                    one_second_data = np.concatenate(chunk_buffer.copy())[:INPUT_CHUNK_SIZE]  # 정확히 1초 길이 유지
                    
                    # 청크 버퍼 초기화
                    chunk_buffer = []
                    raw_chunk_buffer = []
                    chunk_samples_collected = 0
                    
                    # 48kHz에서 16kHz로 다운샘플링 (정확한 크기 보장)
                    resampled_data = resample_audio_exact(one_second_data, INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE, TARGET_CHUNK_SIZE)
                    
                    # 샘플 수 확인
                    log_debug(f"1초 데이터 - 원본: {len(one_second_data)}, 리샘플링 후: {len(resampled_data)} 샘플")
                    
                    # 윈도우 업데이트: 가장 오래된 1초 제거하고 새 1초 추가 (복사본 사용)
                    window_buffer = np.roll(np.copy(window_buffer), -TARGET_CHUNK_SIZE)  # 앞의 1초 제거
                    window_buffer[-TARGET_CHUNK_SIZE:] = resampled_data  # 뒤에 새 1초 추가
                    
                    # 원본 오디오 버퍼 업데이트 (MQTT 전송용)
                    if read_signal() != "start":
                        print("keyword 인식중 stop되었습니다.")
                        continue
                    # 현재 시간과 경과 시간 계산
                    current_time = time.time()
                    elapsed = current_time - last_process_time
                    last_process_time = current_time
                    
                    log_debug(f"현재 시간: {time.strftime('%H:%M:%S')}, 경과 시간: {elapsed:.3f}초")
                    log_debug(f"윈도우 크기: {len(window_buffer)} 샘플 ({WINDOW_SIZE_SECONDS}초)")
                    
                    if not check_audio_signal(window_buffer):
                        log_debug("오디오 레벨이 너무 낮습니다. 이번 3초는 처리하지 않습니다.")
                        continue
                    
                    # YAMNet TFLite 모델로 추론
                    log_debug("모델 추론 시작")
                    inference_start = time.time()
                    scores, embeddings, spectrogram = process_audio_data(window_buffer)
                    inference_time = time.time() - inference_start
                    log_debug(f"추론 시간: {inference_time:.3f}초")
                    
                    # 임베딩 정보 출력
                    if embeddings is not None:
                        log_debug(f"임베딩 처리 시작 - 형태: {embeddings.shape}")
                        
                        if len(embeddings.shape) > 1:
                            # 여러 프레임 임베딩이면 평균 계산
                            avg_embedding = np.mean(embeddings, axis=0)
                            log_debug(f"임베딩 정보: 형태 {embeddings.shape}, 평균 계산됨")
                        else:
                            avg_embedding = embeddings
                            log_debug(f"임베딩 정보: 차원 {embeddings.shape[0]}")
                            
                        # 임베딩 복사본 생성하여 메모리 문제 방지
                        avg_embedding = np.copy(avg_embedding)
                    else:
                        log_debug("임베딩이 None입니다. 커스텀 모델 추론을 건너뜁니다.")
                        avg_embedding = np.zeros((1024,), dtype=np.float32)  # 기본값
                    
                    # 점수 처리
                    log_debug(f"점수 처리 시작 - 형태: {scores.shape}")
                    if len(scores.shape) > 1:  # 2D 이상 (시간 차원 포함)
                        class_scores = np.mean(scores, axis=0)
                        log_debug(f"점수 형태: {scores.shape}, 시간 축 평균 계산됨")
                    else:  # 1D (클래스 스코어만)
                        class_scores = scores
                        log_debug(f"점수 형태: {scores.shape}")
                    
                    # 복사본 생성하여 메모리 문제 방지
                    class_scores = np.copy(class_scores)
                    
                    # 상위 클래스 출력
                    top_n = min(5, len(class_scores))
                    top_indices = np.argsort(class_scores)[-top_n:][::-1]
                    
                    log_debug("감지된 소리:")
                    for idx in top_indices:
                        if idx < len(class_names):
                            class_name = class_names[idx]
                            score = class_scores[idx]
                            log_debug(f"{class_name}: {score:.3f}")
                        else:
                            log_debug(f"Unknown class {idx}: {class_scores[idx]:.3f}")
                    
                    # YAMNet 결과에서 상위 클래스 관련 키워드 확인
                    #yamnet_top_classes = [class_names[idx].lower() for idx in top_indices if idx < len(class_names)]
                    has_yamnet_speech = False
                    has_yamnet_alarm = False
                    has_yamnet_water = False
                    for idx in top_indices:
                        if class_names[idx] in speech_class_names and class_scores[idx] >=0.2:
                            has_yamnet_speech = True
                        # if class_names[idx] in alarm_class_names and class_scores[idx] >=0.2:
                        #     has_yamnet_alarm = True
                        if class_names[idx] in water_class_names and class_scores[idx] >=0.2:
                            has_yamnet_water = True

                    log_debug(f"YAMNet 키워드 확인 - 물: {has_yamnet_water}, 음성: {has_yamnet_speech}")
                    
                    # 커스텀 모델 예측
                    log_debug(f"커스텀 모델 예측 시작 - 임베딩 형태: {avg_embedding.shape}")
                    custom_output = predict_custom_model(avg_embedding)
                    
                    log_debug("커스텀 모델 감지 결과:")
                    for i, prob in enumerate(custom_output):
                        label = custom_class_labels[i]
                        threshold = custom_thresholds[label]
                        log_debug(f"{label}: {prob:.3f} (threshold: {threshold})")
                    
                    # 우선 순위에 따라 처리 (Falling > Alarm > Speech)
                    
                    # Falling 검사 (YAMNet 검증 필요 없음)
                    falling_index = custom_class_labels.index("Falling")
                    falling_prob = custom_output[falling_index]
                    
                    # Alarm 검사
                    alarm_index = custom_class_labels.index("Alarm_bell_ringing")
                    alarm_prob = custom_output[alarm_index]
                    alarm_threshold = custom_thresholds["Alarm_bell_ringing"]
                    
                    # Water 검사
                    water_index = 8
                    water_prob = custom_output[water_index]
                    water_threshold = custom_thresholds["Running_water"]

                    # Speech 검사
                    speech_index = custom_class_labels.index("Speech")
                    speech_prob = custom_output[speech_index]
                    speech_threshold = custom_thresholds["Speech"]
                    
                    # 우선순위 순서대로 확인하고 한 번만 메시지 전송
                    if falling_prob >= 0.2:  # Falling은 0.2 이상이면 전송, YAMNet 검증 필요 없음
                        log_debug(f"낙상 감지: Falling ({falling_prob:.3f}) - 임계값 0.2 이상")
                        log_debug("MQTT로 오디오 데이터 전송 중...")
                        send_audio_to_mqtt("Falling", falling_prob, window_buffer, TARGET_SAMPLE_RATE)
                    
                    # Falling이 감지되지 않은 경우 Alarm 검사 Alarm은 yamnet검사안하기로 했다.
                    elif alarm_prob >= alarm_threshold:
                        log_debug(f"알람 감지: Alarm_bell_ringing ({alarm_prob:.3f}) - YAMNet 검증 통과")
                        log_debug("MQTT로 오디오 데이터 전송 중...")
                        send_audio_to_mqtt("Alarm_bell_ringing", alarm_prob, window_buffer, TARGET_SAMPLE_RATE)
                    # Falling과 Alarm이 감지되지 않으면 Water검사한다. 수도를 튼 소리와 frying은 매우 유사하다. yamnet에서도 검사하자.
                    elif has_yamnet_water and water_prob >= water_threshold:
                        log_debug(f"알람 감지: Running_water ({water_prob:.3f}) - YAMNet 검증 통과")
                        log_debug("MQTT로 오디오 데이터 전송 중...")
                        send_audio_to_mqtt("Running_water", water_prob, window_buffer, TARGET_SAMPLE_RATE)

                    # # Falling과 Alarm이 둘 다 감지되지 않은 경우 Speech 검사
                    # elif has_yamnet_speech and speech_prob >= speech_threshold:
                    #     log_debug(f"음성 감지: Speech ({speech_prob:.3f}) - YAMNet 검증 통과")
                    #     log_debug("MQTT로 오디오 데이터 전송 중...")
                    #     send_audio_to_mqtt("Speech", speech_prob, window_buffer, TARGET_SAMPLE_RATE)
                    
                    # 메모리 정리
                    del class_scores, top_indices, custom_output
                    
                    # 처리 종료 메시지
                    log_debug("1초 데이터 처리 완료")
                time.sleep(0.0015)
            except Exception as e:
                print(f"[ERROR] 루프 내부 예외 발생: {e}", flush=True)
                import traceback
                traceback.print_exc()
                time.sleep(1)  # 에러 발생 시 잠시 대기
        
            if loop_count >= MAX_LOOPS:
                print("[WARNING] 최대 루프 수에 도달하여 프로그램을 종료합니다.", flush=True)
                
    except KeyboardInterrupt:
        print("\n[INFO] 키보드 인터럽트로 프로그램 종료", flush=True)
    except Exception as e:
        print(f"\n[ERROR] 주요 예외 발생: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 정리
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if p is not None:
                p.terminate()
            print("[INFO] 오디오 리소스 정리 완료", flush=True)
        except Exception as e:
            print(f"[ERROR] 리소스 정리 중 오류: {e}", flush=True)
        
        # 마지막으로 메모리 정리
        print("[INFO] 프로그램 종료 전 메모리 정리 수행", flush=True)
        gc.collect()

if __name__ == "__main__":
    main()