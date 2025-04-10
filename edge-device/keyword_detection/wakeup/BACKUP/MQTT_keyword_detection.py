import time
import json
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
from multiprocessing import shared_memory
import threading

# .env 파일 로드: 최상단에 호출하여 모든 환경변수를 먼저 로드합니다.
load_dotenv()

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

SILENCE_THRESHOLD = 2      # 침묵 임계치 (2초)
RMS_THRESHOLD = 200        # RMS 임계치
TIME_SETUP = 1.0           # 선택 신호 대기 시간 (1.0초) 0.5초는 너무 짧아 불가능함.

# 환경변수에서 값 불러오기

# 토픽관련
SELECT_TOPIC = os.getenv("SELECT_TOPIC")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC")
END_TOPIC = os.getenv("END_TOPIC")
TOPIC_AUDIO = os.getenv("TOPIC_AUDIO")
TOPIC_RMS = os.getenv("TOPIC_RMS")

# 키워드인식 관련
HOT_WORD = os.getenv("HOT_WORD")
REFERENCE_FILE = os.getenv("REFERENCE_FILE")

# MQTT 브로커 관련
BROKER_ADDRESS = os.getenv("BROKER_ADDRESS")
BROKER_PORT = int(os.getenv("BROKER_PORT"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))

# 공유 메모리 생성/연결 함수 (16바이트, 기본 신호는 "start")
def create_or_attach_shared_signal(name="signal_shm", size=16):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        shm.buf[:5] = b'start'
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
    return shm

class MQTTClientHandler:
    '''
    MQTT 클라이언트 핸들러 클래스
    브로커설정과 데이터 전송을 담당.
    음성데이터전송, RMS전송, 연결설정 메소드 제공
    '''
    def __init__(self, debug=False):
        self.debug = debug
        self.broker_address = BROKER_ADDRESS
        self.broker_port = BROKER_PORT
        self.username = MQTT_USERNAME
        self.password = MQTT_PASSWORD
        self.topic_audio = TOPIC_AUDIO
        self.topic_rms = TOPIC_RMS
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "keyword_mqtt_client")
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address, self.broker_port)
        
        self.client.loop_start()
        self.client.subscribe(SELECT_TOPIC)
        self.client.subscribe(DESELECT_TOPIC)

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if self.debug:
            print(f"MQTT 브로커 연결 완료, 상태 코드: {reason_code}")

    def publish_audio(self, data):
        self.client.publish(self.topic_audio, data)

    def publish_rms(self, rms_value):
        payload = json.dumps({"keyword_rms": float(rms_value)})
        self.client.publish(self.topic_rms, payload)

class AudioStreamer:
    '''
    오디오 스트리밍 클래스
    파이오디오 설정, 음성 스트리밍
    '''
    def __init__(self, mqtt_handler, debug=False,
                 format=pyaudio.paInt16, channels=1, rate=48000, chunk=2048, device_index=None):
        self.mqtt_handler = mqtt_handler
        self.debug = debug
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index 
        self.silence_threshold = RMS_THRESHOLD       # RMS 임계치 (환경에 따라 조정)
        self.silence_duration_required = SILENCE_THRESHOLD  # 2초 침묵 시 종료

    def debug_print(self, msg):
        if self.debug:
            print(msg)

    def stream(self):
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.device_index
            )
        except Exception as e:
            self.debug_print("스트리밍용 마이크 열기 실패: " + str(e))
            pa.terminate()
            return

        # 음성 전송은 즉시 시작하고, 동시에 선택 및 디셀렉트 신호를 기다리는 스레드를 실행합니다.
        selection_flag = [None]   # True/False 값 저장용
        stop_streaming = [False]    # 선택/디셀렉트 신호 미수신 시 스트리밍 종료 플래그
        
        def wait_for_selection():
            select_event = threading.Event()
            def on_select_message(client, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode('utf-8'))
                    self.debug_print("수신한 선택 메시지: " + str(data))
                    if data.get("selected_device"):
                        selection_flag[0] = True
                    else:
                        selection_flag[0] = False
                except Exception as e:
                    self.debug_print("선택 메시지 파싱 에러: " + str(e))
                    selection_flag[0] = False
                select_event.set()

            topic = SELECT_TOPIC
            self.mqtt_handler.client.message_callback_add(topic, on_select_message)
            self.debug_print("선택 신호 대기중...")
            select_event.wait(timeout=TIME_SETUP)
            self.mqtt_handler.client.message_callback_remove(topic)
            if selection_flag[0] is None:
                selection_flag[0] = False
            if not selection_flag[0]:
                stop_streaming[0] = True

        # 디셀렉트 신호를 처리하는 콜백 함수
        def on_deselect_message(client, userdata, msg):
            #print("DESELECT MESSAGE")
            try:
                # MQTT 메시지 페이로드를 문자열로 디코딩
                payload = msg.payload.decode('utf-8')
                self.debug_print("수신한 디셀렉트 메시지: " + payload)
                # 메시지에 "deselected_device" 문자열이 포함되어 있으면 스트리밍 종료 플래그를 설정
                if "deselected_device" in payload:
                    stop_streaming[0] = True
            except Exception as e:
                self.debug_print("디셀렉트 메시지 처리 에러: " + str(e))

        # 선택 신호 확인 스레드 시작
        selection_thread = threading.Thread(target=wait_for_selection)
        selection_thread.start()

        # 디셀렉트 메시지 구독 등록 (전체 스트리밍 동안 활성)
        self.mqtt_handler.client.message_callback_add(DESELECT_TOPIC, on_deselect_message)

        self.debug_print("MQTT 음성 스트리밍 시작")
        silence_start = None

        while True:
            # 선택 또는 디셀렉트 신호에 의해 스트리밍 종료 플래그가 세팅되면 종료합니다.
            if stop_streaming[0]:
                self.debug_print("선택/디셀렉트 신호 감지, 스트리밍 종료")
                break

            try:
                audio_data = stream.read(self.chunk, exception_on_overflow=False)
            except Exception as e:
                self.debug_print("스트림 읽기 에러: " + str(e))
                continue

            self.mqtt_handler.publish_audio(audio_data)

            data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            talk_rms = np.sqrt(np.mean(np.square(data)))
            if np.isnan(talk_rms):
                self.debug_print("RMS 값이 NaN, 이번 프레임 건너뜀")
                continue

            print("RMS: " + str(talk_rms))
            if talk_rms < self.silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.silence_duration_required:
                    self.debug_print("2초 침묵 감지, 스트리밍 종료")
                    break
            else:
                silence_start = None

        selection_thread.join()
        # 디셀렉트 메시지 콜백 제거
        self.mqtt_handler.client.message_callback_remove(DESELECT_TOPIC)

        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.debug_print("MQTT 음성 스트리밍 종료")
        self.end_device()

    def end_device(self):
        """
        허브에 디바이스 연결 해제 신호를 전송합니다.
        토픽: END_TOPIC (환경변수에서 불러옴)
        """
        topic = END_TOPIC
        payload = json.dumps(True)  # JSON 형태의 불린 값 전송
        self.mqtt_handler.client.publish(topic, payload)
        self.debug_print("디바이스 연결 해제 신호 전송됨.")
        return True

class HotwordController:
    '''
    키워드 인식 클래스
    Efficient wordnet을 사용하여 키워드 인식
    키워드 인식 시 오디오 스트리밍 시작 및 키워드인식rms 전송
    공유 메모리에 신호 변경(start or end) => SED 에서 활용


    '''
    def __init__(self, mqtt_handler, debug=False):
        self.debug = debug
        self.mqtt_handler = mqtt_handler
        self.audio_streamer = AudioStreamer(mqtt_handler, debug=debug)
        self.base_model = Resnet50_Arc_loss()
        self.hotword_detector = HotwordDetector(
            hotword=HOT_WORD,
            model=self.base_model,
            reference_file=REFERENCE_FILE,
            threshold=CONFIDENCE_THRESHOLD,
            relaxation_time=1 #한 번 키워드가 인식되면, 일정 시간동안 다시 키워드를 감지하지 않도록 설정
        )
        self.mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=0.75,custom_channels=1)
        self.mic_stream.start_stream()
        # 공유 메모리 블록 생성 또는 연결
        self.shm_signal = create_or_attach_shared_signal()
        self.current_signal = "start"  # 초기 신호
        self.last_signal_change = time.time()
        self.min_interval = 1  # 최소 1초 간격

    def debug_print(self, msg):
        if self.debug:
            print(msg)

    def set_signal(self, signal_str):
        # 현재 공유 메모리의 신호 읽기
        current = bytes(self.shm_signal.buf[:16]).split(b'\x00')[0].decode('utf-8')
        if current == signal_str:
            if self.debug:
                print(f"공유 메모리 신호가 이미 '{signal_str}'입니다. 업데이트하지 않습니다.")
            return
        b_signal = signal_str.encode('utf-8')
        self.shm_signal.buf[:16] = b'\x00' * 16  # 버퍼 초기화
        self.shm_signal.buf[:len(b_signal)] = b_signal
        if self.debug:
            print(f"공유 메모리 신호가 '{signal_str}'로 업데이트되었습니다.")
        self.current_signal = signal_str
        self.last_signal_change = time.time()

    def run(self):
        self.debug_print("‘자빅스’라고 말하세요")
        while True:
            frame = self.mic_stream.getFrame()
            result = self.hotword_detector.scoreFrame(frame)
            if result is None:
                # 키워드 미감지 시, 최소 간격을 만족하면 "start" 신호 전송
                if self.current_signal != "start" and time.time() - self.last_signal_change >= self.min_interval:
                    self.set_signal("start")
                    self.debug_print("Hotword 미감지 (start signal)")
                continue

            if result.get("match"):
                # 키워드 인식 시 키워드인식할때 RMS를 전송
                self.mqtt_handler.publish_rms(result.get("rms"))

                if self.current_signal != "stop" and time.time() - self.last_signal_change >= self.min_interval:
                    self.set_signal("stop")
                    self.debug_print("Hotword 감지됨! Confidence: " + str(result.get("confidence")))
                    self.debug_print("정지 신호 전송됨.")
                # 키워드 감지 시 오디오 스트리밍 시작
                self.audio_streamer.stream()
                self.debug_print("스트리밍 종료.")
                # 스트리밍 종료 후, 키워드가 해제될 때까지 대기
                while True:
                    frame = self.mic_stream.getFrame()
                    result = self.hotword_detector.scoreFrame(frame)
                    if result is None or not result.get("match"):
                        break
                if self.current_signal != "start" and time.time() - self.last_signal_change >= self.min_interval:
                    self.set_signal("start")
                    self.debug_print("시작 신호 전송됨. 다시 키워드를 말하세요")
            else:
                if self.current_signal != "start" and time.time() - self.last_signal_change >= self.min_interval:
                    self.set_signal("start")
                    self.debug_print("Hotword 미감지. Confidence: " + str(result.get("confidence")))
                else:
                    self.debug_print("Hotword 미감지. Confidence: " + str(result.get("confidence")))

def main():
    # if debug false => no print
    mqtt_handler = MQTTClientHandler(debug=True)
    hotword_controller = HotwordController(mqtt_handler, debug=True)
    hotword_controller.run()

if __name__ == '__main__':
    main()
