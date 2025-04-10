'''
2025 03 19 11:00
마이크 오디오 스트리밍 및 디바이스 선택 기능 통합 코드
'''
import os
import json
import time
import threading
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import pyaudio

# .env 파일에서 환경 변수 로드
load_dotenv()

# MQTT 브로커 관련 환경 변수
BROKER_ADDRESS = os.getenv("BROKER_ADDRESS")
BROKER_PORT = int(os.getenv("BROKER_PORT"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# 토픽 설정
# 오디오 데이터 전송 토픽 (예: "devices/device1/audio")
TOPIC_AUDIO = os.getenv("TOPIC_AUDIO")
# RMS 데이터 토픽 (예: "devices/+/audio/rms" 와 같이 와일드카드를 사용할 수 있음)
TOPIC_RMS = os.getenv("TOPIC_RMS", "devices/+/audio/rms")
SELECT_TOPIC = os.getenv("SELECT_TOPIC", "/jabix/reponse/room1/aircon/M001/signal/select")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC", "/jabix/request/room1/aircon/M001/signal/end")

# 오디오 설정 (마이크와 스피커 모두 동일한 형식 사용)
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 2048

# PyAudio 초기화 및 출력 장치 선택
audio = pyaudio.PyAudio()
print("사용 가능한 출력 장치:")
for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print(f"인덱스 {i}: {dev['name']}")
# 출력 장치 인덱스 (기본 출력 장치를 사용하려면 None)
output_device_index = None  # 특정 장치를 사용하려면 예: 0

stream_out = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    frames_per_buffer=CHUNK,
    output_device_index=output_device_index,
)

class MQTTCombinedHandler:
    def __init__(self, debug=False, aggregation_interval=2.0):
        self.debug = debug
        # 최대 RMS 값 및 선택된 디바이스 정보를 저장할 변수와 동기화 Lock
        self.current_max = 0.0
        self.selected_device = None
        self.lock = threading.Lock()
        self.aggregation_interval = aggregation_interval

        # 통합 MQTT 클라이언트 생성 (클라이언트 ID: python_combined_sub)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_combined_sub")
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(BROKER_ADDRESS, BROKER_PORT)
        self.client.loop_start()

        # 주기적으로 최대 RMS 값을 전송하는 스레드 시작
        self.publisher_thread = threading.Thread(target=self.publish_max_periodically)
        self.publisher_thread.daemon = True
        self.publisher_thread.start()

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if self.debug:
            print(f"MQTT 브로커에 연결됨, 상태 코드: {reason_code}")
        # 오디오 데이터와 RMS 데이터 모두 구독
        client.subscribe(TOPIC_AUDIO)
        if self.debug:
            print(f"구독 시작: {TOPIC_AUDIO}")
        client.subscribe(TOPIC_RMS)
        if self.debug:
            print(f"구독 시작: {TOPIC_RMS}")

    def on_message(self, client, userdata, msg):
        # 메시지 토픽에 따라 오디오 재생 또는 RMS 데이터 처리
        if msg.topic == TOPIC_AUDIO:
            # 오디오 데이터 스트리밍 (바이너리 데이터 그대로 출력)
            try:
                stream_out.write(msg.payload)
            except Exception as e:
                if self.debug:
                    print("오디오 재생 에러:", e)
        elif msg.topic.endswith("/rms"):
            # RMS 데이터는 JSON 형식
            try:
                payload = msg.payload.decode('utf-8')
                data = json.loads(payload)
                # 토픽에서 device_id 추출 (예: "devices/device1/audio/rms")
                topic_parts = msg.topic.split('/')
                device_id = topic_parts[1] if len(topic_parts) > 1 else "unknown"
                # JSON 데이터에서 RMS 값 추출 (키: "keyword_rms")
                rms_value = float(data.get("keyword_rms", 0))
                if self.debug:
                    print(f"받은 RMS 값: {rms_value} from device: {device_id}")
                    print("RMS 데이터:", data)
                # 최대 RMS 값 업데이트
                with self.lock:
                    if rms_value > self.current_max:
                        self.current_max = rms_value
                        self.selected_device = device_id
            except Exception as e:
                if self.debug:
                    print("RMS 데이터 처리 에러:", e)
        else:
            if self.debug:
                print("알 수 없는 토픽:", msg.topic)

    def publish_select(self):
        """
        선택 신호를 SELECT_TOPIC에 전송합니다.
        선택 신호는 선택된 device_id와 최대 RMS 값을 포함하는 JSON 형태입니다.
        예) { "selected_device": "device1", "max_rms": 234.56 }
        """
        payload = json.dumps({
            "selected_device": self.selected_device,
            "max_rms": self.current_max
        })
        self.client.publish(SELECT_TOPIC, payload)
        if self.debug:
            print(f"전송된 셀렉트 토픽({SELECT_TOPIC}): {payload}")

    def publish_max_periodically(self):
        while True:
            time.sleep(self.aggregation_interval)
            with self.lock:
                if self.current_max > 0 and self.selected_device is not None:
                    self.publish_select()
                    # 전송 후 초기화
                    self.current_max = 0.0
                    self.selected_device = None

    def cleanup(self):
        # MQTT 클라이언트 중지 및 연결 해제
        self.client.loop_stop()
        self.client.disconnect()

def main():
    debug = True
    handler = MQTTCombinedHandler(debug=debug)
    try:
        # 프로그램 종료 전까지 무한 대기
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if debug:
            print("프로그램 종료")
        handler.cleanup()
        stream_out.stop_stream()
        stream_out.close()
        audio.terminate()

if __name__ == '__main__':
    main()
