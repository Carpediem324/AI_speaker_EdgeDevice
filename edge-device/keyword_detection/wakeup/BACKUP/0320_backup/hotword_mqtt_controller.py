import time
import json
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

class MQTTClientHandler:
    def __init__(self, debug=False):
        load_dotenv()
        self.debug = debug
        self.broker_address = os.getenv("BROKER_ADDRESS")
        self.broker_port = int(os.getenv("BROKER_PORT"))
        self.username = os.getenv("MQTT_USERNAME")
        self.password = os.getenv("MQTT_PASSWORD")
        self.topic_audio = os.getenv("TOPIC_AUDIO")
        self.topic_rms = os.getenv("TOPIC_RMS")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_audio_pub")
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address, self.broker_port)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        if self.debug:
            print(f"MQTT 브로커 연결 완료, 상태 코드: {reason_code}")

    def publish_audio(self, data):
        self.client.publish(self.topic_audio, data)

    def publish_rms(self, rms_value):
        payload = json.dumps({"keywork_rms": float(rms_value)})
        self.client.publish(self.topic_rms, payload)


class AudioStreamer:
    def __init__(self, mqtt_handler, debug=False,
                 format=pyaudio.paInt16, channels=2, rate=48000, chunk=2048, device_index=None):
        self.mqtt_handler = mqtt_handler
        self.debug = debug
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index

        # 스트리밍 중 침묵 판단을 위한 임계값 설정
        self.silence_threshold = 200       # RMS 임계치 (환경에 따라 조정)
        self.silence_duration_required = 2.0 # 2초 침묵 시 종료

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
            self.debug_print("MQTT 스트리밍용 마이크 열기 실패: " + str(e))
            pa.terminate()
            return

        self.debug_print("MQTT 음성 스트리밍 시작")
        silence_start = None

        while True:
            try:
                audio_data = stream.read(self.chunk, exception_on_overflow=False)
            except Exception as e:
                self.debug_print("스트림 읽기 에러: " + str(e))
                continue

            # 오디오 데이터를 MQTT 토픽으로 발행
            self.mqtt_handler.publish_audio(audio_data)

            # RMS 계산
            data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            talk_rms = np.sqrt(np.mean(np.square(data)))
            if np.isnan(talk_rms):
                self.debug_print("RMS 값이 NaN입니다. 이번 프레임 건너뜁니다.")
                continue

            print("RMS: " + str(talk_rms))
            if talk_rms < self.silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.silence_duration_required:
                    self.debug_print("2초간 침묵 감지, MQTT 스트리밍 종료")
                    break
            else:
                silence_start = None

        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.debug_print("MQTT 음성 스트리밍 종료")


class HotwordController:
    def __init__(self, mqtt_handler, debug=False):
        self.debug = debug
        self.mqtt_handler = mqtt_handler
        self.audio_streamer = AudioStreamer(mqtt_handler, debug=debug)
        # 핫워드 감지를 위한 모델 초기화
        self.base_model = Resnet50_Arc_loss()
        self.hotword_detector = HotwordDetector(
            hotword="test",  # 감지할 키워드
            model=self.base_model,
            reference_file="/home/ubuntu/S12P21S001/edge-device/keyword_detection/refer/hibixby_ref.json",
            threshold=0.7,
            relaxation_time=2
        )
        self.mic_stream = SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.75,
        )
        self.mic_stream.start_stream()

    def debug_print(self, msg):
        if self.debug:
            print(msg)

    def run(self):
        self.debug_print("‘하이 빅스비’라고 말하세요")
        while True:
            frame = self.mic_stream.getFrame()
            result = self.hotword_detector.scoreFrame(frame)
            if result is None:
                continue

            # 핫워드 감지와 함께 RMS 값 발행
            if result.get("match") and "rms" in result:
                keyword_rms = float(result["rms"])
                self.debug_print("Keyword RMS in result: " + str(keyword_rms))
                self.mqtt_handler.publish_rms(keyword_rms)

            if result.get("match"):
                self.debug_print("Hotword 감지됨! Confidence: " + str(result.get("confidence")))
                # 핫워드가 감지되면 오디오 스트리밍 시작
                self.audio_streamer.stream()
                self.debug_print("스트리밍 종료. 다시 ‘하이빅스비’라고 말하세요")
            else:
                self.debug_print("Hotword 미감지. Confidence: " + str(result.get("confidence")))


def main():
    mqtt_handler = MQTTClientHandler(debug=True)
    hotword_controller = HotwordController(mqtt_handler, debug=True)
    hotword_controller.run()


if __name__ == '__main__':
    main()
