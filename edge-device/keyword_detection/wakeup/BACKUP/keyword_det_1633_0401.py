import time
import json
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
from multiprocessing import shared_memory
import threading

import pygame

# .env 파일 로드
load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# ----------------------
# 상수 및 환경변수
# ----------------------
SILENCE_THRESHOLD = 5      # 5초 동안 '침묵'이면 종료
DIFF_THRESHOLD = 200       # (침묵 판정 기준) RMS 차이가 200 미만이면 조용해졌다고 판단
TIME_SETUP = 1.0           # SELECT_TOPIC 대기 시간
RMS_THRESHOLD = 1000        # RMS 임계치 이 수치 이하로 떨어지면 '침묵'으로 판단

# env
SELECT_TOPIC = os.getenv("SELECT_TOPIC")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC")
END_TOPIC = os.getenv("END_TOPIC")
TOPIC_AUDIO = os.getenv("TOPIC_AUDIO")
TOPIC_RMS = os.getenv("TOPIC_RMS")
HOT_WORD = os.getenv("HOT_WORD")
REFERENCE_FILE = os.getenv("REFERENCE_FILE")
BROKER_ADDRESS = os.getenv("BROKER_ADDRESS")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
KEYWORD_CLIENT = os.getenv("KEYWORD_CLIENT", "keyword_mqtt_client_gwangju_250")
# 사운드 파일(키워드 인식 시 재생)
BEEP_FILE = os.getenv("CHECK_MP3_FILE", "./beep.mp3")
VOLUME = float(os.getenv("MP3_VOLUME", 0.9))

# 만약 pygame 이용하여 사운드 재생 중이라면 mixer.init()
pygame.mixer.init()


def create_or_attach_shared_signal(name="signal_shm", size=16):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        shm.buf[:5] = b'start'
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
    return shm

class MQTTClientHandler:
    def __init__(self, debug=False):
        self.debug = debug
        self.broker_address = BROKER_ADDRESS
        self.broker_port = BROKER_PORT
        self.username = MQTT_USERNAME
        self.password = MQTT_PASSWORD
        self.topic_audio = TOPIC_AUDIO
        self.topic_rms = TOPIC_RMS

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=KEYWORD_CLIENT)
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
        """ 오디오 데이터 전송 """
        self.client.publish(self.topic_audio, data)

    def publish_rms(self, rms_value):
        """ 키워드 인식 시 RMS 값 전송 """
        payload = json.dumps({"keyword_rms": float(rms_value)})
        self.client.publish(self.topic_rms, payload)


class AudioStreamer:
    """
    키워드 인식 후 오디오 스트리밍. 
    * '침묵' 판정은 previous_talk_rms와 talk_rms 차이가 200 미만인 경우
    * 5초 동안 유지 시 종료
    """
    def __init__(self, mqtt_handler, debug=False,
                 format=pyaudio.paInt16, channels=1, rate=48000, chunk=2048, device_index=None):

        self.mqtt_handler = mqtt_handler
        self.debug = debug
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index

        # self.silence_threshold = RMS_THRESHOLD -> 사용하지 않음
        self.silence_duration_required = SILENCE_THRESHOLD  # 5초
        self.debug_print(f"AudioStreamer init (diff<={DIFF_THRESHOLD} → silence)")

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
            self.debug_print("Cannot Open Microphone: " + str(e))
            pa.terminate()
            return

        # select/deselect
        selection_flag = [None]   # True / False
        stop_streaming = [False]

        def wait_for_selection():
            select_event = threading.Event()

            def on_select_message(client, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode('utf-8'))
                    self.debug_print(f"Received selection message: {data}")

                    # Check 'select' field explicitly
                    if data.get("select") is True:
                        selection_flag[0] = True
                    else:
                        selection_flag[0] = False

                except Exception as e:
                    self.debug_print("Selection message parsing error: " + str(e))
                    selection_flag[0] = False

                select_event.set()


            topic = SELECT_TOPIC
            self.mqtt_handler.client.message_callback_add(topic, on_select_message)
            self.debug_print("Wait for Selection...")
            select_event.wait(timeout=TIME_SETUP)
            self.mqtt_handler.client.message_callback_remove(topic)

            if selection_flag[0] is None:
                selection_flag[0] = False

            if not selection_flag[0]:
                stop_streaming[0] = True

        def on_deselect_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode('utf-8'))
                self.debug_print("수신한 디셀렉트 메시지: " + str(data))
                if data.get("data") == "deselected_device":
                    stop_streaming[0] = True
            except Exception as e:
                self.debug_print("디셀렉트 메시지 처리 에러: " + str(e))


        selection_thread = threading.Thread(target=wait_for_selection)
        selection_thread.start()

        self.mqtt_handler.client.message_callback_add(DESELECT_TOPIC, on_deselect_message)
        self.debug_print("MQTT 음성 스트리밍 시작")

        silence_start = None
        previous_talk_rms = None  # [추가] 직전 프레임 RMS

        while True:
            if stop_streaming[0]:
                self.debug_print("선택/디셀렉트 감지 -> 스트리밍 종료")
                break

            try:
                audio_data = stream.read(self.chunk, exception_on_overflow=False)
            except Exception as e:
                self.debug_print("스트림 읽기 에러: " + str(e))
                continue

            # 오디오 전송
            self.mqtt_handler.publish_audio(audio_data)

            # talk_rms 계산
            data_arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            talk_rms = np.sqrt(np.mean(np.square(data_arr)))
            if np.isnan(talk_rms):
                self.debug_print("RMS=NaN -> 건너뜀")
                continue

            if previous_talk_rms is None:
                # 첫 프레임이면 그냥 저장
                previous_talk_rms = talk_rms
                self.debug_print(f"첫 talk_rms={talk_rms:.2f}")
                continue

            # [변경] 침묵 판정: 직전 RMS와 현재 RMS 차이가 DIFF_THRESHOLD(200) 미만
            diff_rms = abs(talk_rms - previous_talk_rms)
            print(f"RMS={talk_rms:.2f}, diff={diff_rms:.2f}")

            if diff_rms < DIFF_THRESHOLD or talk_rms < RMS_THRESHOLD:
                # 소리 차이 별로 없음 → 침묵
                if silence_start is None:
                    silence_start = time.time()
                else:
                    if time.time() - silence_start > self.silence_duration_required:
                        self.debug_print("5초간 차이 <200 → 침묵 -> 종료")
                        break
            else:
                # 차이가 충분 -> 침묵 아님
                silence_start = None

            # 매 프레임 후 previous 업데이트
            previous_talk_rms = talk_rms

        selection_thread.join()
        self.mqtt_handler.client.message_callback_remove(DESELECT_TOPIC)

        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.debug_print("MQTT 음성 스트리밍 종료")
        self.end_device()

    def end_device(self):
        topic = END_TOPIC
        payload = json.dumps(True)
        self.mqtt_handler.client.publish(topic, payload)
        self.debug_print("디바이스 연결 해제 신호 전송됨.")


class HotwordController:
    """
    키워드 인식 -> Beep 사운드, 오디오 스트리밍
    매치되기 직전 RMS vs 매치 시점 RMS 차이를 키워드 RMS로 출력
    """
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
            relaxation_time=1
        )
        self.mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=0.75, custom_channels=1)
        self.mic_stream.start_stream()

        self.shm_signal = create_or_attach_shared_signal()
        self.current_signal = "start"
        self.last_signal_change = time.time()
        self.min_interval = 1
        self.count =0
        # [추가] beep sound
        self.beep_sound = None
        if os.path.exists(BEEP_FILE):
            self.beep_sound = pygame.mixer.Sound(BEEP_FILE)
            self.beep_sound.set_volume(VOLUME)
        else:
            if self.debug:
                print(f"경고: BEEP_FILE='{BEEP_FILE}' 없음 -> 재생 불가")

        # [추가] 키워드 RMS 계산용
        self.previous_hotword_rms = 0.0  # 매치되기 직전 RMS

    def debug_print(self, msg):
        if self.debug:
            print(msg)

    def set_signal(self, signal_str):
        current = bytes(self.shm_signal.buf[:16]).split(b'\x00')[0].decode('utf-8')
        if current == signal_str:
            if self.debug:
                print(f"공유 메모리 신호가 이미 '{signal_str}'입니다. 업데이트 안 함.")
            return
        b_signal = signal_str.encode('utf-8')
        self.shm_signal.buf[:16] = b'\x00' * 16
        self.shm_signal.buf[:len(b_signal)] = b_signal
        if self.debug:
            print(f"공유 메모리 신호='{signal_str}'로 변경")
        self.current_signal = signal_str
        self.last_signal_change = time.time()

    def play_beep_immediately(self):
        if self.beep_sound:
            self.beep_sound.play(loops=0)
            #print("[Beep 사운드] 재생 시작")
        else:
            print("[Beep 사운드] 파일 없음")

    def run(self):
        self.debug_print("키워드 대기중...")
        while True:
            frame = self.mic_stream.getFrame()
            result = self.hotword_detector.scoreFrame(frame)
            # result is None -> no detection attempt or mismatch
            if result is None:
                if self.current_signal != "start" and (time.time() - self.last_signal_change >= self.min_interval):
                    self.set_signal("start")
                    self.debug_print("Hotword 미감지 (start)")
                continue

            # result not None -> detection 시도
            current_hotword_rms = result.get("rms", 0.0)
            match_flag = bool(result.get("match", False))

            if match_flag:
                # 키워드 인식

                diff_rms = abs(current_hotword_rms - self.previous_hotword_rms)

                # 1) "매치 직전 RMS"와 현재 RMS 차이 발행
                self.mqtt_handler.publish_rms(diff_rms)

                # 2) "매치 직전 RMS"와 현재 RMS 차이 출력

                print(f"[키워드 RMS] 전/현 차이: {diff_rms:.2f}")

                # 3) signal=stop
                if self.current_signal != "stop" and (time.time() - self.last_signal_change >= self.min_interval):
                    self.set_signal("stop")
                    self.debug_print(f"키워드 감지! Confidence={result.get('confidence')}, 정지 신호 전송.")

                # 4) Beep 재생 & 오디오 스트리밍 병행
                self.play_beep_immediately()
                self.audio_streamer.stream()
                self.debug_print("스트리밍 종료")

                # 스트리밍 후, hotword 해제 대기
                while True:
                    frame2 = self.mic_stream.getFrame()
                    result2 = self.hotword_detector.scoreFrame(frame2)
                    if not result2 or not result2.get("match"):
                        break

                # hotword 해제 후 start 복원
                if self.current_signal != "start" and (time.time() - self.last_signal_change >= self.min_interval):
                    self.set_signal("start")
                    self.debug_print("start 복원. 다시 키워드를 말하세요")
            else:
                # 인식 시도했으나 match=False
                if self.current_signal != "start" and (time.time() - self.last_signal_change >= self.min_interval):
                    self.set_signal("start")
                    self.debug_print(f"Hotword 미감지. conf={result.get('confidence')}")
                else:
                    self.debug_print(f"Hotword 미감지. conf={result.get('confidence')}")

            # [중요] 매치되든 아니든 -> 이전 RMS 갱신

            # 0.75간격의 프레임이니 1.5초마다 갱신
            if(self.count > 2):
                self.previous_hotword_rms = current_hotword_rms
                self.count = 0
            else:
                self.count +=1    

def main():
    mqtt_handler = MQTTClientHandler(debug=True)
    hotword_controller = HotwordController(mqtt_handler, debug=True)
    hotword_controller.run()

if __name__ == '__main__':
    main()
