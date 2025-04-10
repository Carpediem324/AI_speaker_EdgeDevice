import time
import json
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import paho.mqtt.client as mqtt
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly  # 리샘플링
import soundfile as sf  # 파일 저장(테스트용)
import pygame  # 대기 사운드 재생용

# .env 파일 로드
load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')

# 환경 변수 로드
TOPIC_TTS = os.getenv("TOPIC_TTS")
AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC")
MP3_FILE = os.getenv("MP3_FILE")
TTS_CLIENT = os.getenv("TTS_CLIENT", "tts_mqtt_client_gwangju_250")
MP3_VOLUNE = 0.9
# pygame mixer 초기화
pygame.mixer.init()

class WaitSound:
    """
    대기 사운드 (예: wait.mp3)를 재생/정지하는 클래스
    """
    def __init__(self, sound_file_path, volume=0.3):
        # pygame.mixer.Sound 객체 생성
        self.sound = pygame.mixer.Sound(sound_file_path)
        self.sound.set_volume(volume)

    def play(self):
        self.sound.play(loops=2)  # loops=-1로 설정하면 무한 반복
        #print("대기 사운드 재생 시작")

    def stop(self):
        self.sound.stop()
        #print("대기 사운드 재생 정지")


def tts_fast(text):
    """
    주어진 text를 Azure TTS로 합성 후,
    16kHz → 48kHz 리샘플링, Gain 적용, Stereo 변환하여 재생.
    
    실제 오디오 재생을 시작하기 직전의 시간(시스템 epoch)을 반환.
    """
    speech_key = AZURE_TTS_API_KEY
    region = "koreacentral"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_synthesis_voice_name = "ko-KR-SunHiNeural"

    # 원본 출력 포맷: 16kHz, 16비트, Mono PCM
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    )

    # 자동 스피커 재생 방지를 위해 audio_config에 None
    audio_config = None
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # 텍스트 → 음성 변환 (동기 호출)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # ---------------------------
        # (TTS 합성 완료 시점)
        # ---------------------------
        print("✅ [TTS 완료] → 재생 준비")

        stream = speechsdk.AudioDataStream(result)
        audio_bytes = bytearray()
        while True:
            buffer = bytes(4096)
            read_len = stream.read_data(buffer)
            if read_len == 0:
                break
            audio_bytes.extend(buffer[:read_len])

        # 원본 16bit PCM 데이터(샘플레이트 16000Hz, mono)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Mono → Stereo 변환 (좌우 채널 동일)
        stereo = np.column_stack((audio_np, audio_np))

        # float64 변환 후 16kHz → 48kHz 업샘플링
        stereo_float = stereo.astype(np.float64)
        stereo_resampled = resample_poly(stereo_float, up=3, down=1, axis=0)

        # Gain 적용 (예: 2배)
        gain = 2.0
        stereo_resampled *= gain

        # int16 범위로 변환
        stereo_resampled = np.clip(stereo_resampled, -32768, 32767).astype(np.int16)

        # TTS 시작 직전 시간 기록
        tts_start_time = time.time()
        
        print("✅ [TTS 재생 시작]")
        try:
            sd.play(stereo_resampled, samplerate=48000, device=None)
            sd.wait()  # 재생이 끝날 때까지 대기
            print("✅ [TTS 재생 완료]\n")
        except Exception as e:
            print("재생 중 에러 발생:", e)

        # TTS를 시작한 시점 반환
        return tts_start_time
    else:
        print(f"❌ [TTS 실패]: {result.reason}")
        return None


class MQTT_TTS_Handler:
    """
    (1) 디셀렉트 토픽 수신 시점
    (2) 텍스트 토픽 수신 시점
    (3) TTS 오디오 재생 시작 시점
    을 기록하고, TTS 직전까지 대기 사운드(wait.mp3)를 재생.
    """
    def __init__(self, broker_address, broker_port=1883, username=None, password=None, topic=TOPIC_TTS):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.text_topic = topic
        self.deselect_topic = DESELECT_TOPIC

        # 최근 메시지 수신 시각
        self.deselect_received_time = None  # (1)
        self.text_received_time = None      # (2)
        self.tts_start_time = None          # (3)

        # 대기 사운드 로드 (경로는 상황에 맞게 수정)
        self.wait_sound = WaitSound(MP3_FILE, volume=MP3_VOLUNE)

        # MQTT 클라이언트
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=TTS_CLIENT)
        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_text_message  # 기본 on_message -> 텍스트 토픽
        self.client.message_callback_add(self.deselect_topic, self.on_deselect_message)  # 디셀렉트 토픽 콜백

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print("Connected to MQTT broker, status code:", reason_code)
        # 구독
        client.subscribe(self.text_topic)
        print(f"Subscribed to text topic: {self.text_topic}")
        client.subscribe(self.deselect_topic)
        print(f"Subscribed to deselect topic: {self.deselect_topic}")

    def on_text_message(self, client, userdata, msg):
        """
        텍스트 토픽 수신 시
         - 대기 사운드 정지
         - 텍스트 수신 시각 기록
         - TTS 실행 (JSON 메시지에서 data 필드만 사용)
         - TTS 시작 시각 기록
         - 시각 정보 출력
        """
        try:
            payload = msg.payload.decode('utf-8')
            json_payload = json.loads(payload)
            text = json_payload.get("data", "")
            print(f"\n[Text Message Received] Topic: {msg.topic}, Data: {text}")

            # (2) 텍스트 수신 시각
            self.text_received_time = time.time()

            # 대기 사운드 정지
            self.wait_sound.stop()

            # TTS 실행 (data 필드만 전달)
            start_time = tts_fast(text)
            if start_time is not None:
                self.tts_start_time = start_time

            # 정보 출력
            #self._print_time_info()

        except Exception as e:
            print("Error processing text message:", e)

    def on_deselect_message(self, client, userdata, msg):
        """
        디셀렉트 토픽 수신 시
         - (1) 디셀렉트 시각 기록
         - 대기 사운드 재생 시작
        """
        try:
            payload = msg.payload.decode('utf-8')
            print(f"[Deselect Message Received] Topic: {msg.topic}, Payload: {payload}")

            self.deselect_received_time = time.time()

            # 대기 사운드 재생
            self.wait_sound.play()

        except Exception as e:
            print("Error processing deselect message:", e)

    def _print_time_info(self):
        """
        3개 시각이 모두 존재하면 시간을 표시하고
        (1→2), (2→3), (1→3) 시간차를 계산
        """
        if self.deselect_received_time and self.text_received_time and self.tts_start_time:
            d_time = self.deselect_received_time
            t_time = self.text_received_time
            s_time = self.tts_start_time

            diff_1_2 = t_time - d_time  # (텍스트 - 디셀렉트)
            diff_2_3 = s_time - t_time  # (TTS 시작 - 텍스트)
            diff_1_3 = s_time - d_time  # (TTS 시작 - 디셀렉트)

            print("\n=== [Time Info] ===")
            print(f"1) Deselect Received Time  = {d_time:.4f}")
            print(f"2) Text Received Time      = {t_time:.4f}")
            print(f"3) TTS Started Time        = {s_time:.4f}")
            print(f"(1→2) = {diff_1_2:.4f} sec | (2→3) = {diff_2_3:.4f} sec | (1→3) = {diff_1_3:.4f} sec\n")
        else:
            print("아직 3개의 시각(deselect, text, TTS start) 중 일부가 설정되지 않았습니다.")

    def start(self):
        self.client.connect(self.broker_address, self.broker_port)
        self.client.loop_forever()


def main():
    broker_address = os.getenv("BROKER_ADDRESS")
    broker_port = int(os.getenv("BROKER_PORT", 1883))
    mqtt_username = os.getenv("MQTT_USERNAME")
    mqtt_password = os.getenv("MQTT_PASSWORD")

    tts_handler = MQTT_TTS_Handler(
        broker_address, 
        broker_port, 
        mqtt_username, 
        mqtt_password, 
        topic=TOPIC_TTS
    )
    tts_handler.start()

if __name__ == '__main__':
    main()
