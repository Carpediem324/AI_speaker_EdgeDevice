import time
import json
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import paho.mqtt.client as mqtt
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly  # 리샘플링을 위한 모듈 임포트
import soundfile as sf  # 파일 저장 테스트용

# .env 파일 로드
load_dotenv()

# 환경 변수 로드
TOPIC_TTS = os.getenv("TOPIC_TTS")
AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")

def tts_fast(text):
    speech_key = AZURE_TTS_API_KEY
    region = "koreacentral"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_synthesis_voice_name = "ko-KR-SunHiNeural"
    # 원본 출력 포맷: 16kHz, 16비트, Mono PCM
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    )

    # 자동 스피커 재생 방지를 위해 audio_config에 None 전달
    audio_config = None
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ [TTS 완료] → 재생 시작")

        stream = speechsdk.AudioDataStream(result)
        audio_bytes = bytearray()
        while True:
            buffer = bytes(4096)
            read_len = stream.read_data(buffer)
            if read_len == 0:
                break
            audio_bytes.extend(buffer[:read_len])

        # 원본 16bit PCM 데이터 읽기 (샘플레이트 16000Hz, mono)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        print("원본 audio_np shape:", audio_np.shape)
        print("원본 audio_np min:", np.min(audio_np), "max:", np.max(audio_np))

        # Mono 데이터를 Stereo로 변환 (양 채널 동일 데이터)
        stereo = np.column_stack((audio_np, audio_np))
        print("stereo shape:", stereo.shape)

        # 리샘플링 전에 float64로 변환 (값 보존을 위해)
        stereo_float = stereo.astype(np.float64)

        # 16kHz → 48000Hz로 리샘플링 (업샘플링 3배)
        stereo_resampled = resample_poly(stereo_float, up=3, down=1, axis=0)
        print("stereo_resampled shape:", stereo_resampled.shape)
        print("리샘플링 후 min/max (float):", np.min(stereo_resampled), np.max(stereo_resampled))

        # gain 적용 (여기서는 2.0배)
        gain = 2.0
        stereo_resampled = stereo_resampled * gain

        # 리샘플링 후 데이터는 float64이므로 int16 범위로 복원
        stereo_resampled = np.clip(stereo_resampled, -32768, 32767).astype(np.int16)
        print("최종 stereo_resampled dtype:", stereo_resampled.dtype)
        print("최종 min/max:", np.min(stereo_resampled), np.max(stereo_resampled))

        # 파일로 저장 (외부 플레이어 테스트)
        #sf.write("tts_output_stereo.wav", stereo_resampled, 48000)
        #print("tts_output_stereo.wav 파일 저장 완료")

        # 리샘플링된 데이터를 48000Hz, Stereo로 재생 (device 인덱스 사용)
        try:
            #sd.query_devices()로 실제 UACDemoV10 장치 인덱스를 확인 후 device 값을 수정하세요.
            sd.play(stereo_resampled, samplerate=48000, device=None)
            sd.wait()
            print("✅ [재생 완료]")
        except Exception as e:
            print("재생 중 에러 발생:", e)
    else:
        print(f"❌ [TTS 실패]: {result.reason}")

# 필요 시 wait_sound 클래스(추후 기능 확장을 위한 예제)
class WaitSound:
    def stop(self):
        print("Waiting sound stopped.")

# MQTT TTS 클라이언트 핸들러 클래스
class MQTT_TTS_Handler:
    def __init__(self, broker_address, broker_port=1883, username=None, password=None, topic=TOPIC_TTS):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.topic = topic
        # CallbackAPIVersion.VERSION2 사용하여 MQTT 클라이언트 생성
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "tts_mqtt_client")
        if username and password:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print("Connected to MQTT broker, status code:", reason_code)
        client.subscribe(self.topic)
        print(f"Subscribed to topic {self.topic}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            print(f"Received message [{msg.topic}]: {payload}")
            tts_fast(payload)
        except Exception as e:
            print("Error processing message:", e)

    def start(self):
        self.client.connect(self.broker_address, self.broker_port)
        self.client.loop_forever()

def main():
    broker_address = os.getenv("BROKER_ADDRESS")
    broker_port = int(os.getenv("BROKER_PORT", 1883))
    mqtt_username = os.getenv("MQTT_USERNAME")
    mqtt_password = os.getenv("MQTT_PASSWORD")
    
    tts_handler = MQTT_TTS_Handler(broker_address, broker_port, mqtt_username, mqtt_password, topic=TOPIC_TTS)
    tts_handler.start()

if __name__ == '__main__':
    main()
