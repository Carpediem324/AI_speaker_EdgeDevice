import time
import json
import os
import base64
import threading
import io
import numpy as np
import wave

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import paho.mqtt.client as mqtt

load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')

BROKER_ADDRESS = os.getenv("BROKER_ADDRESS")
BROKER_PORT = int(os.getenv("BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")
TTS_VOICE = os.getenv("TTS_VOICE", "ko-KR-SunHiNeural")
REGION = "koreacentral"  # Azure Speech TTS 지역

TTS_CLIENT = os.getenv("TTS_CLIENT", "tts_mqtt_client")

# 여러 디바이스에 대한 TTS 요청을 한 번에 구독하려면 와일드카드 사용
TOPIC_TTS_WILDCARD = os.getenv("TOPIC_TTS_WILDCARD", "/jabix/response/+/+/+/signal/TTS")

########################################################################
# Azure TTS 함수
########################################################################
def azure_tts_to_wav_bytes(text):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_TTS_API_KEY, region=REGION)
    speech_config.speech_synthesis_voice_name = TTS_VOICE
    # 48kHz, 16bit 모노
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
    )

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ [TTS 합성 완료] WAV 바이트 생성 중...")

        # 결과 PCM 데이터 추출
        stream = speechsdk.AudioDataStream(result)
        audio_bytes = bytearray()
        while True:
            buffer = bytes(4096)
            read_len = stream.read_data(buffer)
            if read_len == 0:
                break
            audio_bytes.extend(buffer[:read_len])

        # 모노 → 스테레오 + 게인(2.0)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        stereo = np.column_stack((audio_np, audio_np)).astype(np.float64)
        stereo *= 2.0
        stereo_i16 = np.clip(stereo, -32768, 32767).astype(np.int16)

        # WAV 헤더 추가
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(stereo_i16.tobytes())

        return wav_buf.getvalue()
    else:
        print("❌ [TTS 실패]", result.reason)
        return None

########################################################################
# MQTT Handler
########################################################################
class MQTT_TextToTTSHandler:
    """
    1) /jabix/response/+/+/+/signal/TTS 와일드카드 구독
    2) 각 메시지 topic을 분석해, TTS → audio_data 토픽에 퍼블리시
    """
    def __init__(self):
        self.broker_address = BROKER_ADDRESS
        self.broker_port = BROKER_PORT

        self.client = mqtt.Client(client_id=TTS_CLIENT)
        if MQTT_USERNAME and MQTT_PASSWORD:
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message_catchall

    def on_connect(self, client, userdata, flags, reason_code):
        print("Connected to MQTT broker:", reason_code)
        # 와일드카드 토픽 구독
        client.subscribe(TOPIC_TTS_WILDCARD)
        client.message_callback_add(TOPIC_TTS_WILDCARD, self.on_tts_message)
        print(f"Subscribed to TTS wildcard: {TOPIC_TTS_WILDCARD}")

    def on_message_catchall(self, client, userdata, msg):
        # 디버그용
        # print("CatchAll ->", msg.topic, msg.payload)
        pass

    def on_tts_message(self, client, userdata, msg):
        """
        (예) msg.topic = /jabix/response/S101/aircon/M001/signal/TTS
        1) JSON에서 'data' 필드 추출
        2) TTS 생성
        3) audio_data 토픽 = msg.topic.replace('/TTS','/audio_data')
        4) Base64 + MQTT Publish
        """
        try:
            payload_str = msg.payload.decode('utf-8')
            payload_json = json.loads(payload_str)
            text = payload_json.get("data", "")
            print(f"\n[TTS Message] Topic={msg.topic}, Text={text}")

            # 별도 스레드에서 TTS 처리
            t = threading.Thread(target=self.handle_tts, args=(msg.topic, text))
            t.start()

        except Exception as e:
            print("Error in on_tts_message:", e)

    def handle_tts(self, tts_topic, text):
        # 1) TTS 변환
        wav_data = azure_tts_to_wav_bytes(text)
        if not wav_data:
            return

        # 2) "TTS" -> "audio_data" 치환
        # ex) /jabix/response/S101/aircon/M001/signal/TTS -> /jabix/response/S101/aircon/M001/signal/audio_data
        audio_topic = tts_topic.replace("/TTS", "/audio_data")

        # 3) Base64 인코딩 + 퍼블리시
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
        payload = {
            "audio_data_base64": audio_b64,
            "timestamp": time.time()
        }
        payload_str = json.dumps(payload)

        self.client.publish(audio_topic, payload_str)
        print(f"✅ 오디오 퍼블리시 -> {audio_topic}")

    def start(self):
        self.client.connect(self.broker_address, self.broker_port)
        self.client.loop_start()
        print("MQTT loop started in background.")
        while True:
            time.sleep(1)

def main():
    handler = MQTT_TextToTTSHandler()
    handler.start()

if __name__ == '__main__':
    main()
