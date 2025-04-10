import time
import json
import os
import base64
import io
import wave
import numpy as np
import threading
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import pygame

load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')

BROKER_ADDRESS = os.getenv("BROKER_ADDRESS")
BROKER_PORT = int(os.getenv("BROKER_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

TOPIC_AUDIO_DATA = os.getenv("TOPIC_AUDIO_DATA", "topic_audio_data")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC")
FORCE_STOP_TOPIC = os.getenv("SELECT_TOPIC")
MP3_FILE = os.getenv("MP3_FILE")
MP3_VOLUME = 0.9

AUDIO_PLAYER_CLIENT = os.getenv("AUDIO_PLAYER_CLIENT", "audio_player_client_01")
TOPIC_TTS_FINISHED = os.getenv("TOPIC_TTS_FINISHED")  # TTS 재생 완료 토픽

pygame.mixer.pre_init(frequency=48000, size=-16, channels=2)
pygame.mixer.init()

class WaitSound:
    def __init__(self, sound_file_path, volume=0.3):
        self.sound = pygame.mixer.Sound(sound_file_path)
        self.sound.set_volume(volume)

    def play(self):
        self.sound.play(loops=1)

    def stop(self):
        self.sound.stop()

class MQTT_AudioPlayerHandler:
    def __init__(self):
        self.broker_address = BROKER_ADDRESS
        self.broker_port = BROKER_PORT
        self.current_channel = None  # 현재 재생 중인 채널

        self.wait_sound = WaitSound(MP3_FILE, volume=MP3_VOLUME) if MP3_FILE else None

        self.client = mqtt.Client(client_id=AUDIO_PLAYER_CLIENT)
        if MQTT_USERNAME and MQTT_PASSWORD:
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message_catchall

        self.istalking = False

    def on_connect(self, client, userdata, flags, reason_code):
        print("Connected to MQTT broker, code:", reason_code)

        # 오디오 토픽 구독
        client.subscribe(TOPIC_AUDIO_DATA)
        client.message_callback_add(TOPIC_AUDIO_DATA, self.on_audio_data_message)
        print("Subscribed audio:", TOPIC_AUDIO_DATA)

        # 대기사운드 토픽 구독
        if DESELECT_TOPIC:
            client.subscribe(DESELECT_TOPIC)
            client.message_callback_add(DESELECT_TOPIC, self.on_deselect_message)
            print("Subscribed deselect:", DESELECT_TOPIC)

        # 강제 중단 토픽 구독
        if FORCE_STOP_TOPIC:
            client.subscribe(FORCE_STOP_TOPIC)
            client.message_callback_add(FORCE_STOP_TOPIC, self.on_force_stop_message)
            print("Subscribed force stop:", FORCE_STOP_TOPIC)

    def on_message_catchall(self, client, userdata, msg):
        pass

    def on_audio_data_message(self, client, userdata, msg):
        """
        오디오 데이터(Base64 WAV) 수신 -> 별도 스레드에서 재생 처리
        """
        try:
            # 음성이 들어오면 대기사운드 먼저 중지
            if self.wait_sound:
                self.wait_sound.stop()

            payload_str = msg.payload.decode('utf-8')
            payload = json.loads(payload_str)
            audio_b64 = payload.get("audio_data_base64")
            if not audio_b64:
                print("No audio data!")
                return

            wav_data = base64.b64decode(audio_b64)
            # 별도 스레드에서 재생 (블로킹)
            t = threading.Thread(target=self.play_wav_bytes, args=(wav_data,))
            t.start()

        except Exception as e:
            print("Error in on_audio_data_message:", e)

    def play_wav_bytes(self, wav_data):
        """
        실제 오디오 재생 (이 함수를 별도 스레드에서 동기 블로킹)
        """
        if not wav_data:
            return

        try:
            self.istalking = True

            wav_buf = io.BytesIO(wav_data)
            sound_obj = pygame.mixer.Sound(wav_buf)
            channel = sound_obj.play()

            # 현재 재생 채널 갱신
            self.current_channel = channel
            print("✅ [오디오 재생 시작] (스레드)")

            # 재생 끝날 때까지 대기
            while channel.get_busy():
                pygame.time.wait(50)

            print("✅ [오디오 재생 완료] (스레드)")

            # 재생 완료 후 TTS 완료 토픽 발행
            if TOPIC_TTS_FINISHED:
                finished_msg = json.dumps({"status": "TTS_FINISHED"})
                self.client.publish(TOPIC_TTS_FINISHED, finished_msg)
                print(f"Published TTS finished to topic: {TOPIC_TTS_FINISHED}")

            self.istalking = False
            self.current_channel = None

        except Exception as e:
            print("오디오 재생 오류:", e)

    def on_deselect_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            print("[Deselect] ", payload)
            if self.wait_sound:
                self.wait_sound.play()
        except Exception as e:
            print("Error in on_deselect_message:", e)

    def on_force_stop_message(self, client, userdata, msg):
        """
        강제 중단 토픽 -> 현재 재생 중인 음성을 즉시 stop()
        """
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            if data.get("select") == "True" and self.istalking == True:
                print("[ForceStop] ", data)
                if self.current_channel:
                    print("🔴 재생 중단: current_channel.stop()")
                    self.current_channel.stop()
                    self.current_channel = None
        except Exception as e:
            print("Error in on_force_stop_message:", e)

    def start(self):
        self.client.connect(self.broker_address, self.broker_port)
        # 백그라운드 스레드에서 MQTT 수신
        self.client.loop_start()
        print("MQTT loop started in background.")

        # 메인 스레드는 계속 유지
        while True:
            time.sleep(1)

def main():
    player = MQTT_AudioPlayerHandler()
    player.start()

if __name__ == '__main__':
    main()
