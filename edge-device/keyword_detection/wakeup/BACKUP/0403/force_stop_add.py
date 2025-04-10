import time
import json
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import paho.mqtt.client as mqtt
import numpy as np
import io
import wave
import pygame  # 대기 사운드 및 TTS 재생용

# .env 파일 로드
load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')

# 환경 변수 로드
TOPIC_TTS = os.getenv("TOPIC_TTS")
AZURE_TTS_API_KEY = os.getenv("AZURE_TTS_API_KEY")
DESELECT_TOPIC = os.getenv("DESELECT_TOPIC")
# 내 디바이스가 다시 선택된 경우 강제로 음성을 종료하기 위해 사용하는 토픽
FORCE_STOP_TOPIC = os.getenv("SELECT_TOPIC")  # 기존 코드에서 SELECT_TOPIC 재활용
MP3_FILE = os.getenv("MP3_FILE")
TTS_CLIENT = os.getenv("TTS_CLIENT", "tts_mqtt_client_gwangju_250")
MP3_VOLUNE = 0.9
# 추가: TTS 음성 설정 (환경변수 사용)
TTS_VOICE = os.getenv("TTS_VOICE", "ko-KR-SunHiNeural")
# 추가: TTS 완료 토픽
TOPIC_TTS_FINISHED = os.getenv("TOPIC_TTS_FINISHED")

# pygame mixer 초기화
pygame.mixer.pre_init(frequency=48000, size=-16, channels=2)
pygame.mixer.init()

class WaitSound:
    """
    대기 사운드 (예: wait.mp3)를 재생/정지하는 클래스
    """
    def __init__(self, sound_file_path, volume=0.3):
        self.sound = pygame.mixer.Sound(sound_file_path)
        self.sound.set_volume(volume)

    def play(self):
        self.sound.play(loops=1)  # loops=-1로 하면 무한 반복

    def stop(self):
        self.sound.stop()

def tts_fast(text, tts_handler):
    """
    주어진 text를 Azure TTS로 합성 후,
    48kHz, 16비트 모노 데이터로 받아 스테레오 변환,
    Gain 적용한 뒤 메모리상에서 WAV 헤더를 씌워
    pygame으로 바로 재생. (디스크에 저장하지 않음)

    :param text: 합성할 텍스트
    :param tts_handler: MQTT_TTS_Handler 인스턴스(재생 채널 저장용)
    """
    speech_key = AZURE_TTS_API_KEY
    region = "koreacentral"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_synthesis_voice_name = TTS_VOICE

    # 48kHz, 16비트 모노 PCM 포맷으로 합성
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
    )

    audio_config = None
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,
                                              audio_config=audio_config)

    # 텍스트 → 음성 변환 (동기 호출)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ [TTS 완료] → 재생 준비")

        stream = speechsdk.AudioDataStream(result)
        audio_bytes = bytearray()

        while True:
            buffer = bytes(4096)
            read_len = stream.read_data(buffer)
            if read_len == 0:
                break
            audio_bytes.extend(buffer[:read_len])

        # 48kHz, 16비트, 모노 데이터
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Mono → Stereo 변환
        stereo = np.column_stack((audio_np, audio_np)).astype(np.float64)

        # Gain 2.0 적용
        gain = 2.0
        stereo *= gain

        # int16 범위로 변환
        stereo_amplified = np.clip(stereo, -32768, 32767).astype(np.int16)

        # TTS 시작 직전 시간 기록
        tts_start_time = time.time()

        # === [메모리상에서 WAV 헤더를 씌우기] ===
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(2)           # 스테레오
            wf.setsampwidth(2)          # 16비트(2byte)
            wf.setframerate(48000)      # 48kHz
            wf.writeframes(stereo_amplified.tobytes())

        # 버퍼의 포인터를 맨 앞으로 이동
        wav_buffer.seek(0)

        # pygame.mixer.Sound로 in-memory WAV 재생
        print("✅ [TTS 재생 시작]")
        sound_obj = pygame.mixer.Sound(wav_buffer)
        channel = sound_obj.play()

        # 현재 재생 중인 채널을 핸들러에 기록
        tts_handler.current_channel = channel

        # 재생 끝날 때까지 동기 대기
        while channel.get_busy():
            pygame.time.wait(50)  # 50ms 간격으로 체크
            # 여기서도 MQTT 콜백이 별도 스레드로 들어오면 channel.stop() 가능

        print("✅ [TTS 재생 완료]\n")
        return tts_start_time
    else:
        print(f"❌ [TTS 실패]: {result.reason}")
        return None

class MQTT_TTS_Handler:
    """
    (1) 디셀렉트 토픽 수신 시각,
    (2) 텍스트 토픽 수신 시각,
    (3) TTS 오디오 재생 시작 시각을 기록하고, 
    TTS 직전까지 대기 사운드(wait.mp3)를 pygame으로 재생.
    
    또한 FORCE_STOP_TOPIC 수신 시 현재 재생 중인 TTS를 즉시 중단.
    """
    def __init__(self, broker_address, broker_port=1883,
                 username=None, password=None, topic=TOPIC_TTS):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.text_topic = topic
        self.deselect_topic = DESELECT_TOPIC
        self.force_stop_topic = FORCE_STOP_TOPIC

        # 최근 메시지 수신 시각
        self.deselect_received_time = None
        self.text_received_time = None
        self.tts_start_time = None

        # 대기 사운드 로드 (pygame으로 mp3)
        self.wait_sound = WaitSound(MP3_FILE, volume=MP3_VOLUNE)

        # 현재 재생 중인 TTS 채널(없으면 None)
        self.current_channel = None

        # MQTT 클라이언트 생성
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=TTS_CLIENT)
        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_text_message
        # 토픽별 콜백 분기
        self.client.message_callback_add(self.deselect_topic, self.on_deselect_message)
        # 강제 중단 콜백 연결
        if self.force_stop_topic:
            self.client.message_callback_add(self.force_stop_topic, self.on_force_stop_message)

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print("Connected to MQTT broker, status code:", reason_code)
        # 텍스트 토픽 구독
        client.subscribe(self.text_topic)
        print(f"Subscribed to text topic: {self.text_topic}")
        # 디셀렉트 토픽 구독
        client.subscribe(self.deselect_topic)
        print(f"Subscribed to deselect topic: {self.deselect_topic}")
        # 강제중단 토픽 구독
        if self.force_stop_topic:
            client.subscribe(self.force_stop_topic)
            print(f"Subscribed to force stop topic: {self.force_stop_topic}")

    def on_text_message(self, client, userdata, msg):
        """
        text_topic에 대한 기본 콜백이므로,
        JSON 안의 "data" 필드를 추출해서 TTS 재생을 진행.
        """
        # force_stopped 등 별도 처리 로직이 필요하다면 여기에 구현 가능
        if msg.topic == self.text_topic:
            try:
                payload = msg.payload.decode('utf-8')
                json_payload = json.loads(payload)
                text = json_payload.get("data", "")
                print(f"\n[Text Message Received] Topic: {msg.topic}, Data: {text}")

                # 텍스트 수신 시각 기록
                self.text_received_time = time.time()

                # 대기 사운드 정지
                self.wait_sound.stop()

                # TTS 실행
                start_time = tts_fast(text, self)
                if start_time is not None:
                    self.tts_start_time = start_time

                    # TTS 재생 완료 후 피니쉬 토픽에 완료 메시지 발행
                    if TOPIC_TTS_FINISHED:
                        finish_payload = json.dumps({
                            "status": "TTS_FINISHED",
                            "tts_start_time": start_time
                        })
                        self.client.publish(TOPIC_TTS_FINISHED, finish_payload)
                        print(f"MQTT 발행: {TOPIC_TTS_FINISHED} 토픽에 TTS 완료 메시지 발행")

            except Exception as e:
                print("Error processing text message:", e)

    def on_deselect_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            print(f"[Deselect Message Received] Topic: {msg.topic}, Payload: {payload}")

            self.deselect_received_time = time.time()

            # 대기 사운드 재생
            self.wait_sound.play()

        except Exception as e:
            print("Error processing deselect message:", e)

    def on_force_stop_message(self, client, userdata, msg):
        """
        FORCE_STOP_TOPIC 구독 콜백:
        현재 재생 중인 TTS를 즉시 중단하고, 이후 메시지도 무시하고 싶다면
        추가 로직을 넣을 수 있음.
        """
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            print(f"Received selection message: {data}")
            if data.get("select") == "True":
                if self.current_channel:  # None인지 확인
                    print("🔴 강제 중단: 현재 재생 중인 TTS를 즉시 중지합니다.")
                    self.current_channel.stop()
                    self.current_channel = None  # 사용 후에는 None으로 초기화
                #else:
                    #print("⚠️ TTS 채널이 None이므로, 현재 재생 중이 아닙니다.")

        except Exception as e:
            print("Error processing force stop message:", e)

    def _print_time_info(self):
        """
        디버그용으로 3개의 시각(deselect, text, TTS start)을 확인하는 함수
        """
        if self.deselect_received_time and self.text_received_time and self.tts_start_time:
            d_time = self.deselect_received_time
            t_time = self.text_received_time
            s_time = self.tts_start_time

            diff_1_2 = t_time - d_time
            diff_2_3 = s_time - t_time
            diff_1_3 = s_time - d_time

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
