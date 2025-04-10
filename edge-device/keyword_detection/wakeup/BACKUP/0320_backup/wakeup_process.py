import time
import threading
import gc
import json
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# ===== 디버그 출력 제어 변수 =====
DEBUG = True  # True면 print 출력, False면 출력 안 함

def debug_print(msg):
    if DEBUG:
        print(msg)

# ===== 환경변수 로드 =====
load_dotenv()

# ===== MQTT 설정 (환경변수 사용) =====
broker_address = os.getenv("BROKER_ADDRESS")
broker_port = int(os.getenv("BROKER_PORT"))
mqtt_username = os.getenv("MQTT_USERNAME")
mqtt_password = os.getenv("MQTT_PASSWORD")
topic_audio = os.getenv("TOPIC_AUDIO")
topic_rms = os.getenv("TOPIC_RMS")
# MQTT 클라이언트 초기화
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_audio_pub")
client.username_pw_set(mqtt_username, mqtt_password)

def on_connect(client, userdata, flags, reason_code, properties):
    debug_print(f"MQTT 브로커 연결 완료, 상태 코드: {reason_code}")

client.on_connect = on_connect
client.connect(broker_address, broker_port)
client.loop_start()

# ===== MQTT용 PyAudio 설정 (음성 스트리밍용) =====
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 2048
DEVICE_INDEX = None

def stream_audio_mqtt():
    """
    마이크로부터 음성을 캡처하여 MQTT로 스트리밍하는 함수입니다.
    침묵이 3초간 감지되면 스트리밍을 종료합니다.
    오디오 데이터는 MQTT로 발행하고, 여기서 계산되는 RMS 값은
    침묵 감지용으로만 사용됩니다.
    """
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=DEVICE_INDEX
        )
    except Exception as e:
        debug_print("MQTT 스트리밍용 마이크 열기 실패: " + str(e))
        pa.terminate()
        return

    debug_print("MQTT 음성 스트리밍 시작")
    silence_threshold = 200       # RMS 임계치 (환경에 따라 조정)
    silence_duration_required = 3.0 # 3초 침묵 시 종료
    silence_start = None

    while True:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            debug_print("스트림 읽기 에러: " + str(e))
            continue

        # 오디오 데이터를 "devices/device1/audio" 토픽으로 전송 (바이너리 데이터)
        client.publish(topic_audio, audio_data)

        # int16 데이터를 float32로 변환하여 RMS 계산 (정확한 산술 연산을 위해)
        data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        talk_rms = np.sqrt(np.mean(np.square(data)))
        if np.isnan(talk_rms):
            debug_print("RMS 값이 NaN입니다. 이번 프레임 건너뜁니다.")
            continue

        print("RMS: " + str(talk_rms))
        if talk_rms < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration_required:
                debug_print("3초간 침묵 감지, MQTT 스트리밍 종료")
                break
        else:
            silence_start = None

    stream.stop_stream()
    stream.close()
    pa.terminate()
    debug_print("MQTT 음성 스트리밍 종료")

def main():
    # Hotword 감지 모델 및 detector 초기화
    base_model = Resnet50_Arc_loss()
    test_hw = HotwordDetector(
        hotword="test",  # 감지할 키워드
        model=base_model,
        reference_file="/home/ubuntu/S12P21S001/edge-device/keyword_detection/refer/hibixby_ref.json",
        threshold=0.7,
        relaxation_time=2
    )
    # hotword 감지를 위한 SimpleMicStream 생성
    mic_stream = SimpleMicStream(
        window_length_secs=1.5,
        sliding_window_secs=0.75,
    )
    mic_stream.start_stream()

    debug_print("‘하이 빅스비’라고 말하세요")
    while True:
        frame = mic_stream.getFrame()
        result = test_hw.scoreFrame(frame)
        if result is None:
            continue

        if result["match"] and "rms" in result:
            # numpy.float32를 Python float로 변환하여 JSON 직렬화 오류 방지
            keyword_rms = float(result["rms"])
            debug_print("Keyword RMS in result: " + str(keyword_rms))
            rms_payload = json.dumps({"keywork_rms": keyword_rms})
            client.publish(topic_rms, rms_payload)

            
        if result["match"]:
            debug_print("Hotword 감지됨! Confidence: " + str(result["confidence"]))
            # 매치 시 MQTT 음성 스트리밍 시작
            stream_audio_mqtt()
            debug_print("스트리밍 종료. 다시 ‘하이빅스비’라고 말하세요")
            result = None  # result 초기화하여 중복 처리 방지
        else:
            debug_print("Hotword 미감지. Confidence: " + str(result["confidence"]))

if __name__ == '__main__':
    main()
