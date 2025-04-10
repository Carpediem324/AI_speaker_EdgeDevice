import time
import threading
import gc
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt

from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# ===== MQTT 설정 =====
broker_address = "192.168.100.34"
broker_port = 1884
topic = "audio/stream"

# ===== MQTT용 PyAudio 설정 (음성 스트리밍용) =====
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 2048
DEVICE_INDEX = None

# MQTT 클라이언트 초기화
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_audio_pub")
client.username_pw_set("myuser", "1")
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT 브로커 연결 완료, 상태 코드: {reason_code}")
client.on_connect = on_connect
client.connect(broker_address, broker_port)
client.loop_start()

def stream_audio_mqtt():
    """
    마이크로부터 음성을 캡처하여 MQTT로 스트리밍하는 함수입니다.
    침묵이 3초간 감지되면 스트리밍을 종료합니다.
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
        print("MQTT 스트리밍용 마이크 열기 실패:", e)
        pa.terminate()
        return

    print("MQTT 음성 스트리밍 시작")
    silence_threshold = 500       # RMS 임계치 (환경에 따라 조정)
    silence_duration_required = 3.0 # 3초 침묵 시 종료
    silence_start = None

    while True:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print("스트림 읽기 에러:", e)
            continue

        client.publish(topic, audio_data)

        data = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(data**2))
        # print("RMS:", rms)  # 디버깅용

        if rms < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration_required:
                print("3초간 침묵 감지, MQTT 스트리밍 종료")
                break
        else:
            silence_start = None

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("MQTT 음성 스트리밍 종료")

def main():
    # Hotword 감지 모델 및 detector 초기화
    base_model = Resnet50_Arc_loss()
    test_hw = HotwordDetector(
        hotword="test",  # 감지할 키워드
        model=base_model,
        reference_file="/home/ubuntu/wakeup/test/test_ref.json",
        threshold=0.7,
        relaxation_time=2
    )
    # hotword 감지를 위한 SimpleMicStream 생성
    mic_stream = SimpleMicStream(
        window_length_secs=1.5,
        sliding_window_secs=0.75,
    )
    mic_stream.start_stream()

    print("‘test’라고 말하세요")
    while True:
        frame = mic_stream.getFrame()
        result = test_hw.scoreFrame(frame)
        if result is None:
            continue
        if result["match"]:
            print("Hotword 감지됨! Confidence:", result["confidence"])
            # hotword 감지 시 기존 스트림을 종료(객체 삭제)하고 가비지 컬렉션 실행
            try:
                del mic_stream
                gc.collect()
                # 장치 해제를 위해 충분한 시간(예: 5초) 대기
                time.sleep(5)
                print("Hotword 스트림 종료 완료")
            except Exception as e:
                print("mic_stream 종료 실패:", e)
            # MQTT 음성 스트리밍 시작
            stream_audio_mqtt()
            # MQTT 스트리밍 종료 후, 다시 hotword 감지를 위해 새 SimpleMicStream 생성 전에 대기
            time.sleep(5)
            mic_stream = SimpleMicStream(
                window_length_secs=1.5,
                sliding_window_secs=0.75,
            )
            mic_stream.start_stream()
            print("스트리밍 종료. 다시 ‘test’라고 말하세요")
        else:
            print("Hotword 미감지. Confidence:", result["confidence"])

if __name__ == '__main__':
    main()
