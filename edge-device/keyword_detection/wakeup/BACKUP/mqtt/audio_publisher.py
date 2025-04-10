import pyaudio
import paho.mqtt.client as mqtt

# MQTT 설정
broker_address = "192.168.100.34"
broker_port = 1884
topic = "audio/stream"

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK = 2048
DEVICE_INDEX = None
# 최신 Paho MQTT 클라이언트 초기화
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_audio_pub")
client.username_pw_set("myuser", "1")

# 최신 콜백 함수 정의 (Paho v2.0 기준)
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"MQTT 브로커 연결 완료, 상태 코드: {reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"메시지 전송 완료, mid: {mid}, 상태 코드: {reason_code}")

client.on_connect = on_connect
#client.on_publish = on_publish

client.connect(broker_address, broker_port)
client.loop_start()

# 오디오 스트림 초기화
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

print("음성 송신 시작 (Ctrl+C로 종료)")

try:
    while True:
        audio_data = stream.read(CHUNK)
        client.publish(topic, audio_data)

except KeyboardInterrupt:
    print("음성 송신 종료")

# 종료 처리
stream.stop_stream()
stream.close()
audio.terminate()
client.loop_stop()
client.disconnect()
