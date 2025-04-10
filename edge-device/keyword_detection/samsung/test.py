import pyaudio

def check_microphone():
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()
    microphone_found = False

    print("사용 가능한 오디오 장치 목록:")
    for i in range(device_count):
        device_info = pa.get_device_info_by_index(i)
        # 입력 채널(maxInputChannels)이 0보다 크면 마이크 등 입력 장치로 인식
        if device_info.get('maxInputChannels', 0) > 0:
            print(f"  - {device_info.get('name')} (인덱스: {i})")
            microphone_found = True

    if microphone_found:
        print("마이크(입력 장치)가 감지되었습니다.")
    else:
        print("연결된 마이크(입력 장치)가 없습니다.")

    pa.terminate()

if __name__ == "__main__":
    check_microphone()
