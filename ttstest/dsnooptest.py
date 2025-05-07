import pyaudio

# 설정 값
RATE = 48000                 # 샘플링 레이트 (Hz)
CHANNELS = 2                 # 채널 수 (스테레오)
FORMAT = pyaudio.paInt16     # 16비트 오디오
CHUNK = 1024                 # 한 번에 읽어올 프레임 수 (period_size)

def main():
    p = pyaudio.PyAudio()
    
    # 시스템에 연결된 모든 오디오 디바이스 목록 출력
    print("사용 가능한 오디오 장치 목록:")
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            print(f"  인덱스 {i}: {info['name']}")
        except Exception as e:
            print(f"  인덱스 {i}: 정보 조회 실패 ({e})")
    
    # 사용자에게 입력 장치 인덱스를 입력받음
    try:
        device_index = int(input("사용할 입력 디바이스 인덱스를 입력하세요: "))
    except ValueError:
        print("올바른 숫자를 입력하세요.")
        p.terminate()
        return

    # 입력 스트림 열기 (사용자가 선택한 장치 사용)
    try:
        input_stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=device_index,
                              frames_per_buffer=CHUNK)
    except Exception as e:
        print("입력 스트림을 열 수 없습니다:", e)
        p.terminate()
        return

    # 출력 스트림 열기 (기본 디바이스 사용)
    try:
        output_stream = p.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               output=True,
                               frames_per_buffer=CHUNK)
    except Exception as e:
        print("출력 스트림을 열 수 없습니다:", e)
        input_stream.close()
        p.terminate()
        return

    print("녹음 및 재생 중입니다... (중지하려면 Ctrl+C 입력)")
    try:
        while True:
            try:
                # exception_on_overflow=False로 설정하여 오버플로우 발생 시 예외를 내지 않음
                data = input_stream.read(CHUNK, exception_on_overflow=False)
                output_stream.write(data)
            except OSError as e:
                # 오버플로우 등의 오류 발생 시, 오류 메시지 출력 후 다음 반복으로 넘어감
                print("오류 발생:", e)
                continue
    except KeyboardInterrupt:
        print("\n중지합니다.")
    finally:
        if input_stream.is_active():
            input_stream.stop_stream()
        input_stream.close()
        if output_stream.is_active():
            output_stream.stop_stream()
        output_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
