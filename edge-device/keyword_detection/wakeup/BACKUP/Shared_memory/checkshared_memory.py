import time
from multiprocessing import shared_memory
import pyaudio

class SharedMemoryAudioController:
    def __init__(self, shm_name="signal_shm", shm_size=16, speaker_on=True,
                 input_device_index=None, output_device_index=6):
        """
        :param shm_name: 공유 메모리 이름 (핫워드 프로세스와 동일해야 함)
        :param shm_size: 공유 메모리 크기 (바이트)
        :param speaker_on: True이면 스피커로 재생, False이면 재생하지 않음
        :param input_device_index: 마이크 입력 장치 인덱스 (None이면 기본 장치 사용)
        :param output_device_index: 스피커 출력 장치 인덱스 (None이면 기본 장치 사용)
        """
        # 공유 메모리 연결 (이미 생성되어 있어야 함)
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.speaker_on = speaker_on

        self.p = pyaudio.PyAudio()
        # 기본 입력/출력 장치 정보 사용 (지원하는 샘플레이트 자동 결정)
        default_input_info = self.p.get_default_input_device_info()
        default_output_info = self.p.get_default_output_device_info()

        self.input_rate = int(default_input_info.get("defaultSampleRate", 44100))
        self.output_rate = int(default_output_info.get("defaultSampleRate", 44100))
        # 여기서는 입력과 출력 샘플레이트를 동일하게 사용합니다.
        self.rate = self.input_rate

        self.chunk = 1024  # 프레임 크기 (조정 가능)
        self.format = pyaudio.paInt16
        self.channels = 1  # 모노

        self.input_device_index = input_device_index if input_device_index is not None else default_input_info["index"]
        self.output_device_index = output_device_index if output_device_index is not None else default_output_info["index"]

    def read_signal(self):
        # 공유 메모리의 첫 16바이트를 읽어 신호 문자열로 복원
        signal_bytes = bytes(self.shm.buf[:16])
        signal = signal_bytes.split(b'\x00')[0].decode('utf-8')
        return signal

    def run(self):
        print("SharedMemoryAudioController 실행 중... (종료: Ctrl+C)")
        # 입력 스트림 열기
        input_stream = self.p.open(format=self.format,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chunk,
                                   input_device_index=self.input_device_index)
        # 스피커가 켜져 있다면 출력 스트림 열기
        output_stream = None
        if self.speaker_on:
            output_stream = self.p.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        output=True,
                                        frames_per_buffer=self.chunk,
                                        output_device_index=self.output_device_index)

        try:
            while True:
                current_signal = self.read_signal()
                if current_signal == "start":
                    print("신호가 'start'입니다. 오디오 캡처 및 재생 시작...")
                    # 공유 메모리 신호가 "start"인 동안 오디오를 캡처하여 출력합니다.
                    while self.read_signal() == "start":
                        try:
                            audio_data = input_stream.read(self.chunk, exception_on_overflow=False)
                        except Exception as e:
                            print("오디오 캡처 에러:", e)
                            continue
                        if output_stream:
                            output_stream.write(audio_data)
                        # 너무 빠른 루프로 인한 부하를 줄이기 위해 짧은 지연 추가
                        time.sleep(0.01)
                else:
                    print("신호가 'stop'입니다. 대기 중...")
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("SharedMemoryAudioController 종료...")
        finally:
            input_stream.stop_stream()
            input_stream.close()
            if output_stream:
                output_stream.stop_stream()
                output_stream.close()
            self.p.terminate()
            self.shm.close()

if __name__ == '__main__':
    controller = SharedMemoryAudioController(speaker_on=True)
    controller.run()
