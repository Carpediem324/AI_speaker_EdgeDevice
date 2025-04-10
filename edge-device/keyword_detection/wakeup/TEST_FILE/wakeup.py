import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc

def main():
    # base_model: Resnet50 기반 모델로 Arc Loss를 적용해 임베딩을 추출합니다.
    base_model = Resnet50_Arc_loss()

    # HotwordDetector 설정:
    test_hw = HotwordDetector(
        hotword="test",  # 감지할 wake word (여기서는 "test")
        model=base_model,  # 사용될 음성 임베딩 모델
        reference_file="/home/ubuntu/wakeup/test/test_ref.json",  # 기준 임베딩 파일 경로
        threshold=0.7,  # 임계치 (이 값 이상이면 match가 True)
        relaxation_time=2  # 감지 후 재감지를 방지하기 위한 대기 시간(초)
    )

    # SimpleMicStream 설정:
    mic_stream = SimpleMicStream(
        window_length_secs=1.5,  # 한 번에 처리할 오디오 프레임의 길이(초)
        sliding_window_secs=0.75,  # 연속 프레임 간 중첩 시간(초)
        custom_channels=2,
        custom_rate=48000,
        custom_device_index=None
    )

    mic_stream.start_stream()

    print("Say test")
    while True:
        frame = mic_stream.getFrame()
        result = test_hw.scoreFrame(frame)
        if result is None:
            # 음성 활동이 없는 경우
            continue
        if result["match"]:
            # 임계치를 초과한 경우
            print("test : Wakeword uttered", result["confidence"])
        else:
            # 임계치보다 낮은 경우
            print("Wakeword not triggered. Confidence:", result["confidence"])

if __name__ == '__main__':
    main()
