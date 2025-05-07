# 음성녹화자동화
import subprocess
import os

def record_audio(file_path):
    """
    arecord 명령어를 통해 음성 녹음을 시작하고,
    엔터 입력 시 녹음을 종료하여 file_path에 저장하는 함수
    """
    print("녹음 시작... (종료하려면 엔터를 입력하세요)")
    # arecord 명령어: -D default: 기본 장치, -f S16_LE: 포맷, -r 48000: 샘플레이트, -c 2: 채널 수
    # #######################채널수체크해야함################
    process = subprocess.Popen(["arecord", "-D", "default", "-f", "S16_LE", "-r", "48000", "-c", "1", file_path])
    # 녹음 종료를 위해 사용자 엔터 입력 대기
    input()  
    # 녹음 종료: 프로세스 종료
    process.terminate()
    process.wait()
    print(f"녹음 종료: 파일 저장됨 -> {file_path}")

def main():
    # 파일이 저장될 경로 설정
    dataset_dir = "/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 사용자 이름 입력
    name = input("이름을 입력하세요 (예: shh, honggildong): ").strip()
    
    # FA 또는 TA 선택 (1: FA, 2: TA)
    selection = input("번호를 선택하세요 (1: FA, 2: TA): ").strip()
    if selection == "1":
        prefix = "FA_"
    elif selection == "2":
        prefix = "TA_"
    else:
        print("잘못된 선택입니다. 기본값 TA로 설정합니다.")
        prefix = "TA_"
    
    index = 1
    print("녹음 시작하려면 엔터, 프로그램 종료는 'q' 입력 후 엔터를 누르세요.")
    while True:
        user_input = input("명령 입력 (엔터: 녹음, q: 종료): ").strip()
        if user_input.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        else:
            # 파일명 생성: 예) TA_shh1.wav, FA_honggildong2.wav
            file_name = f"{prefix}{name}{index}.wav"
            file_path = os.path.join(dataset_dir, file_name)
            print("녹음을 시작합니다. 녹음을 종료하려면 엔터를 누르세요.")
            record_audio(file_path)
            index += 1

if __name__ == "__main__":
    main()
