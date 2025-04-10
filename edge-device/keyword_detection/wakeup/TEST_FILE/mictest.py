# arecord -D hw:0,7 --dump-hw-params
# 이걸로 채널같은거 체크
import os
import subprocess

# 환경변수 ALSA_PCM_DEVICE를 사용할 경우 미리 설정 (이미 설정되어 있으면 건너뜁니다)
os.environ.setdefault("ALSA_PCM_DEVICE", "hw:1,0")
print(f"ALSA_PCM_DEVICE 환경변수: {os.environ['ALSA_PCM_DEVICE']}")

# ALSA에 등록된 기기 목록 출력 (arecord -l 명령어)
try:
    output = subprocess.check_output("arecord -l", shell=True, universal_newlines=True)
    print("ALSA 등록 가능한 기기 목록:")
    print(output)
except Exception as e:
    print("ALSA 장치 목록을 가져오는 중 오류 발생:", e)

# 녹음 시간 및 저장 파일 설정
record_time = 3  # 녹음 시간(초)
output_dir = "/home/ubuntu/wakeup/mictest"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "test.wav")

# arecord 명령어 옵션:
# -D: 사용할 ALSA 장치 (환경변수 ALSA_PCM_DEVICE에 설정된 값 사용)
# -f S16_LE: 16비트 little-endian PCM 형식
# -r 16000: 샘플레이트 16000Hz
# -c 2: 2채널 (장치가 지원하는 값)
# -d: 녹음 시간 (초)
command = [
    "arecord",
    "-D", os.environ.get("ALSA_PCM_DEVICE", "default"),
    "-f", "S16_LE",
    "-r", "16000",
    "-c", "2",
    "-d", str(record_time),
    output_file
]

print(f"\n{record_time}초 동안 녹음합니다...")
subprocess.run(command)

print(f"\n녹음된 오디오가 {output_file} 파일로 저장되었습니다.")
