import os
import glob
import wave
import numpy as np
import json
from scipy.signal import resample
from dotenv import load_dotenv
from eff_word_net.streams import CustomAudioStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss

# 환경 변수 로드
load_dotenv(dotenv_path='/home/ubuntu/S12P21S001/edge-device/.env')
HOT_WORD = os.getenv("HOT_WORD")
REFERENCE_FILE = os.getenv("REFERENCE_FILE")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))

TARGET_RATE = 16000

class SimpleWavStream(CustomAudioStream):
    def __init__(self, wav_path, window_length_secs=1.5, sliding_window_secs=0.75):
        # WAV 파일 로드
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)

            # 모노 처리
            if channels > 1:
                data = np.mean(data.reshape(-1, channels), axis=1).astype(np.int16)

            # 원본 샘플레이트에서 48000Hz로 변환
            if sample_rate != 48000:
                data = resample(data, int(len(data) * 48000 / sample_rate)).astype(np.int16)

            # 48000Hz에서 TARGET_RATE(16000Hz)로 다운샘플링
            data = resample(data, int(len(data) * TARGET_RATE / 48000)).astype(np.int16)

        self.audio_data = data
        self.current_idx = 0
        self.chunk_size = int(sliding_window_secs * TARGET_RATE)  # 정확히 12000 samples

        def get_next_frame():
            if self.current_idx >= len(self.audio_data):
                # 끝부분 도달 시 정확히 chunk_size만큼 0-padding
                return np.zeros(self.chunk_size, dtype=np.int16)
            chunk = self.audio_data[self.current_idx:self.current_idx+self.chunk_size]
            self.current_idx += self.chunk_size
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')
            return chunk

        super().__init__(
            open_stream=lambda: None,
            close_stream=lambda: None,
            get_next_frame=get_next_frame,
            window_length_secs=window_length_secs,
            sliding_window_secs=sliding_window_secs,
            sample_rate=TARGET_RATE
        )

def evaluate_wav_file(wav_path):
    print(f"Processing {wav_path}...")
    wav_stream = SimpleWavStream(wav_path)

    model = Resnet50_Arc_loss()
    detector = HotwordDetector(
        hotword=HOT_WORD,
        model=model,
        reference_file=REFERENCE_FILE,
        threshold=CONFIDENCE_THRESHOLD,
        relaxation_time=0
    )

    max_confidence = 0.0
    while True:
        frame = wav_stream.getFrame()
        if np.all(frame == 0):
            break
        result = detector.scoreFrame(frame)
        if result:
            confidence = float(result.get('confidence', 0))
            max_confidence = max(max_confidence, confidence)

    # 파일명 추출 후 접두어에 따라 type 지정
    base_name = os.path.basename(wav_path)
    if base_name.startswith("TA_"):
        file_type = "TA"
    elif base_name.startswith("FA_"):
        file_type = "FA"
    else:
        file_type = "Unknown"

    return {
        "wav_file": base_name,
        "max_confidence": max_confidence,
        "type": file_type
    }

def evaluate_wav_files(wav_dir, output_json):
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    results = [evaluate_wav_file(wav_file) for wav_file in wav_files]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    wav_directory = "/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/wav/"
    output_filename = "/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/results.json"
    evaluate_wav_files(wav_directory, output_filename)
