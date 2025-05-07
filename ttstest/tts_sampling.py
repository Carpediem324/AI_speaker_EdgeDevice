import json
import requests
import base64
import os
from google.oauth2 import service_account
import google.auth.transport.requests

def get_access_token(service_account_file):
    # Cloud Text-to-Speech API 사용을 위한 스코프
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes)
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token

def synthesize_text(text, project_id, access_token, voice_name, output_file):
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
        "Authorization": "Bearer " + access_token
    }
    payload = {
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": "ko-KR",
            "name": voice_name
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        audio_content = result.get("audioContent")
        if audio_content:
            # audioContent는 base64 인코딩되어 있음
            audio_data = base64.b64decode(audio_content)
            with open(output_file, "wb") as out_file:
                out_file.write(audio_data)
            print(f"음성 파일이 '{output_file}'로 저장되었습니다. (Voice: {voice_name})")
        else:
            print("응답에 audioContent가 없습니다:", result)
    else:
        print("오류 발생:", response.status_code, response.text)

if __name__ == "__main__":
    # 합성할 텍스트 입력
    user_text = input("합성할 텍스트를 입력하세요: ")
    # 자신의 프로젝트 ID와 서비스 계정 키 파일 경로를 입력하세요.
    project_id = "webstt-b290c"  # 예: "my-gcp-project"
    service_account_file = "/home/ubuntu/ttstest/webstt-b290c-2503f9c8cfd5.json"
    
    token = get_access_token(service_account_file)
    
    # 사용할 음성 모델 목록 (총 8개)
    voices = [
        "ko-KR-Chirp3-HD-Aoede",
        "ko-KR-Chirp3-HD-Charon",
        "ko-KR-Chirp3-HD-Fenrir",
        "ko-KR-Chirp3-HD-Kore",
        "ko-KR-Chirp3-HD-Leda",
        "ko-KR-Chirp3-HD-Orus",
        "ko-KR-Chirp3-HD-Puck",
        "ko-KR-Chirp3-HD-Zephyr"
    ]
    
    # output 디렉토리가 없으면 생성
    output_dir = "/home/ubuntu/ttstest/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 각 음성 모델별로 파일 생성 (파일명은 output_<모델명>.wav)
    for voice in voices:
        output_filename = os.path.join(output_dir, f"output_{voice}.wav")
        synthesize_text(user_text, project_id, token, voice, output_filename)
