import pyaudio

audio = pyaudio.PyAudio()
print("오디오 장치 목록:")
for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print(f"{i}: {dev['name']} - Channels: {dev['maxInputChannels']}, Rate: {dev['defaultSampleRate']}")

audio.terminate()
