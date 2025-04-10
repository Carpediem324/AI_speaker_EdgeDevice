import paho.mqtt.client as mqtt

# Callback API 버전 값을 첫 번째 인자로 전달 (대문자 VERSION2 사용)
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "python_pub")
# 인증 정보 설정 (예: 사용자 이름 "myuser", 비밀번호 "mypassword")
mqttc.username_pw_set("myuser", "1")
mqttc.connect("192.168.100.34", 1884)
mqttc.publish("test/docker", "fuckyou")
mqttc.disconnect()
