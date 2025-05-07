import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# JSON 파일 경로 설정 (환경에 맞게 수정하세요)
json_path = '/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/results.json'

# JSON 데이터 로드
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 모든 confidence 값 추출 (각 데이터는 dict 형태이며 'max_confidence' 키를 가짐)
confidences = np.array([d['max_confidence'] for d in data]).reshape(-1, 1)

# K-Means 클러스터링 (클러스터 개수 k=2)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(confidences)
cluster_centers = sorted(kmeans.cluster_centers_.flatten())  # 낮은값, 높은값 순으로 정렬

# 임계값은 두 중심의 평균값으로 결정
threshold = np.mean(cluster_centers)

# 기존 데이터 중 FA와 TA 그룹 분리 (원래 라벨을 그대로 사용)
order = {'FA': 0, 'TA': 1}
sorted_data = sorted(data, key=lambda x: order.get(x['type'], 2))
fa_data = [d for d in sorted_data if d['type'] == 'FA']
ta_data = [d for d in sorted_data if d['type'] == 'TA']

# X축: FA 데이터는 인덱스 0부터 시작, TA 데이터는 FA 그룹 다음 인덱스부터 시작
fa_x = list(range(len(fa_data)))
ta_x = list(range(len(fa_data), len(fa_data) + len(ta_data)))
fa_y = [d['max_confidence'] for d in fa_data]
ta_y = [d['max_confidence'] for d in ta_data]


plt.figure(figsize=(10, 6))
plt.scatter(fa_x, fa_y, color='red', label='FA')
plt.scatter(ta_x, ta_y, color='blue', label='TA')

# 기존 방식: TA 그룹 중 가장 낮은 confidence 값과 FA 그룹 중 가장 높은 confidence 값 표시
# if ta_y:
#     min_ta = min(ta_y)
#     plt.axhline(y=min_ta, color='blue', linestyle='--', label='Min TA')
#     plt.text(0.5, min_ta, f"Min TA: {min_ta:.2f}", color='blue', fontsize=12,
#              verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
# if fa_y:
#     max_fa = max(fa_y)
#     plt.axhline(y=max_fa, color='red', linestyle='--', label='Max FA')
#     plt.text(0.5, max_fa, f"Max FA: {max_fa:.2f}", color='red', fontsize=12,
#              verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

# 클러스터링을 통한 임계값 표시 (초록색 수평선)
plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold (KMeans)')
plt.text(0.5, threshold, f"Threshold: {threshold:.2f}", color='green', fontsize=12,
         verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Index', fontsize=14)
plt.ylabel('Confidence', fontsize=14)
plt.title('FA/TA Confidence Scatter Plot with K-Means Threshold', fontsize=16)
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()

# 파일로 저장
plt.savefig('/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/result_plot.png')
print("그래프가 'result_plot.png' 파일로 저장되었습니다.")

plt.show()
