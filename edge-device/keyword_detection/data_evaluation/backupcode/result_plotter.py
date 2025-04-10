#신뢰도 값 추출:

# 모든 데이터에서 'max_confidence' 값만 뽑아서 배열로 만듭니다.

# 이상치 제거 (IQR 방식):

# 데이터의 25% 값(Q1)과 75% 값(Q3)을 구합니다.

# IQR = Q3 - Q1로 계산합니다.

# Q1 - 1.5 * IQR와 Q3 + 1.5 * IQR 범위 밖의 값들을 제거합니다.

# 클러스터링 (K-Means):

# 이상치가 제거된 신뢰도 값들을 2개의 그룹으로 나눕니다.

# 각 그룹의 중심값을 구합니다.

# 임계값 결정:

# 두 그룹 중심값의 평균을 임계값으로 사용합니다.

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

# IQR을 이용한 이상치 제거
q1 = np.percentile(confidences, 25)# 25% 지점
q3 = np.percentile(confidences, 75)# 75% 지점
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_confidences = confidences[(confidences >= lower_bound) & (confidences <= upper_bound)]
# 필터링된 결과가 1차원 배열이 되어버릴 수 있으므로, 2차원 배열로 변환합니다.
filtered_confidences = filtered_confidences.reshape(-1, 1)

# K-Means 클러스터링 (클러스터 개수 k=2, 이상치 제거된 데이터로 진행)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(filtered_confidences)
cluster_centers = sorted(kmeans.cluster_centers_.flatten())  # 낮은 값, 높은 값 순으로 정렬

# 임계값(threshold)은 두 클러스터 중심의 평균값으로 결정
threshold = np.mean(cluster_centers)

# 기존 데이터 중 FA와 TA 그룹 분리 (원래 라벨 그대로 사용)
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

# 클러스터링을 통한 임계값 표시 (초록색 수평선)
plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold (KMeans)')
plt.text(0.5, threshold, f"Threshold: {threshold:.2f}", color='green', fontsize=12,
         verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Index', fontsize=14)
plt.ylabel('Confidence', fontsize=14)
plt.title('FA/TA Confidence Scatter Plot with K-Means Threshold (이상치 제거 반영)', fontsize=16)
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()

# 파일로 저장
plt.savefig('/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/result_plot.png')
print("그래프가 'result_plot.png' 파일로 저장되었습니다.")

plt.show()
#신뢰도 값 추출:

# 모든 데이터에서 'max_confidence' 값만 뽑아서 배열로 만듭니다.

# 이상치 제거 (IQR 방식):

# 데이터의 25% 값(Q1)과 75% 값(Q3)을 구합니다.

# IQR = Q3 - Q1로 계산합니다.

# Q1 - 1.5 * IQR와 Q3 + 1.5 * IQR 범위 밖의 값들을 제거합니다.

# 클러스터링 (K-Means):

# 이상치가 제거된 신뢰도 값들을 2개의 그룹으로 나눕니다.

# 각 그룹의 중심값을 구합니다.

# 임계값 결정:

# 두 그룹 중심값의 평균을 임계값으로 사용합니다.

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

# IQR을 이용한 이상치 제거
q1 = np.percentile(confidences, 25)# 25% 지점
q3 = np.percentile(confidences, 75)# 75% 지점
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
filtered_confidences = confidences[(confidences >= lower_bound) & (confidences <= upper_bound)]
# 필터링된 결과가 1차원 배열이 되어버릴 수 있으므로, 2차원 배열로 변환합니다.
filtered_confidences = filtered_confidences.reshape(-1, 1)

# K-Means 클러스터링 (클러스터 개수 k=2, 이상치 제거된 데이터로 진행)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(filtered_confidences)
cluster_centers = sorted(kmeans.cluster_centers_.flatten())  # 낮은 값, 높은 값 순으로 정렬

# 임계값(threshold)은 두 클러스터 중심의 평균값으로 결정
threshold = np.mean(cluster_centers)

# 기존 데이터 중 FA와 TA 그룹 분리 (원래 라벨 그대로 사용)
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

# 클러스터링을 통한 임계값 표시 (초록색 수평선)
plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold (KMeans)')
plt.text(0.5, threshold, f"Threshold: {threshold:.2f}", color='green', fontsize=12,
         verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Index', fontsize=14)
plt.ylabel('Confidence', fontsize=14)
plt.title('FA/TA Confidence Scatter Plot with K-Means Threshold (이상치 제거 반영)', fontsize=16)
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()

# 파일로 저장
plt.savefig('/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/result_plot.png')
print("그래프가 'result_plot.png' 파일로 저장되었습니다.")

plt.show()
