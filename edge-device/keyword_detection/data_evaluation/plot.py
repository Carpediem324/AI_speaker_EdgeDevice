import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    """
    JSON 데이터를 로드하여 FA/TA를 구분한 뒤,
    IQR 방식으로 이상치를 제거하고 K-Means(클러스터 2개)를 수행하여
    두 클러스터 중심값의 평균을 임계값으로 결정해 시각화하는 스크립트입니다.
    """
    
    # [1] 파일 경로 설정 (환경에 맞게 수정하세요)
    json_path = '/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/results.json'
    output_path = '/home/ubuntu/S12P21S001/edge-device/keyword_detection/data_evaluation/result_plot.png'
    
    # [2] JSON 데이터 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # [3] 모든 데이터에서 'max_confidence' 값 추출
    confidences = np.array([d['max_confidence'] for d in data]).reshape(-1, 1)
    
    # [4] IQR 방식 이상치 제거
    q1 = np.percentile(confidences, 25)  # 25% 지점
    q3 = np.percentile(confidences, 75)  # 75% 지점
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 조건에 맞는 값만 필터링
    filtered_confidences = confidences[
        (confidences >= lower_bound) & (confidences <= upper_bound)
    ].reshape(-1, 1)  # 2차원 형태로 재변환
    
    # [5] K-Means 클러스터링 (k=2, 이상치 제거된 데이터 사용)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(filtered_confidences)
    
    # 두 클러스터 중심값을 정렬 (낮은 값, 높은 값 순)
    cluster_centers = np.sort(kmeans.cluster_centers_.ravel())
    
    # 임계값(threshold)을 두 클러스터 중심의 평균으로 결정
    threshold = np.mean(cluster_centers)
    
    # [6] FA와 TA 그룹 분리 (원본 데이터 기준)
    # 정렬 기준(FA: 0, TA: 1)
    order = {'FA': 0, 'TA': 1}
    sorted_data = sorted(data, key=lambda x: order.get(x['type'], 2))
    
    # 실제 분류
    fa_data = [d for d in sorted_data if d['type'] == 'FA']
    ta_data = [d for d in sorted_data if d['type'] == 'TA']
    
    # X 좌표와 Y 좌표 분리
    fa_x = list(range(len(fa_data)))
    ta_x = list(range(len(fa_data), len(fa_data) + len(ta_data)))
    
    fa_y = [d['max_confidence'] for d in fa_data]
    ta_y = [d['max_confidence'] for d in ta_data]
    
    # [7] 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(fa_x, fa_y, color='red', label='FA', alpha=0.8)
    plt.scatter(ta_x, ta_y, color='blue', label='TA', alpha=0.8)
    
    # 임계값 표시 (녹색 점선)
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold (KMeans)')
    plt.text(
        0.5, threshold, 
        f"Threshold: {threshold:.2f}",
        color='green', fontsize=12,
        verticalalignment='bottom', 
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # 플롯 설정
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.title('FA/TA Confidence Scatter Plot with K-Means Threshold (Remove outlier)', fontsize=16)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    
    # [8] 결과 저장 및 출력
    plt.savefig(output_path)
    print(f"그래프가 '{output_path}' 파일로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()
