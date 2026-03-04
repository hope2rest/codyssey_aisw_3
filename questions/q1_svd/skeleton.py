"""Q1. SVD 기반 차원축소 복원분석 — 뼈대 코드"""


def main():
    data_path = "data/sensor_data.csv"

    # TODO: 데이터 로드 (CSV, 헤더 없음)

    # TODO: 데이터 전처리 (표준화)

    # TODO: SVD 분해

    # TODO: 설명된 분산 비율 계산

    # TODO: 최적 k 탐색 (누적 분산 >= 95%)

    # TODO: 데이터 복원 (상위 k개 성분 사용)

    # TODO: 복원 MSE 계산

    result = {
        "optimal_k": None,
        "cumulative_variance_at_k": None,
        "reconstruction_mse": None,
        "top_5_singular_values": [],
        "explained_variance_ratio_top5": [],
    }

    # TODO: result를 JSON 파일로 저장


if __name__ == "__main__":
    main()
