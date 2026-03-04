"""Q4. 고객리뷰 감성분석 — 뼈대 코드"""


def rule_based_predict(text, sentiment_dict):
    """감성 사전 기반 규칙 예측: 긍정(1) / 부정(0)"""
    # TODO: 토큰화

    # TODO: 부정어/강조어 처리

    # TODO: 감성 점수 합산 → 예측 반환
    pass


def compute_metrics(y_true, y_pred):
    """정확도, 정밀도, 재현율, F1 계산"""
    # TODO: 구현
    pass


def main():
    data_dir = "data"

    # TODO: 리뷰 데이터 로드 (reviews.csv)

    # TODO: 감성 사전 로드 (sentiment_dict.json)

    # TODO: 결측치 처리

    # === 규칙 기반 ===
    # TODO: 각 리뷰에 rule_based_predict 적용

    # TODO: 규칙 기반 메트릭 계산

    # === ML 기반 ===
    # TODO: train/test 분할

    # TODO: 클래스 불균형 처리 (train에서만)

    # TODO: TF-IDF 벡터화 (train fit, test transform)

    # TODO: 모델 학습 및 예측

    # TODO: ML 메트릭 계산

    # === SHAP 해석 ===
    # TODO: SHAP 값 계산

    # TODO: 긍정/부정 상위 5개 단어 추출

    result = {
        "rule_based": {},
        "ml_based": {},
        "shap_top5_positive": [],
        "shap_top5_negative": [],
        "business_summary": "",
    }

    # TODO: result를 JSON 파일로 저장


if __name__ == "__main__":
    main()
