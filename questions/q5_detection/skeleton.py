"""Q5. 부품결함검출 딥러닝기초 전이학습 — 뼈대 코드"""


class DefectImageLoader:
    """부품 이미지 로더"""

    def __init__(self, image_dir):
        # TODO: 이미지 디렉토리 설정
        pass

    def load(self, part_ids):
        """part_id 목록에 해당하는 이미지를 로드하여 특징 배열 반환"""
        # TODO: 이미지 로드, 리사이즈, 정규화, 평탄화
        pass


class InspectionLogProcessor:
    """검수 기록 전처리기"""

    def __init__(self, log_path):
        # TODO: 검수 로그 로드
        pass

    def process(self):
        """유효 레이블만 필터링하여 정제된 데이터프레임 반환"""
        # TODO: 레이블 정규화, 유효 레이블 필터링
        pass


def conv2d(image, kernel):
    """2D 합성곱 연산 (NumPy only)"""
    # TODO: 구현
    pass


def relu(z):
    """ReLU 활성화 함수"""
    # TODO: 구현
    pass


def softmax(z):
    """Softmax 활성화 함수"""
    # TODO: 구현
    pass


def nn_forward(X, W1, b1, W2, b2, mean, std):
    """2층 신경망 순전파"""
    # TODO: 입력 정규화

    # TODO: 은닉층 (Linear + ReLU)

    # TODO: 출력층 (Linear + Softmax)
    pass


def main():
    image_dir = "data/part_images"
    log_path = "data/inspection_log.csv"
    weights_path = "data/pretrained_nn_weights.npz"
    features_path = "data/pretrained_features.npy"

    # === 데이터 준비 ===
    # TODO: 이미지 로드 (DefectImageLoader)

    # TODO: 검수 로그 전처리 (InspectionLogProcessor)

    # TODO: 데이터 요약 통계

    # TODO: train/test 분할 (stratified)

    # === 규칙 기반 ===
    # TODO: Sobel 엣지 기반 이진 분류 (양품 vs 불량)

    # === ML 기반 ===
    # TODO: PCA 차원축소 + TF-IDF(inspector_note) 결합

    # TODO: 분류 모델 학습 및 평가

    # === 신경망 순전파 ===
    # TODO: 사전학습 가중치 로드

    # TODO: nn_forward로 예측 및 평가

    # === 전이학습 비교 ===
    # TODO: scratch(PCA) vs pretrained 특징 비교

    # === 개선 실험 ===
    # TODO: 소수 클래스 F1 개선

    result = {
        "data_summary": {},
        "rule_based": {},
        "ml_based": {},
        "nn_forward": {},
        "pretrained": {},
        "transfer_gain": None,
        "improvement": {},
        "report": {},
    }

    # TODO: result를 JSON 파일로 저장


if __name__ == "__main__":
    main()
