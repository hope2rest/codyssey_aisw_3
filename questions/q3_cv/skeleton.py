"""Q3. 이미지 객체카운팅 규칙기반 한계분석 — 뼈대 코드"""


def conv2d(image, kernel):
    """2D 합성곱 연산 (NumPy only, 외부 라이브러리 사용 금지)"""
    # TODO: 구현
    pass


def count_boxes(image):
    """이미지에서 사각형 객체 수를 카운팅"""
    # TODO: 엣지 검출 (Sobel 커널 + conv2d)

    # TODO: 이진화

    # TODO: 연결 요소 분석 (라벨링)

    # TODO: 노이즈 필터링 (최소 면적)

    # TODO: 객체 수 반환
    pass


def main():
    image_dir = "data/images"
    label_path = "data/labels.json"

    # TODO: 라벨 로드

    # TODO: 이미지별 카운팅 수행

    # TODO: 카테고리별(easy/medium/hard) 메트릭 계산 (MAE, accuracy)

    # TODO: worst case 이미지 식별

    result = {
        "predictions": {},
        "metrics": {},
        "worst_case_image": None,
        "failure_reasons": [],
        "why_learning_based": "",
    }

    # TODO: result를 JSON 파일로 저장


if __name__ == "__main__":
    main()
