"""
q3_solution.py
문제 3: 이미지 기반 객체 카운팅 (규칙 기반 구현 및 한계 분석)

학생 풀이:
 - Part A : NumPy 기반 conv2d 구현 + Sobel 엣지 검출
 - Part B : 이진화 → Connected Component → 최소 면적 필터
 - Part C : 카테고리별 MAE / Accuracy 계산 + 실패 원인 분석
"""

import numpy as np
import json
import os
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing

# -------------------------------------------------------
# 하이퍼파라미터 (명시적 변수로 정의)
# -------------------------------------------------------
THRESHOLD  = 30    # Sobel 엣지 이진화 임계값
MIN_AREA   = 100   # 최소 연결 컴포넌트 면적 (픽셀 수, 노이즈 제거)

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(DATA_DIR, "images")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "result_q3.json")


# -------------------------------------------------------
# Part A-1: NumPy만으로 2D 컨볼루션 구현 (valid 모드)
#           cv2.filter2D 등 외부 편의 함수 일절 사용 금지
# -------------------------------------------------------
def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D 컨볼루션 (valid 모드, NumPy 순수 구현).

    수학적 정의에 따라 커널을 180도 뒤집은 뒤
    슬라이딩 윈도우와의 element-wise 곱·합산으로 구현합니다.
    NumPy stride_tricks를 이용해 반복문 없이 효율적으로 처리합니다.

    Parameters
    ----------
    image  : 2D ndarray  (H x W)
    kernel : 2D ndarray  (kH x kW)

    Returns
    -------
    output : 2D ndarray  ((H-kH+1) x (W-kW+1))  — valid 모드
    """
    kH, kW = kernel.shape
    iH, iW = image.shape
    oH = iH - kH + 1
    oW = iW - kW + 1

    # 컨볼루션 정의에 따라 커널 180° 회전 (cross-correlation 아님)
    k_flip = kernel[::-1, ::-1]

    # NumPy stride_tricks: 슬라이딩 윈도우 뷰 생성 (메모리 복사 없음)
    shape   = (oH, oW, kH, kW)
    strides = (image.strides[0], image.strides[1],
               image.strides[0], image.strides[1])
    windows = np.lib.stride_tricks.as_strided(
        image, shape=shape, strides=strides
    )

    # 각 윈도우와 뒤집힌 커널의 element-wise 곱의 합
    output = np.einsum('ijkl,kl->ij', windows, k_flip)
    return output


# -------------------------------------------------------
# Part A-2: Sobel 3×3 커널 정의
# -------------------------------------------------------
SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

# 가우시안 블러 커널 (노이즈 제거용, conv2d로 직접 적용)
GAUSS3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0


# -------------------------------------------------------
# 유틸리티: 그레이스케일 변환 (문제 지시 공식)
# -------------------------------------------------------
def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """gray = 0.299·R + 0.587·G + 0.114·B"""
    return (0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2])


def pad_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """valid 모드로 줄어든 배열을 원본 크기로 복원 (edge 패딩)."""
    ph = target_h - arr.shape[0]
    pw = target_w - arr.shape[1]
    return np.pad(arr,
                  ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                  mode='edge')


def compute_edge_magnitude(gray: np.ndarray) -> np.ndarray:
    """
    가우시안 블러 → Sobel Gx/Gy → edge_magnitude = sqrt(Gx² + Gy²)
    """
    h, w = gray.shape

    # 가우시안 블러 (conv2d 직접 구현 사용)
    blurred = conv2d(gray, GAUSS3)
    blurred = pad_to(blurred, h, w)

    # Sobel 엣지 검출 (수평/수직)
    Gx = conv2d(blurred, SOBEL_X)
    Gy = conv2d(blurred, SOBEL_Y)

    # 엣지 크기
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = pad_to(magnitude, h, w)

    return magnitude


# -------------------------------------------------------
# Part B: 박스 카운팅 파이프라인
# -------------------------------------------------------
def count_boxes(image_path: str,
                threshold: float = THRESHOLD,
                min_area: int = MIN_AREA) -> int:
    """
    박스 카운팅 파이프라인:
    1. 이미지 로드 → RGB 변환 (PIL)
    2. 그레이스케일 변환 (직접 구현)
    3. conv2d(가우시안) + conv2d(Sobel) → 엣지 크기 (Part A)
    4. 이진화 (threshold)                    (Part B-3)
    5. Morphological closing (엣지 연결)
    6. Connected Component 분석 (scipy)      (Part B-4)
    7. 최소 면적 필터 (노이즈 제거)           (Part B-5)
    """
    # (1) 이미지 로드 (PIL 사용 허용, RGBA → RGB 자동 변환)
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)

    # (2) 그레이스케일 변환
    gray = to_grayscale(rgb)

    # (3) Part A: 엣지 크기 계산
    edge_mag = compute_edge_magnitude(gray)

    # (4) Part B-3: 이진화
    binary = (edge_mag > threshold).astype(np.uint8)

    # (5) Morphological closing — 끊어진 박스 외곽 엣지 연결
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct, iterations=3)

    # (6) Part B-4: Connected Component 분석
    labeled_array, num_features = scipy_label(closed)

    # (7) Part B-5: 최소 면적 필터
    valid_count = 0
    for comp_id in range(1, num_features + 1):
        area = int(np.sum(labeled_array == comp_id))
        if area >= min_area:
            valid_count += 1

    return valid_count


# -------------------------------------------------------
# Part C-6: 카테고리별 지표 계산
# -------------------------------------------------------
def compute_metrics(predictions: dict, labels: dict, category: str) -> dict:
    """MAE와 Accuracy 계산."""
    keys = sorted([
        k for k in labels
        if k.startswith(category + "_") and k in predictions
    ])
    if not keys:
        return {"mae": 0.0, "accuracy": 0.0}

    errors = [abs(predictions[k] - labels[k]) for k in keys]
    mae      = float(np.mean(errors))
    accuracy = float(sum(1 for e in errors if e == 0) / len(errors))
    return {"mae": round(mae, 4), "accuracy": round(accuracy, 4)}


def find_worst_case(predictions: dict, labels: dict, category: str) -> str:
    """카테고리에서 오차가 가장 큰 이미지 이름 반환."""
    keys = [
        k for k in labels
        if k.startswith(category + "_") and k in predictions
    ]
    if not keys:
        return ""
    return max(keys, key=lambda k: abs(predictions[k] - labels[k]))


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # test_ 접두사 키 제외
    image_names = sorted([k for k in labels if not k.startswith("test")])

    print("=" * 55)
    print("Part A + B: 박스 카운팅 파이프라인")
    print(f"  THRESHOLD={THRESHOLD}, MIN_AREA={MIN_AREA}")
    print("=" * 55)

    predictions = {}
    for name in image_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        if not os.path.exists(img_path):
            print(f"  [SKIP] {name} - 파일 없음")
            continue

        pred = count_boxes(img_path)
        predictions[name] = pred
        true_val = labels[name]
        mark = "O" if pred == true_val else "X"
        print(f"  {name}: 예측={pred:2d}, 정답={true_val:2d}  [{mark}]")

    print()
    print("=" * 55)
    print("Part C: 카테고리별 성능 지표")
    print("=" * 55)

    categories = ["easy", "medium", "hard"]
    metrics = {}
    for cat in categories:
        m = compute_metrics(predictions, labels, cat)
        metrics[cat] = m
        print(f"  {cat:6s}: MAE={m['mae']:.4f}, Accuracy={m['accuracy']:.4f}")

    # Part C-7: 최악 케이스 분석
    worst_case = find_worst_case(predictions, labels, "hard")
    if worst_case:
        wc_pred = predictions[worst_case]
        wc_true = labels[worst_case]
        print(f"\n  worst case (hard): {worst_case}")
        print(f"    예측={wc_pred}, 정답={wc_true}, 오차={abs(wc_pred - wc_true)}")

    # Part C-7: 실패 원인 3가지 이상
    failure_reasons = [
        "박스들이 밀집하거나 서로 겹쳐 있을 경우 Sobel 엣지가 연결되어 여러 박스가 하나의 연결 컴포넌트로 병합되므로, 규칙 기반 카운팅은 실제 개수를 심각하게 과소 추정한다.",
        "적재(Stacked) 형태나 불규칙한 다각형 형태에서는 단일 고정 임계값과 2D 엣지만으로 박스 경계를 올바르게 분리할 수 없으며, 깊이 정보 없이는 앞뒤 박스를 구분하기 불가능하다.",
        "크기 편차가 매우 큰 환경에서는 하나의 고정 min_area 값으로 소형 박스(노이즈와 유사)와 대형 박스를 동시에 처리할 수 없어 소형 박스가 노이즈로 오인되어 필터링된다.",
        "조명 불균일, 그림자, 박스 표면 질감에 의해 박스 내부에도 강한 엣지가 생성되어 단일 박스가 여러 컴포넌트로 분리되거나, 배경 텍스처가 박스로 오인식되는 위양성이 발생한다."
    ]

    # Part C-8: 학습 기반 접근법이 필요한 이유 (200자 이내)
    why_learning_based = (
        "규칙 기반 방법은 고정 임계값과 단순 형태 분석에 의존하므로 조명 변화, "
        "박스 겹침, 크기 편차, 적재 구조 등 복잡한 실세계 조건에 일반화할 수 없다. "
        "CNN 등 학습 기반 모델은 대규모 데이터로부터 특징을 자동 학습하여 "
        "다양한 환경에서도 강인한 객체 탐지가 가능하다."
    )

    assert len(why_learning_based) <= 200, f"200자 초과: {len(why_learning_based)}자"

    # 결과 JSON 저장
    result = {
        "predictions": predictions,
        "metrics": metrics,
        "worst_case_image": worst_case if worst_case else "",
        "failure_reasons": failure_reasons,
        "why_learning_based": why_learning_based
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
