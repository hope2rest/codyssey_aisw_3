"""문제 3 정답"""
import numpy as np
from PIL import Image
from scipy.ndimage import label
import json, os

# ──────────────────────────────────────
# 1. conv2d 직접 구현 (stride=1, valid mode)
# ──────────────────────────────────────
def conv2d(image, kernel):
    """2D 합성곱 — NumPy 순수 구현 (딥러닝 프레임워크 미사용)"""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return out


# ──────────────────────────────────────
# 2. 유틸리티 및 커널 정의
# ──────────────────────────────────────
def to_gray(arr):
    """ITU-R 601 가중 그레이스케일 변환"""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

# 가우시안 블러 3×3 (노이즈 제거)
gauss3 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]], dtype=float) / 16.0

# Sobel 엣지 커널
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


# ──────────────────────────────────────
# 3. 하이퍼파라미터
# ──────────────────────────────────────
THRESHOLD = 50        # Sobel 엣지 이진화 임계값 (보조용)
MIN_AREA = 500        # 최소 연결 영역 크기 (노이즈 제거)
BG_DIST_THRESH = 25   # 배경과의 최소 색상 거리
BELT_DIST_THRESH = 35 # 벨트와의 최소 색상 거리


# ──────────────────────────────────────
# 4. 박스 검출 파이프라인
# ──────────────────────────────────────
def count_boxes(path):
    """
    파이프라인:
    (a) 이미지 로드 → RGBA 대응 → 그레이스케일
    (b) 배경/벨트 색상 자동 추정
    (c) 색상 기반 박스 영역 분할 (갈색 계열)
    (d) conv2d: 가우시안 블러 → Sobel 엣지 검출
    (e) 강한 엣지를 barrier로 사용하여 인접 박스 분리
    (f) Connected Component Labeling + 면적 필터링
    """
    # ── (a) 이미지 로드 ──
    img = Image.open(path).convert("RGB")  # RGBA→RGB 자동 변환
    arr = np.array(img, dtype=float)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = to_gray(arr)

    # ── (b) 배경/벨트 색상 자동 추정 ──
    # 배경: 상단 15% 영역의 중앙 색상
    bg_region = arr[:int(h * 0.15), int(w * 0.2):int(w * 0.8), :]
    bg_r = np.median(bg_region[:, :, 0])
    bg_g = np.median(bg_region[:, :, 1])
    bg_b = np.median(bg_region[:, :, 2])
    bg_dist = np.sqrt((r - bg_r)**2 + (g - bg_g)**2 + (b - bg_b)**2)

    # 벨트: 어두운 수평 밴드 (행 평균 밝기로 자동 탐지)
    row_mean = gray.mean(axis=1)
    belt_rows = np.where(row_mean < np.median(row_mean))[0]
    if len(belt_rows) > 0:
        belt_sample = arr[belt_rows[0]:belt_rows[-1], :10, :]
        belt_r = np.median(belt_sample[:, :, 0])
        belt_g = np.median(belt_sample[:, :, 1])
        belt_b = np.median(belt_sample[:, :, 2])
    else:
        belt_r, belt_g, belt_b = 20, 40, 65
    belt_dist = np.sqrt((r - belt_r)**2 + (g - belt_g)**2 + (b - belt_b)**2)

    # ── (c) 색상 기반 박스 마스크 ──
    # 박스 = 갈색 계열: R > G > B, R-B 차이 큼, 배경/벨트와 다름
    box_mask = (
        (r > 120)                        # 충분한 밝기
        & ((r - b) > 15)                 # 갈색 (Red-Blue 차이)
        & (g > 80)                       # 녹색 채널 적정
        & (g < r + 10)                   # R ≥ G (갈색)
        & (belt_dist > BELT_DIST_THRESH) # 벨트 아님
        & (bg_dist > BG_DIST_THRESH)     # 배경 아님
    )

    # ROI 제한: 벨트 영역 상하 부근만 분석
    if len(belt_rows) > 0:
        roi_top = max(0, belt_rows[0] - 70)
        roi_bot = min(h, belt_rows[-1] + 10)
        box_mask[:roi_top, :] = False
        box_mask[roi_bot:, :] = False

    # ── (d) conv2d 기반 엣지 검출 ──
    gray_smooth = conv2d(gray, gauss3)
    ph, pw = h - gray_smooth.shape[0], w - gray_smooth.shape[1]
    gray_smooth = np.pad(gray_smooth, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)))

    gx = conv2d(gray_smooth, sobel_x)
    gy = conv2d(gray_smooth, sobel_y)
    edge_magnitude = np.sqrt(gx**2 + gy**2)
    ph2, pw2 = h - edge_magnitude.shape[0], w - edge_magnitude.shape[1]
    edge_magnitude = np.pad(edge_magnitude, ((ph2//2, ph2-ph2//2), (pw2//2, pw2-pw2//2)))

    # ── (e) 강한 엣지를 barrier로 사용 ──
    # 상위 3% 엣지만 선택 (박스 외곽선, 벨트-배경 경계)
    # 골판지 텍스처 같은 약한 엣지는 무시
    strong_thresh = np.percentile(edge_magnitude, 97)
    strong_edge = edge_magnitude > strong_thresh

    # 색상 마스크에서 강한 엣지를 제거하여 인접 박스 분리
    box_interior = box_mask & (~strong_edge)

    # ── (f) Connected Component + 면적 필터 ──
    labeled, num_features = label(box_interior.astype(np.int32))

    count = 0
    for i in range(1, num_features + 1):
        area = int(np.sum(labeled == i))
        if area >= MIN_AREA:
            count += 1

    return count


# ──────────────────────────────────────
# 5. 전체 이미지 분석 및 결과 저장
# ──────────────────────────────────────
with open("labels.json") as f:
    labels = json.load(f)

preds = {}
cats = {"easy": [], "medium": [], "hard": []}

for name, true in sorted(labels.items()):
    path = os.path.join("images", f"{name}.png")
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: 이미지 없음")
        continue

    pred = count_boxes(path)
    preds[name] = pred
    cat = name.split("_")[0]
    if cat in cats:
        cats[cat].append((name, true, pred))
    mark = "✓" if pred == true else "✗"
    print(f"  {name}: 예측={pred}, 정답={true} {mark}")

# ── 카테고리별 메트릭 ──
metrics = {}
for cat, items in cats.items():
    if not items:
        continue
    errs = [abs(p - t) for _, t, p in items]
    metrics[cat] = {
        "mae": round(float(np.mean(errs)), 4),
        "accuracy": round(float(sum(1 for e in errs if e == 0) / len(items)), 4),
    }

# ── 최악 케이스 ──
hard_errs = [(n, abs(p - t)) for n, t, p in cats.get("hard", [])]
worst = max(hard_errs, key=lambda x: x[1])[0] if hard_errs else "unknown"

# ── 결과 JSON ──
result = {
    "predictions": preds,
    "metrics": metrics,
    "worst_case_image": worst,
    "failure_reasons": [
        "박스 간 겹침 및 적재로 인접 박스들이 하나의 연결 영역으로 병합되어 개수가 과소 추정됨",
        "고정된 Sobel 임계값과 최소 면적 기준이 크기 편차가 큰 박스에 동시 적응하지 못함",
        "그림자와 박스 테두리가 유사한 엣지 응답을 생성하여 거짓 양성과 거짓 음성이 혼재됨",
    ],
    "why_learning_based": (
        "고정된 색상 임계값과 Sobel 커널로는 조명 변화·겹침·적재·크기 편차 등에 적응할 수 없으며 "
        "CNN은 다양한 학습 데이터로부터 객체의 형태·질감·문맥 특징을 자동 학습하여 "
        "겹침과 크기 변형에 강건하게 일반화한다"
    ),
}

with open("result_q3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result_q3.json 저장 완료!")
