"""
q5_solution.py
자동차 부품 결함 검출 시스템 - AI/SW 심화 시험 문제 5
"""

import os
import sys
import json
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ─── 경로 설정 ───────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(DATA_DIR, "part_images")
CSV_PATH = os.path.join(DATA_DIR, "inspection_log.csv")
PRETRAINED_FEATURES_PATH = os.path.join(DATA_DIR, "pretrained_features.npy")
PRETRAINED_WEIGHTS_PATH = os.path.join(DATA_DIR, "pretrained_nn_weights.npz")
OUTPUT_PATH = os.path.join(DATA_DIR, "result_q5.json")

VALID_LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(VALID_LABELS)}


# ════════════════════════════════════════════════════════════
# Part A-1: DefectImageLoader
# ════════════════════════════════════════════════════════════
class DefectImageLoader:
    """이미지를 로드하고 전처리하는 클래스."""

    def __init__(self, image_dir, size=(64, 64)):
        self.image_dir = image_dir
        self.size = size

    def load(self, part_ids):
        """
        part_ids 목록에 해당하는 이미지를 로드한다.
        손상된 이미지는 건너뛰고, 유효한 이미지만 반환한다.

        Returns:
            valid_ids  : 유효한 part_id 목록 (str, zero-padded 4자리)
            images     : 각 이미지를 flatten한 numpy 배열 (N, 12288)
        """
        valid_ids = []
        images = []

        for pid in part_ids:
            filename = f"{int(pid):04d}.png"
            filepath = os.path.join(self.image_dir, filename)

            if not os.path.exists(filepath):
                continue

            try:
                img = Image.open(filepath)
                img.verify()          # 파일 무결성 검사
                img = Image.open(filepath)  # verify 후 다시 열어야 함
                img = img.convert("RGB")
                img = img.resize(self.size, Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0   # [0,1] 정규화
                flat = arr.flatten()   # 64*64*3 = 12288
                valid_ids.append(pid)
                images.append(flat)
            except (UnidentifiedImageError, OSError, Exception):
                # 손상된 파일 건너뛰기
                continue

        images = np.array(images, dtype=np.float32) if images else np.empty((0, 12288), dtype=np.float32)
        return valid_ids, images


# ════════════════════════════════════════════════════════════
# Part A-2: InspectionLogProcessor
# ════════════════════════════════════════════════════════════
class InspectionLogProcessor:
    """검사 기록 CSV를 정제하는 클래스."""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def process(self, valid_image_ids):
        """
        CSV를 로드하고 정제한다.

        Returns:
            df_clean : 정제된 DataFrame
        """
        df = pd.read_csv(self.csv_path)

        # 1) 중복 part_id 제거 (keep='first')
        df = df.drop_duplicates(subset="part_id", keep="first")

        # 2) defect_type 유니코드 정규화(NFC) + 공백 제거
        df["defect_type"] = df["defect_type"].apply(
            lambda x: unicodedata.normalize("NFC", str(x)).strip() if pd.notna(x) else x
        )

        # 3) 유효 레이블만 남기기
        df = df[df["defect_type"].isin(VALID_LABELS)].copy()

        # 4) inspector_note 결측값 → 빈 문자열
        df["inspector_note"] = df["inspector_note"].fillna("").astype(str)

        # 5) 이미지가 유효한 part_id만 남기기
        valid_set = set(valid_image_ids)
        df = df[df["part_id"].isin(valid_set)].copy()

        df = df.reset_index(drop=True)
        return df


# ════════════════════════════════════════════════════════════
# 규칙 기반 유틸: conv2d (NumPy only, valid mode)
# ════════════════════════════════════════════════════════════
def conv2d(image, kernel):
    """
    2D convolution (valid mode) - NumPy만 사용.
    image: (H, W), kernel: (kH, kW)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    output = np.zeros((out_H, out_W), dtype=np.float64)
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return output


def compute_edge_magnitude(flat_img):
    """
    flatten된 이미지(12288,)를 받아 Sobel 엣지 평균 강도를 반환한다.
    RGB → 그레이스케일(평균) → Sobel
    """
    img = flat_img.reshape(64, 64, 3)
    gray = img.mean(axis=2)   # (64, 64)

    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    Gx = conv2d(gray.astype(np.float64), sobel_x)
    Gy = conv2d(gray.astype(np.float64), sobel_y)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude.mean()


# ════════════════════════════════════════════════════════════
# NN 유틸: ReLU, Softmax
# ════════════════════════════════════════════════════════════
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    """수치 안정 Softmax (각 행에서 max를 빼고 exp 적용)."""
    z_stable = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def nn_forward(X, W1, b1, W2, b2, feature_mean, feature_std):
    """2층 NN forward pass."""
    # 정규화
    X_norm = (X - feature_mean) / (feature_std + 1e-8)
    z1 = X_norm @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Part A: 데이터 전처리")
    print("=" * 60)

    # ── A-1: 이미지 로드 ─────────────────────────────────────
    loader = DefectImageLoader(IMAGE_DIR)
    all_ids = list(range(500))   # 0000 ~ 0499
    valid_ids, images = loader.load(all_ids)
    print(f"유효 이미지 수: {len(valid_ids)}")

    # ── A-2: CSV 정제 ─────────────────────────────────────────
    processor = InspectionLogProcessor(CSV_PATH)
    df = processor.process(valid_ids)
    print(f"정제 후 유효 샘플 수: {len(df)}")

    # 이미지와 CSV를 part_id 기준으로 정렬/매칭
    valid_id_set = set(df["part_id"].tolist())

    # images를 df에 맞게 재구성 (part_id 순서 맞추기)
    id_to_img = {pid: img for pid, img in zip(valid_ids, images)}

    df = df[df["part_id"].isin(id_to_img.keys())].copy().reset_index(drop=True)
    X_images = np.array([id_to_img[pid] for pid in df["part_id"]])
    labels_str = df["defect_type"].tolist()
    labels = np.array([LABEL2IDX[l] for l in labels_str])
    notes = df["inspector_note"].tolist()

    # 레이블 분포
    from collections import Counter
    label_count = Counter(labels_str)
    label_dist = {lbl: label_count.get(lbl, 0) for lbl in VALID_LABELS}
    max_count = max(label_dist.values())
    min_count = min(v for v in label_dist.values() if v > 0)
    imbalance_ratio = round(max_count / min_count, 4)

    print(f"레이블 분포: {label_dist}")
    print(f"불균형 비율: {imbalance_ratio}")

    total_valid = len(df)

    # ── 데이터 분할 ───────────────────────────────────────────
    print("\n데이터 분할 (70/30, stratify, random_state=42)")
    idx_all = np.arange(total_valid)
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.3, random_state=42, stratify=labels
    )

    X_train_img = X_images[idx_train]
    X_test_img = X_images[idx_test]
    y_train = labels[idx_train]
    y_test = labels[idx_test]
    notes_train = [notes[i] for i in idx_train]
    notes_test = [notes[i] for i in idx_test]
    part_ids_train = [df["part_id"].iloc[i] for i in idx_train]
    part_ids_test = [df["part_id"].iloc[i] for i in idx_test]

    print(f"Train size: {len(idx_train)}, Test size: {len(idx_test)}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Part B-1: 규칙 기반 (엣지 강도 임계값)")
    print("=" * 60)

    # 엣지 강도 계산 (train 전체 + test 전체)
    print("  엣지 강도 계산 중 (train)...")
    edge_train = np.array([compute_edge_magnitude(img) for img in X_train_img])
    print("  엣지 강도 계산 중 (test)...")
    edge_test = np.array([compute_edge_magnitude(img) for img in X_test_img])

    # threshold = train 데이터에서 양품(label=0) 엣지 강도 중앙값
    good_mask = (y_train == LABEL2IDX["양품"])
    threshold = float(np.median(edge_train[good_mask]))
    print(f"  Threshold (양품 중앙값): {threshold:.4f}")

    # 이진 분류: 엣지 ≤ threshold → 양품(0), 그 외 → 불량(1)
    y_test_binary = (y_test != LABEL2IDX["양품"]).astype(int)
    y_pred_rule = (edge_test > threshold).astype(int)

    rule_acc = round(accuracy_score(y_test_binary, y_pred_rule), 4)
    print(f"  Test Accuracy (binary): {rule_acc}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Part B-2: ML 기반 (PCA + TF-IDF + LR)")
    print("=" * 60)

    # PCA: 95% 분산 설명
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_img)
    X_test_pca = pca.transform(X_test_img)
    pca_n = pca.n_components_
    print(f"  PCA n_components (95% 분산): {pca_n}")

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    X_train_tfidf = tfidf.fit_transform(notes_train).toarray()
    X_test_tfidf = tfidf.transform(notes_test).toarray()

    # 결합 (hstack)
    X_train_ml = np.hstack([X_train_pca, X_train_tfidf])
    X_test_ml = np.hstack([X_test_pca, X_test_tfidf])

    # LogisticRegression (5클래스)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_ml, y_train)
    y_pred_ml = lr.predict(X_test_ml)

    ml_acc = round(accuracy_score(y_test, y_pred_ml), 4)
    ml_f1 = round(f1_score(y_test, y_pred_ml, average="macro", zero_division=0), 4)
    print(f"  Test Accuracy: {ml_acc}")
    print(f"  Test Macro F1: {ml_f1}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Part B-3: 2층 NN Forward Pass (NumPy)")
    print("=" * 60)

    # 가중치 로드
    weights = np.load(PRETRAINED_WEIGHTS_PATH)
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    feature_mean = weights["feature_mean"]
    feature_std = weights["feature_std"]

    # pretrained_features.npy 로드 (500×128)
    # 인덱스는 원본 part_id 순서(0000~0499)에 대응
    pretrained_all = np.load(PRETRAINED_FEATURES_PATH)   # (500, 128)

    # test set에 해당하는 pretrained 특징 가져오기
    # part_id가 정수이므로 그대로 인덱스로 사용
    X_nn_test = pretrained_all[part_ids_test]

    # 전체 데이터에 대한 레이블 (원본 500개 중 유효한 것)
    # NN은 test set에서 평가
    nn_preds, nn_probs = nn_forward(X_nn_test, W1, b1, W2, b2, feature_mean, feature_std)

    nn_acc = round(accuracy_score(y_test, nn_preds), 4)
    nn_f1 = round(f1_score(y_test, nn_preds, average="macro", zero_division=0), 4)
    print(f"  Test Accuracy: {nn_acc}")
    print(f"  Test Macro F1: {nn_f1}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Part C: 전이학습 비교 (Scratch vs Pretrained)")
    print("=" * 60)

    # Scratch 모델: PCA 이미지 특징만 (텍스트 제외)
    lr_scratch = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_scratch.fit(X_train_pca, y_train)
    y_pred_scratch = lr_scratch.predict(X_test_pca)
    scratch_acc = round(accuracy_score(y_test, y_pred_scratch), 4)
    print(f"  Scratch Test Accuracy: {scratch_acc}")

    # Pretrained 모델: pretrained_features.npy 특징
    X_pretrained_train = pretrained_all[part_ids_train]
    X_pretrained_test = pretrained_all[part_ids_test]

    lr_pretrained = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_pretrained.fit(X_pretrained_train, y_train)
    y_pred_pretrained = lr_pretrained.predict(X_pretrained_test)

    pretrained_acc = round(accuracy_score(y_test, y_pred_pretrained), 4)
    pretrained_f1_macro = round(f1_score(y_test, y_pred_pretrained, average="macro", zero_division=0), 4)

    transfer_gain = round(pretrained_acc - scratch_acc, 4)
    print(f"  Pretrained Test Accuracy: {pretrained_acc}")
    print(f"  Transfer Gain: {transfer_gain}")

    # 클래스별 F1
    f1_per_class = f1_score(y_test, y_pred_pretrained, average=None, zero_division=0, labels=list(range(5)))
    class_f1 = {lbl: round(float(f1_per_class[i]), 4) for i, lbl in enumerate(VALID_LABELS)}
    print(f"  Class F1: {class_f1}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_pretrained, labels=list(range(5)))
    cm_list = cm.tolist()

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Part D: 개선 실험 (class_weight='balanced')")
    print("=" * 60)

    # before: pretrained 모델 macro F1
    before_f1 = pretrained_f1_macro

    # after: class_weight='balanced' 적용
    lr_balanced = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    lr_balanced.fit(X_pretrained_train, y_train)
    y_pred_balanced = lr_balanced.predict(X_pretrained_test)

    after_f1 = round(f1_score(y_test, y_pred_balanced, average="macro", zero_division=0), 4)

    # 클래스별 F1 개선량 확인
    f1_before_class = f1_score(y_test, y_pred_pretrained, average=None, zero_division=0, labels=list(range(5)))
    f1_after_class = f1_score(y_test, y_pred_balanced, average=None, zero_division=0, labels=list(range(5)))
    improvement_per_class = f1_after_class - f1_before_class
    most_improved_idx = int(np.argmax(improvement_per_class))
    most_improved_class = VALID_LABELS[most_improved_idx]

    print(f"  Before F1 (macro): {before_f1}")
    print(f"  After F1  (macro): {after_f1}")
    print(f"  Most Improved Class: {most_improved_class}")
    print(f"  Improvement per class: {dict(zip(VALID_LABELS, improvement_per_class.round(4)))}")

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("비즈니스 보고서 작성")
    print("=" * 60)

    report = {
        "purpose": (
            "본 시스템은 자동차 부품 생산 라인에서 결함을 자동으로 탐지하고 분류하기 위해 개발되었습니다. "
            "딥러닝 기반 전이학습을 활용하여 스크래치, 크랙, 변색, 이물질 등 5가지 유형을 구별함으로써 품질 관리 효율을 향상시킵니다. "
            "자동화된 결함 탐지로 인건비 절감과 검사 일관성 확보가 기대됩니다."
        ),
        "key_results": (
            f"규칙 기반 이진 분류 정확도는 {rule_acc:.4f}, ML 기반(PCA+TF-IDF+LR) 5클래스 정확도는 {ml_acc:.4f}(Macro F1={ml_f1:.4f})을 달성하였습니다. "
            f"사전학습 특징을 활용한 전이학습 모델은 {pretrained_acc:.4f}의 정확도와 {pretrained_f1_macro:.4f}의 Macro F1을 기록하였습니다. "
            f"class_weight='balanced' 적용 후 Macro F1이 {before_f1:.4f}에서 {after_f1:.4f}로 개선되었으며, 특히 {most_improved_class} 클래스 성능이 가장 크게 향상되었습니다."
        ),
        "transfer_learning_effect": (
            f"사전학습 특징을 사용한 모델은 스크래치 학습 모델 대비 정확도가 {transfer_gain:.4f} 향상되었습니다. "
            "이는 대규모 데이터로 사전학습된 특징 추출기가 소규모 공정 데이터에도 유효한 표현력을 제공함을 보여줍니다. "
            "전이학습은 데이터 수집 비용과 학습 시간을 동시에 절감하는 실용적인 접근법입니다."
        ),
        "improvement_suggestion": (
            "불균형 클래스 문제는 class_weight='balanced' 외에도 오버샘플링(SMOTE) 기법을 추가 적용하면 더 효과적입니다. "
            "이미지 데이터에 대해 Data Augmentation(회전, 반전, 밝기 조정)을 적용하면 모델 일반화 성능을 높일 수 있습니다. "
            "더 많은 레이블 데이터 확보와 딥러닝 기반 Fine-tuning을 통해 결함 탐지 정확도를 지속적으로 개선할 것을 권장합니다."
        ),
    }

    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("결과 JSON 저장")
    print("=" * 60)

    result = {
        "data_summary": {
            "total_valid_samples": total_valid,
            "label_distribution": label_dist,
            "imbalance_ratio": imbalance_ratio,
        },
        "rule_based": {
            "test_accuracy": rule_acc,
            "method": "edge_threshold_binary",
        },
        "ml_based": {
            "test_accuracy": ml_acc,
            "test_f1_macro": ml_f1,
            "pca_n_components": int(pca_n),
        },
        "nn_forward": {
            "test_accuracy": nn_acc,
            "test_f1_macro": nn_f1,
        },
        "pretrained": {
            "test_accuracy": pretrained_acc,
            "test_f1_macro": pretrained_f1_macro,
            "class_f1": class_f1,
            "confusion_matrix": cm_list,
        },
        "transfer_gain": transfer_gain,
        "improvement": {
            "before_f1": before_f1,
            "after_f1": after_f1,
            "most_improved_class": most_improved_class,
        },
        "report": report,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_PATH}")
    print("\n완료!")
    return result


if __name__ == "__main__":
    main()
