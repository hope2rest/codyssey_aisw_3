"""문제 5.1 정답(자동차 부품 결함 검출)"""
import numpy as np, pandas as pd, json, os, re, warnings, unicodedata
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")

LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]
LABEL2IDX = {l:i for i,l in enumerate(LABELS)}

# ═══════════════════════════════════════
# Part A: 데이터 전처리 (클래스 기반)
# ═══════════════════════════════════════
class DefectImageLoader:
    """부품 이미지 로드 + 전처리"""
    def __init__(self, img_dir="part_images", target_size=(64,64)):
        self.img_dir = img_dir
        self.target_size = target_size
    
    def load_single(self, part_id):
        """단일 이미지 로드 → RGB → resize → [0,1] → flatten"""
        fpath = os.path.join(self.img_dir, f"{part_id}.png")
        if not os.path.exists(fpath):
            return None
        if os.path.getsize(fpath) == 0:
            return None
        try:
            img = Image.open(fpath)
            img.verify()
            # verify 후 다시 열어야 함
            img = Image.open(fpath)
            # [Trap 1,2] Grayscale/RGBA → RGB 변환
            img = img.convert("RGB")
            img = img.resize(self.target_size)
            arr = np.array(img, dtype=np.float64) / 255.0
            return arr.flatten()  # 12288차원
        except:
            return None
    
    def load_batch(self, part_ids):
        """배치 로드, 유효 인덱스 반환"""
        features = []
        valid_ids = []
        for pid in part_ids:
            feat = self.load_single(pid)
            if feat is not None:
                features.append(feat)
                valid_ids.append(pid)
        return np.array(features), valid_ids


class InspectionLogProcessor:
    """검사 기록 CSV 전처리"""
    def __init__(self, csv_path="inspection_log.csv"):
        self.csv_path = csv_path
        self.valid_labels = LABELS
    
    def process(self, valid_image_ids=None):
        """CSV 정제 → DataFrame 반환"""
        df = pd.read_csv(self.csv_path, dtype={"part_id": str})
        print(f"원본: {len(df)}행")
        
        # [Trap 8] 중복 part_id 제거
        df = df.drop_duplicates(subset="part_id", keep="first")
        print(f"중복 제거: {len(df)}행")
        
        # part_id 0-패딩 보정
        df["part_id"] = df["part_id"].apply(lambda x: str(x).zfill(4))
        
        # [Trap 6+9] defect_type: NFC 정규화 + 공백 제거
        df["defect_type"] = df["defect_type"].apply(
            lambda x: unicodedata.normalize("NFC", str(x).strip()))
        
        # 유효 레이블만 유지
        df = df[df["defect_type"].isin(self.valid_labels)].copy()
        print(f"유효 레이블: {len(df)}행")
        
        # [Trap 7] NaN inspector_note → 빈 문자열
        nan_count = df["inspector_note"].isna().sum()
        if nan_count > 0:
            print(f"⚠ NaN 노트 {nan_count}건 → 빈 문자열로 대체")
            df["inspector_note"] = df["inspector_note"].fillna("")
        
        # 이미지 유효성 필터
        if valid_image_ids is not None:
            df = df[df["part_id"].isin(valid_image_ids)].copy()
            print(f"유효 이미지 매칭: {len(df)}행")
        
        df = df.reset_index(drop=True)
        return df


# ═══════════════════════════════════════
# 0. 데이터 로드 및 정제
# ═══════════════════════════════════════
loader = DefectImageLoader()
processor = InspectionLogProcessor()

# 먼저 CSV 정제 (이미지 필터 전)
df_raw = processor.process()

# 이미지 로드
all_pids = df_raw["part_id"].tolist()
X_img, valid_pids = loader.load_batch(all_pids)
print(f"유효 이미지: {len(valid_pids)}개")

# 이미지 유효 ID로 CSV 재필터
df = df_raw[df_raw["part_id"].isin(valid_pids)].copy().reset_index(drop=True)
print(f"최종 유효 샘플: {len(df)}행")

# 이미지 행렬을 df 순서에 맞춤
pid_to_img = dict(zip(valid_pids, X_img))
X_images = np.array([pid_to_img[pid] for pid in df["part_id"]])

# 레이블
y = np.array([LABEL2IDX[l] for l in df["defect_type"]])
label_dist = {l: int((y == i).sum()) for i, l in enumerate(LABELS)}
imbalance_ratio = max(label_dist.values()) / max(min(label_dist.values()), 1)

print(f"\n레이블 분포: {label_dist}")
print(f"불균형 비율: {imbalance_ratio:.2f}")

data_summary = {
    "total_valid_samples": len(df),
    "label_distribution": label_dist,
    "imbalance_ratio": round(imbalance_ratio, 4)
}

# ═══════════════════════════════════════
# Part B: 3단계 모델 비교
# ═══════════════════════════════════════

# 데이터 분할
X_train_idx, X_test_idx = train_test_split(
    np.arange(len(df)), test_size=0.3, random_state=42, stratify=y)
y_train, y_test = y[X_train_idx], y[X_test_idx]

# ─── B-1. 규칙 기반 (엣지 강도) ───
def conv2d(image, kernel):
    """NumPy 직접 구현: valid 모드 2D 컨볼루션"""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# Sobel 커널
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

def compute_edge_intensity(flat_img):
    """flatten 이미지 → 그레이스케일 → 엣지 강도 평균"""
    img = flat_img.reshape(64, 64, 3)
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    gx = conv2d(gray, sobel_x)
    gy = conv2d(gray, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude.mean()

print("\n규칙 기반: 엣지 강도 계산 중...")
edge_intensities = np.array([compute_edge_intensity(X_images[i]) for i in range(len(X_images))])

# 임계값: train 양품의 엣지 강도 중앙값
train_good_edges = edge_intensities[X_train_idx][y_train == 0]
threshold = np.median(train_good_edges)

# 이진 분류: 양품(0) vs 불량(1)
y_test_binary = (y_test > 0).astype(int)
pred_rule = (edge_intensities[X_test_idx] > threshold).astype(int)
rule_acc = accuracy_score(y_test_binary, pred_rule)
print(f"규칙 기반 (이진): accuracy={rule_acc:.4f}")

rule_result = {
    "test_accuracy": round(rule_acc, 4),
    "method": "edge_threshold_binary"
}

# ─── B-2. ML 기반 (PCA + TF-IDF + LR) ───
print("\nML 기반: PCA + TF-IDF 추출 중...")

# 이미지 PCA
pca = PCA(n_components=0.95, random_state=42)
X_img_pca = pca.fit_transform(X_images[X_train_idx])
X_img_pca_test = pca.transform(X_images[X_test_idx])
n_pca = pca.n_components_
print(f"PCA: {n_pca} components (95% variance)")

# 텍스트 TF-IDF
texts = df["inspector_note"].values
tfidf = TfidfVectorizer(max_features=100)
X_text_train = tfidf.fit_transform(texts[X_train_idx]).toarray()
X_text_test = tfidf.transform(texts[X_test_idx]).toarray()

# 결합
X_ml_train = np.hstack([X_img_pca, X_text_train])
X_ml_test = np.hstack([X_img_pca_test, X_text_test])

# LR 학습
lr_ml = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_ml.fit(X_ml_train, y_train)
pred_ml = lr_ml.predict(X_ml_test)
ml_acc = accuracy_score(y_test, pred_ml)
ml_f1 = f1_score(y_test, pred_ml, average="macro")
print(f"ML 기반 (5클래스): accuracy={ml_acc:.4f}, f1_macro={ml_f1:.4f}")

ml_result = {
    "test_accuracy": round(ml_acc, 4),
    "test_f1_macro": round(ml_f1, 4),
    "pca_n_components": int(n_pca)
}

# ─── B-3. 2층 신경망 Forward Pass ───
print("\nNN Forward: 가중치 로드 및 추론 중...")

# 가중치 로드
weights = np.load("pretrained_nn_weights.npz")
W1, b1 = weights["W1"], weights["b1"]
W2, b2 = weights["W2"], weights["b2"]
feat_mean, feat_std = weights["feature_mean"], weights["feature_std"]

# 사전학습 특징 로드
pretrained_feats = np.load("pretrained_features.npy")

# 유효 part_id에 대응하는 특징만 추출
pid_to_feat = {}
for i in range(500):
    pid = f"{i:04d}"
    if pid in set(df["part_id"]):
        pid_to_feat[pid] = pretrained_feats[i]

X_pretrained = np.array([pid_to_feat[pid] for pid in df["part_id"]])

# 정규화
X_norm = (X_pretrained - feat_mean) / feat_std

# Forward Pass (NumPy)
def relu(x):
    return np.maximum(0, x)

def softmax(z):
    """수치 안정 Softmax"""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

z1 = X_norm[X_test_idx] @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
probs = softmax(z2)
pred_nn = probs.argmax(axis=1)

nn_acc = accuracy_score(y_test, pred_nn)
nn_f1 = f1_score(y_test, pred_nn, average="macro")
print(f"NN Forward (5클래스): accuracy={nn_acc:.4f}, f1_macro={nn_f1:.4f}")

nn_result = {
    "test_accuracy": round(nn_acc, 4),
    "test_f1_macro": round(nn_f1, 4)
}

# ═══════════════════════════════════════
# Part C: 전이학습 비교
# ═══════════════════════════════════════
print("\n전이학습 비교...")

# Scratch: 이미지 PCA 특징만 → LR (텍스트 제외, 공정 비교)
lr_scratch = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_scratch.fit(X_img_pca, y_train)
pred_scratch = lr_scratch.predict(X_img_pca_test)
scratch_acc = accuracy_score(y_test, pred_scratch)
scratch_f1 = f1_score(y_test, pred_scratch, average="macro")
print(f"Scratch (PCA image only): accuracy={scratch_acc:.4f}, f1={scratch_f1:.4f}")

# Pretrained: pretrained_features → LR
lr_pre = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_pre.fit(X_pretrained[X_train_idx], y_train)
pred_pre = lr_pre.predict(X_pretrained[X_test_idx])
pre_acc = accuracy_score(y_test, pred_pre)
pre_f1 = f1_score(y_test, pred_pre, average="macro")
print(f"Pretrained LR: accuracy={pre_acc:.4f}, f1_macro={pre_f1:.4f}")

transfer_gain = pre_acc - scratch_acc
print(f"Transfer Gain: {transfer_gain:+.4f}")

# 클래스별 F1
class_f1 = {}
per_class_f1 = f1_score(y_test, pred_pre, average=None)
for i, lbl in enumerate(LABELS):
    class_f1[lbl] = round(float(per_class_f1[i]), 4)

# Confusion Matrix
cm = confusion_matrix(y_test, pred_pre).tolist()

pretrained_result = {
    "test_accuracy": round(pre_acc, 4),
    "test_f1_macro": round(pre_f1, 4),
    "class_f1": class_f1,
    "confusion_matrix": cm
}

# ═══════════════════════════════════════
# Part D: 성능 평가 + 개선 + 보고서
# ═══════════════════════════════════════
print("\n개선 실험: class_weight='balanced'...")

# Before (이미 계산됨)
before_f1 = pre_f1
before_class_f1 = per_class_f1.copy()

# After
lr_balanced = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                  class_weight="balanced")
lr_balanced.fit(X_pretrained[X_train_idx], y_train)
pred_balanced = lr_balanced.predict(X_pretrained[X_test_idx])
after_f1 = f1_score(y_test, pred_balanced, average="macro")
after_class_f1 = f1_score(y_test, pred_balanced, average=None)

# 가장 개선된 클래스
f1_improvements = after_class_f1 - before_class_f1
most_improved_idx = np.argmax(f1_improvements)
most_improved_class = LABELS[most_improved_idx]

print(f"Before F1: {before_f1:.4f} → After F1: {after_f1:.4f}")
print(f"가장 개선된 클래스: {most_improved_class} ({f1_improvements[most_improved_idx]:+.4f})")

improvement_result = {
    "before_f1": round(before_f1, 4),
    "after_f1": round(after_f1, 4),
    "most_improved_class": most_improved_class
}

# 비즈니스 보고서
report = {
    "purpose": f"자동차 부품 생산 라인에서 결함을 자동 검출하여 품질 관리 효율을 높이고 불량 유출을 방지합니다. 5종 결함(양품/스크래치/크랙/변색/이물질)을 이미지와 검사 기록으로 분류합니다. 연간 수작업 검사 비용을 절감하고 검출 일관성을 확보할 수 있습니다.",
    "key_results": f"사전학습 특징 기반 모델이 {pre_acc:.1%} 정확도, F1 {pre_f1:.4f}로 가장 우수했습니다. 규칙 기반(엣지 임계값)은 {rule_acc:.1%}로 단순 이진 분류에만 적용 가능했고, ML 기반은 {ml_acc:.1%}였습니다. 소수 클래스(이물질)의 F1은 {class_f1['이물질']}로 개선 여지가 있습니다.",
    "transfer_learning_effect": f"사전학습 특징을 활용한 모델은 직접 추출 대비 정확도가 {transfer_gain:+.1%}p 향상되었습니다. 특히 소수 클래스 인식에서 큰 차이를 보이며, 적은 데이터로도 높은 성능을 달성할 수 있음을 확인했습니다. 산업 현장에서 레이블링 비용을 줄이는 핵심 전략입니다.",
    "improvement_suggestion": f"class_weight='balanced' 적용으로 Macro F1이 {before_f1:.4f}→{after_f1:.4f}로 변화했으며, {most_improved_class} 클래스가 가장 큰 개선을 보였습니다. 향후 CNN 기반 특징 추출과 데이터 증강으로 소수 클래스 성능을 추가 개선할 수 있습니다. 실시간 라인 적용을 위해 경량 모델과 추론 속도 최적화도 필요합니다."
}

# ═══════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════
result = {
    "data_summary": data_summary,
    "rule_based": rule_result,
    "ml_based": ml_result,
    "nn_forward": nn_result,
    "pretrained": pretrained_result,
    "transfer_gain": round(transfer_gain, 4),
    "improvement": improvement_result,
    "report": report
}

with open("result_q6.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print("result_q6.json 저장 완료")
print(json.dumps(result, ensure_ascii=False, indent=2))
