import numpy as np
import pandas as pd
import json
import unicodedata
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\ks.lee\Desktop\aisw\aisw\codyssey_aisw\questions\q4_sentiment\data"

# ── 데이터 로드 ──
df = pd.read_csv(f"{DATA_DIR}/reviews.csv")
with open(f"{DATA_DIR}/sentiment_dict.json", "r", encoding="utf-8") as f:
    sd = json.load(f)

# ── 데이터 전처리: NaN 제거 + NFC 정규화 ──
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].apply(lambda x: unicodedata.normalize("NFC", str(x)))
df = df.reset_index(drop=True)

print(f"총 리뷰: {len(df)}건, 긍정: {(df['label']==1).sum()}건, 부정: {(df['label']==0).sum()}건")

# ─────────────────────────────────────────────
# Part A: 규칙 기반 감성 분석
# ─────────────────────────────────────────────

def rule_sentiment(text):
    tokens = str(text).split()
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # 부정어: 다음 토큰 점수 * -1
        if t in sd["negation"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            ns = sd["positive"].get(nt, sd["negative"].get(nt, 0.0))
            score += ns * (-1.0)
            i += 2
            continue
        # 강조어: 다음 토큰 점수 * 배수
        if t in sd["intensifier"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            ns = sd["positive"].get(nt, sd["negative"].get(nt, 0.0))
            score += ns * sd["intensifier"][t]
            i += 2
            continue
        # 일반 토큰: positive 우선 조회
        ts = sd["positive"].get(t, sd["negative"].get(t, 0.0))
        score += ts
        i += 1
    return 1 if score > 0 else 0

# 전체 데이터에 규칙 기반 적용
all_rule_preds = df["text"].apply(rule_sentiment).values

# ─────────────────────────────────────────────
# Part B: ML 기반 감성 분석
# ─────────────────────────────────────────────

# 3. 데이터 분할 (70/30, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    df["text"], df["label"], test_size=0.30, random_state=42
)

# 4. 불균형 처리: np.random.choice로 소수 클래스(부정=0) 오버샘플링
tr_df = pd.DataFrame({"text": X_tr.values, "label": y_tr.values})
pos_df = tr_df[tr_df["label"] == 1]
neg_df = tr_df[tr_df["label"] == 0]

np.random.seed(42)
neg_over_idx = np.random.choice(np.arange(len(neg_df)), size=len(pos_df), replace=True)
neg_oversampled = neg_df.iloc[neg_over_idx]

bal = pd.concat([pos_df, neg_oversampled]).reset_index(drop=True)
# 셔플
np.random.seed(42)
bal = bal.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train 오버샘플 후: {len(bal)}건, 긍정: {(bal['label']==1).sum()}, 부정: {(bal['label']==0).sum()}")

# 5. TF-IDF (fit은 train에서만)
vec = TfidfVectorizer(sublinear_tf=False, smooth_idf=True)
X_tr_tf = vec.fit_transform(bal["text"])
X_te_tf = vec.transform(X_te)

# 6. LogisticRegression
mdl = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)
mdl.fit(X_tr_tf, bal["label"])
ml_preds = mdl.predict(X_te_tf)

# ─────────────────────────────────────────────
# Part C: 성능 비교
# ─────────────────────────────────────────────

def calc_metrics(yt, yp):
    return {
        "accuracy":      round(float(accuracy_score(yt, yp)), 4),
        "precision_pos": round(float(precision_score(yt, yp, pos_label=1, zero_division=0)), 4),
        "recall_pos":    round(float(recall_score(yt, yp, pos_label=1, zero_division=0)), 4),
        "precision_neg": round(float(precision_score(yt, yp, pos_label=0, zero_division=0)), 4),
        "recall_neg":    round(float(recall_score(yt, yp, pos_label=0, zero_division=0)), 4),
        "f1_macro":      round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
    }

# 규칙 기반: 테스트 인덱스에 해당하는 예측만 추출
rule_m = calc_metrics(y_te.values, all_rule_preds[X_te.index])
ml_m   = calc_metrics(y_te.values, ml_preds)

print("Rule-based:", rule_m)
print("ML-based:  ", ml_m)

# ─────────────────────────────────────────────
# SHAP 분석 (LinearExplainer)
# ─────────────────────────────────────────────

exp = shap.LinearExplainer(mdl, X_tr_tf)
sv  = exp.shap_values(X_te_tf)          # shape: (n_test, n_features)

fn = vec.get_feature_names_out()
ms = np.asarray(np.mean(sv, axis=0)).flatten()   # 평균 SHAP 값

# 긍정 기여 상위 5
top_pos_idx = np.argsort(ms)[::-1][:5]
shap_top5_positive = [
    {"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_pos_idx
]

# 부정 기여 상위 5 (가장 낮은 SHAP)
top_neg_idx = np.argsort(ms)[:5]
shap_top5_negative = [
    {"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_neg_idx
]

print("SHAP top5 positive:", shap_top5_positive)
print("SHAP top5 negative:", shap_top5_negative)

# ─────────────────────────────────────────────
# 비즈니스 요약
# ─────────────────────────────────────────────

neg_kw = shap_top5_negative[0]['word']
pos_kw = shap_top5_positive[0]['word']
business_summary = (
    f"머신러닝 모델은 규칙 기반 모델 대비 부정 리뷰 탐지율이 높아 고객 불만을 보다 정확히 식별합니다. "
    f"긍정 예측에는 '{pos_kw}' 등의 단어가, 부정 예측에는 '{neg_kw}' 등의 단어가 주요하게 작용합니다. "
    f"마케팅팀은 긍정 키워드를 홍보에 활용하고, 영업팀은 부정 키워드 중심으로 고객 불만 원인을 파악하여 서비스를 개선할 수 있습니다."
)

# ─────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────

result = {
    "rule_based":          rule_m,
    "ml_based":            ml_m,
    "shap_top5_positive":  shap_top5_positive,
    "shap_top5_negative":  shap_top5_negative,
    "business_summary":    business_summary,
}

out_path = f"{DATA_DIR}/result_q4.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: {out_path}")
print(json.dumps(result, ensure_ascii=False, indent=2))
