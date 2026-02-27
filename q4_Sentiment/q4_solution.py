"""문항 4 모범 정답 — 원본 JSON 키"""
import numpy as np, pandas as pd, json, warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
warnings.filterwarnings("ignore")

# CSV 로드 (쉼표 포함 텍스트도 정상 처리)
df = pd.read_csv("reviews.csv")
with open("sentiment_dict.json", "r", encoding="utf-8") as f:
    sd = json.load(f)
print(f"리뷰: {len(df)}건")

# ── Part A: 규칙 기반 ──
def rule_sentiment(text):
    tokens = str(text).split()
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in sd["negation"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            # positive 먼저 조회
            ns = sd["positive"].get(nt, sd["negative"].get(nt, 0))
            score += ns * (-1)
            i += 2
            continue
        if t in sd["intensifier"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            ns = sd["positive"].get(nt, sd["negative"].get(nt, 0))
            score += ns * sd["intensifier"][t]
            i += 2
            continue
        # positive 먼저 조회
        ts = sd["positive"].get(t, sd["negative"].get(t, 0))
        score += ts
        i += 1
    return 1 if score > 0 else 0

rule_preds = df["text"].apply(rule_sentiment).values

# ── Part B: ML 기반 ──
X_tr, X_te, y_tr, y_te = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42)

tr_df = pd.DataFrame({"text": X_tr.values, "label": y_tr.values})
pos = tr_df[tr_df["label"] == 1]
neg = tr_df[tr_df["label"] == 0]
neg_over = neg.sample(n=len(pos), replace=True, random_state=42)
bal = pd.concat([pos, neg_over]).reset_index(drop=True)

vec = TfidfVectorizer(sublinear_tf=False, smooth_idf=True)
X_tr_tf = vec.fit_transform(bal["text"])
X_te_tf = vec.transform(X_te)

mdl = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)
mdl.fit(X_tr_tf, bal["label"])
ml_preds = mdl.predict(X_te_tf)

# ── Part C: 비교 + SHAP ──
def calc_metrics(yt, yp):
    return {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "precision_pos": round(float(precision_score(yt, yp, pos_label=1, zero_division=0)), 4),
        "recall_pos": round(float(recall_score(yt, yp, pos_label=1, zero_division=0)), 4),
        "precision_neg": round(float(precision_score(yt, yp, pos_label=0, zero_division=0)), 4),
        "recall_neg": round(float(recall_score(yt, yp, pos_label=0, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
    }

rule_m = calc_metrics(y_te, rule_preds[X_te.index])
ml_m = calc_metrics(y_te, ml_preds)

exp = shap.LinearExplainer(mdl, X_tr_tf)
sv = exp.shap_values(X_te_tf)
fn = vec.get_feature_names_out()
ms = np.mean(sv, axis=0)
if hasattr(ms, 'A1'): ms = ms.A1
ms = np.asarray(ms).flatten()
top_pos = np.argsort(ms)[-5:][::-1]
top_neg = np.argsort(ms)[:5]
sp = [{"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_pos]
sn = [{"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_neg]

summary = f"ML 기반 분석이 규칙 기반 대비 부정 리뷰 탐지에서 우수합니다. 고객 불만의 주요 키워드는 {sn[0]['word']}, {sn[1]['word']} 등으로 긍정 리뷰와 구분됩니다. 부정 키워드 중심의 서비스 개선이 필요합니다."

result = {
    "rule_based": rule_m,
    "ml_based": ml_m,
    "shap_top5_positive": sp,
    "shap_top5_negative": sn,
    "business_summary": summary
}

with open("result_q4.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result_q4.json 저장 완료!")
