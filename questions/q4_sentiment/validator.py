"""Q4 감성분석 검증 Validator (7항목, 100점)"""
import json
import unicodedata
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from grading.core.base_validator import BaseValidator
from grading.core.check_item import CheckItem

warnings.filterwarnings("ignore")


class Q4SentimentValidator(BaseValidator):
    mission_id = "aiml_level2_q4_sentiment"
    name = "Q4. 고객리뷰 감성분석"

    def setup(self):
        """sklearn 참조 계산 + 학생 결과 로드"""
        data_dir = self.submission_dir

        # 참조 기준 생성 (verify_q4.py 로직)
        df = pd.read_csv(data_dir / "reviews.csv")

        # NaN label 필터링 + NFC 정규화 (정답 전처리)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df["text"] = df["text"].apply(
            lambda x: unicodedata.normalize("NFC", str(x))
        )

        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        X_tr, X_te, y_tr, y_te = train_test_split(
            df["text"], df["label"], test_size=0.3, random_state=42
        )
        tr_df = pd.DataFrame({"text": X_tr.values, "label": y_tr.values})
        pos = tr_df[tr_df["label"] == 1]
        neg = tr_df[tr_df["label"] == 0]
        neg_o = neg.sample(n=len(pos), replace=True, random_state=42)
        bal = pd.concat([pos, neg_o]).reset_index(drop=True)

        vec = TfidfVectorizer(sublinear_tf=False, smooth_idf=True)
        X_tr_tf = vec.fit_transform(bal["text"])
        X_te_tf = vec.transform(X_te)
        m = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)
        m.fit(X_tr_tf, bal["label"])
        ep = m.predict(X_te_tf)

        self._exp_acc = round(float(accuracy_score(y_te, ep)), 4)
        self._exp_f1 = round(float(f1_score(y_te, ep, average="macro")), 4)

        # rule_based 참조 정확도 계산
        with open(data_dir / "sentiment_dict.json", "r", encoding="utf-8") as f:
            sd = json.load(f)

        def _rule_sentiment(text):
            tokens = str(text).split()
            score = 0.0
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t in sd["negation"] and i + 1 < len(tokens):
                    nt = tokens[i + 1]
                    ns = sd["positive"].get(nt, sd["negative"].get(nt, 0.0))
                    score += ns * (-1.0)
                    i += 2
                    continue
                if t in sd["intensifier"] and i + 1 < len(tokens):
                    nt = tokens[i + 1]
                    ns = sd["positive"].get(nt, sd["negative"].get(nt, 0.0))
                    score += ns * sd["intensifier"][t]
                    i += 2
                    continue
                ts = sd["positive"].get(t, sd["negative"].get(t, 0.0))
                score += ts
                i += 1
            return 1 if score > 0 else 0

        rule_preds = df["text"].apply(_rule_sentiment).values
        self._exp_rule_acc = round(float(accuracy_score(df["label"], rule_preds)), 4)

        # 학생 결과 로드
        result_path = self._resolve_file("result_q4.json", "result_file_override")
        with open(result_path, "r", encoding="utf-8") as f:
            self._ans = json.load(f)

        # 코드 로드
        code_path = self._resolve_file("q4_solution.py", "solution_file_override")
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                self._code = f.read()
        except Exception:
            self._code = ""

    def build_checklist(self):
        ans = self._ans
        exp_acc = self._exp_acc
        exp_f1 = self._exp_f1
        code = self._code

        req_metrics = [
            "accuracy", "precision_pos", "recall_pos",
            "precision_neg", "recall_neg", "f1_macro",
        ]

        # 1. rule_based 구조 + 정확도 검증 (10점)
        exp_rule_acc = self._exp_rule_acc

        def check_rule_based():
            rb = ans.get("rule_based", {})
            missing = [k for k in req_metrics if k not in rb]
            if missing:
                return (False, f"누락 지표: {missing}")
            rb_acc = rb.get("accuracy", 0)
            if abs(rb_acc - exp_rule_acc) > 0.03:
                return (False, f"accuracy 불일치 (결과: {rb_acc}, 기대: {exp_rule_acc})")
            return True

        self.checklist.add_item(CheckItem(
            id="1",
            description="rule_based 6개 지표 + accuracy 검증 (오차 < 0.03)",
            points=10,
            validator=check_rule_based,
        ))

        # 2. ml_based accuracy (15점) — 허용 오차 0.01
        def check_ml_acc():
            ml = ans.get("ml_based", {})
            ml_acc = ml.get("accuracy", 0)
            if abs(ml_acc - exp_acc) < 0.01:
                return True
            return (False, f"결과: {ml_acc}, 기대: {exp_acc}")

        self.checklist.add_item(CheckItem(
            id="2",
            description="ML accuracy (허용 오차 < 0.01)",
            points=15,
            validator=check_ml_acc,
            hint="데이터 전처리, 오버샘플링, 모델 학습 순서를 확인하세요.",
        ))

        # 3. ml_based f1 (15점) — 허용 오차 0.01
        def check_ml_f1():
            ml = ans.get("ml_based", {})
            ml_f1 = ml.get("f1_macro", 0)
            if abs(ml_f1 - exp_f1) < 0.01:
                return True
            return (False, f"결과: {ml_f1}, 기대: {exp_f1}")

        self.checklist.add_item(CheckItem(
            id="3",
            description="ML F1 macro (허용 오차 < 0.01)",
            points=15,
            validator=check_ml_f1,
        ))

        # 4. SHAP 부호 (15점)
        def check_shap():
            sp = ans.get("shap_top5_positive", [])
            sn = ans.get("shap_top5_negative", [])
            if (
                len(sp) == 5
                and len(sn) == 5
                and all(x.get("shap_value", 0) > 0 for x in sp)
                and all(x.get("shap_value", 0) < 0 for x in sn)
            ):
                return True
            msg = f"positive: {len(sp)}개, negative: {len(sn)}개"
            return (False, msg)

        self.checklist.add_item(CheckItem(
            id="4",
            description="SHAP 부호 정확 (positive 5개 + negative 5개)",
            points=15,
            validator=check_shap,
            hint="shap.LinearExplainer로 SHAP 값을 계산하세요.",
        ))

        # 5. fit/transform 분리 (15점)
        def check_data_leakage():
            if "fit_transform" in code and ".transform(" in code:
                return True
            return (False, "fit_transform + .transform() 패턴 미확인")

        self.checklist.add_item(CheckItem(
            id="5",
            description="데이터 누수 방지 (fit/transform 분리)",
            points=15,
            validator=check_data_leakage,
            hint="학습 데이터에 fit_transform, 테스트 데이터에 transform 사용",
            ai_trap="전체 데이터에 fit_transform 사용 시 데이터 누수 발생",
        ))

        # 6. business_summary (15점)
        def check_summary():
            bs = ans.get("business_summary", "")
            has_pos = "긍정" in bs
            has_neg = "부정" in bs
            if has_pos and has_neg and len(bs) >= 20:
                return True
            parts = []
            if not has_pos:
                parts.append("'긍정' 누락")
            if not has_neg:
                parts.append("'부정' 누락")
            if len(bs) < 20:
                parts.append(f"길이 부족({len(bs)}자)")
            return (False, ", ".join(parts))

        self.checklist.add_item(CheckItem(
            id="6",
            description="비즈니스 요약 ('긍정'/'부정' 포함, 20자+)",
            points=15,
            validator=check_summary,
        ))

        # 7. ml_based 6개 지표 완성 (15점)
        def check_ml_metrics():
            ml = ans.get("ml_based", {})
            if all(k in ml for k in req_metrics):
                return True
            missing = [k for k in req_metrics if k not in ml]
            return (False, f"누락 지표: {missing}")

        self.checklist.add_item(CheckItem(
            id="7",
            description="ml_based 6개 지표 완성",
            points=15,
            validator=check_ml_metrics,
        ))
