"""Q1 SVD 검증 Validator (5항목, 100점)"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from grading.core.base_validator import BaseValidator
from grading.core.check_item import CheckItem


class Q1SvdValidator(BaseValidator):
    mission_id = "aiml_level1_q1_svd"
    name = "Q1. SVD 기반 차원축소 복원분석"

    def setup(self):
        """데이터 로드 → 참조 답안 생성 → 학생 result JSON 로드"""
        data_dir = self.submission_dir
        csv_path = data_dir / "sensor_data.csv"

        # 참조 답안 생성 (verify_q1.py gen_ref 로직)
        df = pd.read_csv(csv_path, header=None)
        d = df.values.astype(float)
        for j in range(d.shape[1]):
            col = d[:, j]
            m = np.isnan(col)
            if m.any():
                d[m, j] = np.nanmean(col)
        s = np.std(d, axis=0, ddof=0)
        d = d[:, s > 1e-10]
        m = np.mean(d, axis=0)
        s = np.std(d, axis=0, ddof=0)
        X = (d - m) / s
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        evr = (S ** 2) / np.sum(S ** 2)
        cum = np.cumsum(evr)
        k = int(np.argmax(cum >= 0.95) + 1)
        mse = float(np.mean((X - (U[:, :k] * S[:k]) @ Vt[:k, :]) ** 2))

        self._ref = {
            "optimal_k": k,
            "cumulative_variance_at_k": round(float(cum[k - 1]), 6),
            "reconstruction_mse": round(float(mse), 6),
            "top_5_singular_values": [round(float(sv), 6) for sv in S[:5]],
            "explained_variance_ratio_top5": [round(float(r), 6) for r in evr[:5]],
        }

        # 학생 결과 로드
        result_path = data_dir / "result_q1.json"
        with open(result_path, "r", encoding="utf-8") as f:
            self._ans = json.load(f)

    def build_checklist(self):
        ref = self._ref
        ans = self._ans

        # 1. optimal_k (20점)
        def check_optimal_k():
            a = ans.get("optimal_k")
            r = ref["optimal_k"]
            if a is not None and a == r:
                return True
            return (False, f"결과: {a}, 기대: {r}")

        self.checklist.add_item(CheckItem(
            id="1",
            description="optimal_k 일치",
            points=20,
            validator=check_optimal_k,
            hint="cumulative variance >= 0.95인 최소 k를 구하세요.",
            ai_trap="ddof=0 사용 필요 (모집단 표준편차)",
        ))

        # 2. cumulative_variance_at_k (20점)
        def check_cumulative_variance():
            a = ans.get("cumulative_variance_at_k")
            r = ref["cumulative_variance_at_k"]
            if a is not None and abs(a - r) < 1e-4:
                return True
            return (False, f"결과: {a}, 기대: {r}")

        self.checklist.add_item(CheckItem(
            id="2",
            description="cumulative_variance_at_k 정확도",
            points=20,
            validator=check_cumulative_variance,
            hint="S**2 / sum(S**2)로 분산 비율을 계산하세요.",
        ))

        # 3. reconstruction_mse (20점)
        def check_mse():
            a = ans.get("reconstruction_mse")
            r = ref["reconstruction_mse"]
            if a is not None and abs(a - r) < 1e-4:
                return True
            return (False, f"결과: {a}, 기대: {r}")

        self.checklist.add_item(CheckItem(
            id="3",
            description="reconstruction_mse 정확도",
            points=20,
            validator=check_mse,
            hint="X_reconstructed = U[:,:k] * S[:k] @ Vt[:k,:]",
        ))

        # 4. top_5_singular_values (20점)
        def check_sv():
            sv = ans.get("top_5_singular_values", [])
            rsv = ref["top_5_singular_values"]
            if len(sv) == 5 and all(abs(a - b) < 1e-3 for a, b in zip(sv, rsv)):
                return True
            return (False, f"결과: {sv}, 기대: {rsv}")

        self.checklist.add_item(CheckItem(
            id="4",
            description="top_5_singular_values 정확도",
            points=20,
            validator=check_sv,
            hint="SVD 분해 후 S 벡터의 처음 5개 값.",
        ))

        # 5. explained_variance_ratio_top5 (20점)
        def check_evr():
            se = ans.get("explained_variance_ratio_top5", [])
            re_ = ref["explained_variance_ratio_top5"]
            if len(se) == 5 and all(abs(a - b) < 1e-4 for a, b in zip(se, re_)):
                return True
            return (False, f"결과: {se}, 기대: {re_}")

        self.checklist.add_item(CheckItem(
            id="5",
            description="explained_variance_ratio_top5 정확도",
            points=20,
            validator=check_evr,
            hint="각 특이값의 분산 비율 = S[i]**2 / sum(S**2)",
        ))
