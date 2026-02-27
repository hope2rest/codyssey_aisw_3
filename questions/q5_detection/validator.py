"""Q5 결함검출 검증 Validator (10항목, 100점)"""
import ast
import json
import os
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from grading.core.base_validator import BaseValidator
from grading.core.check_item import CheckItem


LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]


class Q5DetectionValidator(BaseValidator):
    mission_id = "aiml_level3_q5_detection"
    name = "Q5. 부품결함검출 딥러닝기초 전이학습"

    def setup(self):
        """데이터 정제 기준 산출 + 코드 분석 + 결과 로드"""
        data_dir = self.submission_dir

        # 정답 산출 (verify_q5.py 로직)
        df = pd.read_csv(data_dir / "inspection_log.csv", dtype={"part_id": str})
        df = df.drop_duplicates(subset="part_id", keep="first")
        df["part_id"] = df["part_id"].apply(lambda x: str(x).zfill(4))
        df["defect_type"] = df["defect_type"].apply(
            lambda x: unicodedata.normalize("NFC", str(x).strip())
        )
        df = df[df["defect_type"].isin(LABELS)]
        df["inspector_note"] = df["inspector_note"].fillna("")

        def _valid(pid):
            p = data_dir / "part_images" / f"{pid}.png"
            if not p.exists() or p.stat().st_size == 0:
                return False
            try:
                img = Image.open(p)
                img.verify()
                return True
            except Exception:
                return False

        df = df[df["part_id"].apply(_valid)]
        self._expected_samples = len(df)
        self._expected_dist = df["defect_type"].value_counts().to_dict()

        # 코드 분석
        # Q5 문제는 q6_solution.py를 사용 (파일명 매핑)
        solution_name = self.config.get("solution_file", "q6_solution.py")
        code_path = self._resolve_file(solution_name, "solution_file_override")
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                self._code = f.read()
            tree = ast.parse(self._code)
            self._cls = [
                n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            ]
            self._funcs = [
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            ]
        except Exception:
            self._code = ""
            self._cls = []
            self._funcs = []

        # 학생 결과 로드
        result_name = self.config.get("result_file", "result_q6.json")
        result_path = self._resolve_file(result_name, "result_file_override")
        with open(result_path, "r", encoding="utf-8") as f:
            self._ans = json.load(f)

    def build_checklist(self):
        ans = self._ans
        code = self._code
        cls = self._cls
        expected_samples = self._expected_samples
        expected_dist = self._expected_dist

        # 1. 클래스 구현 (10점)
        def check_classes():
            if "DefectImageLoader" in cls and "InspectionLogProcessor" in cls:
                return True
            return (False, f"발견된 클래스: {cls}")

        self.checklist.add_item(CheckItem(
            id="1",
            description="클래스 구현 (DefectImageLoader + InspectionLogProcessor)",
            points=10,
            validator=check_classes,
        ))

        # 2. 데이터 정제 결과 (10점)
        def check_data_cleaning():
            ds = ans.get("data_summary", {})
            tvs = ds.get("total_valid_samples", 0)
            ld = ds.get("label_distribution", {})

            if abs(tvs - expected_samples) > 2:
                return (
                    False,
                    f"유효 샘플 수 불일치 (기대: {expected_samples}, 결과: {tvs})",
                )
            for lbl in LABELS:
                exp = expected_dist.get(lbl, 0)
                got = ld.get(lbl, 0)
                if abs(exp - got) > 2:
                    return (
                        False,
                        f"레이블 분포 불일치: {lbl} (기대: {exp}, 결과: {got})",
                    )
            return True

        self.checklist.add_item(CheckItem(
            id="2",
            description="데이터 정제 (total_valid_samples + label_distribution)",
            points=10,
            validator=check_data_cleaning,
            hint="중복 제거, NFC 정규화, 유효 레이블 필터, 이미지 존재 확인",
        ))

        # 3. conv2d 직접 구현 + Sobel (10점)
        def check_conv2d_sobel():
            has_conv2d = "def conv2d" in code
            no_cv2_filter = "filter2D" not in code
            has_sobel = (
                "sobel" in code.lower()
                or ("[-1, 0, 1]" in code and "[-2, 0, 2]" in code)
                or "[-1,0,1]" in code
            )
            if has_conv2d and no_cv2_filter and has_sobel:
                return True
            reasons = []
            if not has_conv2d:
                reasons.append("conv2d 함수 없음")
            if not no_cv2_filter:
                reasons.append("filter2D 사용 감지")
            if not has_sobel:
                reasons.append("Sobel 커널 미정의")
            return (False, ", ".join(reasons))

        self.checklist.add_item(CheckItem(
            id="3",
            description="conv2d 직접 구현 + Sobel 커널",
            points=10,
            validator=check_conv2d_sobel,
        ))

        # 4. 규칙 기반 결과 (10점)
        def check_rule_based():
            rb = ans.get("rule_based", {})
            rb_acc = rb.get("test_accuracy", 0)
            if 0.5 <= rb_acc <= 0.95 and rb.get("method") == "edge_threshold_binary":
                return True
            return (False, f"accuracy={rb_acc}, method={rb.get('method')}")

        self.checklist.add_item(CheckItem(
            id="4",
            description="규칙 기반 (accuracy 0.5~0.95, edge_threshold_binary)",
            points=10,
            validator=check_rule_based,
        ))

        # 5. ML 기반 결과 (10점)
        def check_ml_based():
            ml = ans.get("ml_based", {})
            ml_acc = ml.get("test_accuracy", 0)
            ml_f1 = ml.get("test_f1_macro", 0)
            pca_n = ml.get("pca_n_components", 0)
            if ml_acc > 0.85 and ml_f1 > 0.7 and 50 <= pca_n <= 400:
                return True
            return (
                False,
                f"acc={ml_acc}, f1={ml_f1}, PCA={pca_n}",
            )

        self.checklist.add_item(CheckItem(
            id="5",
            description="ML 기반 (acc>0.85, f1>0.7, PCA 50~400)",
            points=10,
            validator=check_ml_based,
        ))

        # 6. NN Forward Pass (10점)
        def check_nn_forward():
            nn = ans.get("nn_forward", {})
            nn_acc = nn.get("test_accuracy", 0)
            has_relu = (
                "relu" in code.lower()
                or "maximum(0" in code
                or "np.maximum(0" in code
            )
            has_softmax = "softmax" in code.lower() or "exp(" in code
            has_weight_load = "pretrained_nn_weights" in code

            if nn_acc > 0.8 and has_relu and has_softmax and has_weight_load:
                return True
            reasons = []
            if nn_acc <= 0.8:
                reasons.append(f"accuracy 낮음({nn_acc})")
            if not has_relu:
                reasons.append("ReLU 미구현")
            if not has_softmax:
                reasons.append("Softmax 미구현")
            if not has_weight_load:
                reasons.append("가중치 미로드")
            return (False, ", ".join(reasons))

        self.checklist.add_item(CheckItem(
            id="6",
            description="NN Forward Pass (acc>0.8, ReLU+Softmax+가중치 로드)",
            points=10,
            validator=check_nn_forward,
        ))

        # 7. 전이학습 비교 (10점)
        def check_transfer():
            pre = ans.get("pretrained", {})
            pre_acc = pre.get("test_accuracy", 0)
            tg = ans.get("transfer_gain")

            if not (pre_acc > 0.85 and tg is not None and isinstance(tg, (int, float))):
                return (
                    False,
                    f"pre_acc={pre_acc}, transfer_gain={tg}",
                )

            cf1 = pre.get("class_f1", {})
            cm = pre.get("confusion_matrix", [])
            has_all_f1 = all(lbl in cf1 for lbl in LABELS)
            has_cm = len(cm) == 5 and all(len(row) == 5 for row in cm)

            if has_all_f1 and has_cm:
                return True
            return (False, "class_f1 또는 confusion_matrix 불완전")

        self.checklist.add_item(CheckItem(
            id="7",
            description="전이학습 비교 (pre_acc>0.85 + class_f1 + confusion_matrix)",
            points=10,
            validator=check_transfer,
        ))

        # 8. 개선 실험 (10점)
        def check_improvement():
            imp = ans.get("improvement", {})
            before = imp.get("before_f1", 0)
            after = imp.get("after_f1", 0)
            mic = imp.get("most_improved_class", "")

            if not (before > 0 and after > 0 and mic in LABELS):
                return (False, f"before_f1={before}, after_f1={after}, class={mic}")

            has_cw = "class_weight" in code and "balanced" in code
            if has_cw:
                return True
            return (False, "class_weight='balanced' 미사용")

        self.checklist.add_item(CheckItem(
            id="8",
            description="개선 실험 (class_weight='balanced' 사용)",
            points=10,
            validator=check_improvement,
        ))

        # 9. 비즈니스 보고서 (10점)
        def check_report():
            report = ans.get("report", {})
            req_sections = [
                "purpose",
                "key_results",
                "transfer_learning_effect",
                "improvement_suggestion",
            ]
            for sec in req_sections:
                val = report.get(sec, "")
                if not isinstance(val, str) or len(val) < 10:
                    missing = [
                        s for s in req_sections
                        if not isinstance(report.get(s, ""), str)
                        or len(report.get(s, "")) < 10
                    ]
                    return (False, f"섹션 누락/내용 부족: {missing}")
            return True

        self.checklist.add_item(CheckItem(
            id="9",
            description="비즈니스 보고서 (4섹션 완성)",
            points=10,
            validator=check_report,
        ))

        # 10. 라이브러리 적절 사용 (10점)
        def check_libraries():
            uses_tfidf = "TfidfVectorizer" in code
            uses_lr = "LogisticRegression" in code
            uses_pca = "PCA" in code
            no_keras = "keras" not in code.lower() and "torch" not in code.lower()

            if uses_tfidf and uses_lr and uses_pca and no_keras:
                return True
            reasons = []
            if not uses_tfidf:
                reasons.append("TfidfVectorizer 미사용")
            if not uses_lr:
                reasons.append("LogisticRegression 미사용")
            if not uses_pca:
                reasons.append("PCA 미사용")
            if not no_keras:
                reasons.append("keras/torch 사용 감지")
            return (False, ", ".join(reasons))

        self.checklist.add_item(CheckItem(
            id="10",
            description="라이브러리 적절 사용 (sklearn O, keras/torch X)",
            points=10,
            validator=check_libraries,
            ai_trap="keras/torch 대신 sklearn만 사용해야 함",
        ))
