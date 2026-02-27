"""Q3 CV 검증 Validator (7항목, 100점)"""
import ast
import json
import os
import re
import numpy as np
from pathlib import Path

from grading.core.base_validator import BaseValidator
from grading.core.check_item import CheckItem


class Q3CvValidator(BaseValidator):
    mission_id = "aiml_level2_q3_cv"
    name = "Q3. 이미지 객체카운팅 규칙기반 한계분석"

    @property
    def passing_score(self) -> float:
        return self.config.get("passing_score", 0.85)

    def setup(self):
        """labels, 코드, 결과 로드"""
        data_dir = self.submission_dir

        with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
            self._labels = json.load(f)

        images_dir = data_dir / "images"
        self._valid = [
            n for n in self._labels
            if (images_dir / f"{n}.png").exists()
        ]

        # 코드 파싱
        code_path = data_dir / "q3_solution.py"
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                self._code = f.read()
            tree = ast.parse(self._code)
            self._funcs = [
                n.name for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
            ]
        except Exception:
            self._code = ""
            self._funcs = []

        # 학생 결과 로드
        with open(data_dir / "result_q3.json", "r", encoding="utf-8") as f:
            self._ans = json.load(f)

    def build_checklist(self):
        ans = self._ans
        labels = self._labels
        valid = self._valid
        code = self._code
        funcs = self._funcs
        kr = re.compile(r"[가-힣]")

        # 1. conv2d 직접 구현 (15점)
        def check_conv2d():
            if "conv2d" in funcs and "filter2D" not in code:
                return True
            if "conv2d" not in funcs:
                return (False, "conv2d 함수 미구현")
            return (False, "filter2D 사용 감지 - 직접 구현 필요")

        self.checklist.add_item(CheckItem(
            id="1",
            description="conv2d 직접 구현 (filter2D 미사용)",
            points=15,
            validator=check_conv2d,
            hint="NumPy로 합성곱 연산을 직접 구현하세요.",
        ))

        # 2. valid images only (15점)
        def check_valid_images():
            preds = ans.get("predictions", {})
            if set(preds.keys()) == set(valid):
                return True
            extra = set(preds.keys()) - set(valid)
            missing = set(valid) - set(preds.keys())
            msg = ""
            if extra:
                msg += f"존재하지 않는 이미지 포함: {extra} "
            if missing:
                msg += f"누락: {missing}"
            return (False, msg.strip())

        self.checklist.add_item(CheckItem(
            id="2",
            description=f"유효 이미지만 분석 ({len(valid)}개, test_01 제외)",
            points=15,
            validator=check_valid_images,
            hint="labels.json 중 실제 이미지 파일이 있는 것만 분석하세요.",
            ai_trap="test_01은 이미지가 없으므로 제외해야 함",
        ))

        # 3. easy MAE - 부분 점수 (15점)
        def check_easy_mae():
            preds = ans.get("predictions", {})
            easy = {
                k: v for k, v in labels.items()
                if k.startswith("easy") and k in valid
            }
            easy_errs = [abs(preds.get(k, 0) - v) for k, v in easy.items()]
            mae = float(np.mean(easy_errs)) if easy_errs else 999

            if mae == 0.0:
                return (0, "과적합 또는 데이터 누수 의심 (MAE=0.0)")
            elif mae <= 3.0:
                return 15  # 만점
            elif mae <= 10.0:
                return (10, f"easy MAE: {mae:.2f} (부분 점수 10/15)")
            elif mae <= 30.0:
                return (5, f"easy MAE: {mae:.2f} (부분 점수 5/15)")
            else:
                return (0, f"오차 기준치 초과 (MAE: {mae:.2f})")

        self.checklist.add_item(CheckItem(
            id="3",
            description="easy 카테고리 MAE",
            points=15,
            validator=check_easy_mae,
            hint="easy 이미지는 MAE <= 3.0이면 만점.",
            ai_trap="MAE=0.0은 과적합으로 0점 처리",
        ))

        # 4. metrics 3카테고리 (10점)
        def check_metrics():
            m = ans.get("metrics", {})
            if all(c in m for c in ["easy", "medium", "hard"]):
                return True
            return (False, "easy/medium/hard 3카테고리 모두 필요")

        self.checklist.add_item(CheckItem(
            id="4",
            description="metrics 3카테고리 완성",
            points=10,
            validator=check_metrics,
        ))

        # 5. failure_reasons 한국어 (20점)
        def check_failure_reasons():
            fr = ans.get("failure_reasons", [])
            if len(fr) >= 3 and all(len(r) >= 20 and kr.search(r) for r in fr):
                return True
            if len(fr) < 3:
                return (False, f"{len(fr)}개 제출 (최소 3개 필요)")
            return (False, "각 항목이 20자 이상 한국어여야 합니다.")

        self.checklist.add_item(CheckItem(
            id="5",
            description="failure_reasons (3개 이상, 한국어, 20자+)",
            points=20,
            validator=check_failure_reasons,
        ))

        # 6. why_learning_based (15점)
        def check_why_learning():
            wlb = ans.get("why_learning_based", "")
            if 30 <= len(wlb) <= 200 and kr.search(wlb):
                return True
            msg = f"길이: {len(wlb)}자"
            if not kr.search(wlb):
                msg += ", 한국어 미포함"
            return (False, msg)

        self.checklist.add_item(CheckItem(
            id="6",
            description="why_learning_based (30~200자, 한국어)",
            points=15,
            validator=check_why_learning,
        ))

        # 7. worst_case_image (10점)
        def check_worst_case():
            wc = ans.get("worst_case_image", "")
            if wc.startswith("hard"):
                return True
            return (False, f"결과: '{wc}' ('hard'로 시작해야 함)")

        self.checklist.add_item(CheckItem(
            id="7",
            description="worst_case_image가 hard 카테고리",
            points=10,
            validator=check_worst_case,
            hint="가장 큰 오차가 발생한 이미지는 hard 난이도여야 합니다.",
        ))
