"""Grader: 동적 Validator 로더 및 실행기"""
import importlib
import importlib.util
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base_validator import BaseValidator
from .validation_result import ValidationResult


class Grader:
    """config.yaml의 validator 클래스를 동적 로드하여 실행"""

    def __init__(self):
        self._results: List[ValidationResult] = []

    @staticmethod
    def load_validator(
        module_path: str,
        class_name: str,
        submission_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseValidator:
        """모듈 경로와 클래스명으로 Validator 인스턴스 생성 (기존 방식)"""
        module = importlib.import_module(module_path)
        validator_cls = getattr(module, class_name)
        if not issubclass(validator_cls, BaseValidator):
            raise TypeError(f"{class_name}은(는) BaseValidator의 서브클래스가 아닙니다.")
        return validator_cls(submission_dir=submission_dir, config=config)

    @staticmethod
    def load_validator_from_file(
        file_path: str,
        class_name: str,
        submission_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseValidator:
        """파일 경로에서 Validator 클래스를 직접 로드하여 인스턴스 생성 (새 방식)"""
        fp = Path(file_path).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"Validator 파일 없음: {fp}")
        module_name = f"_validator_{fp.stem}_{id(fp)}"
        spec = importlib.util.spec_from_file_location(module_name, str(fp))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validator_cls = getattr(module, class_name)
        if not issubclass(validator_cls, BaseValidator):
            raise TypeError(f"{class_name}은(는) BaseValidator의 서브클래스가 아닙니다.")
        return validator_cls(submission_dir=submission_dir, config=config)

    def run_single(
        self,
        module_path: str,
        class_name: str,
        submission_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """단일 Validator 실행 (모듈 경로 방식)"""
        validator = self.load_validator(module_path, class_name, submission_dir, config)
        result = validator.validate()
        self._results.append(result)
        return result

    def run_from_config(
        self,
        config: Dict[str, Any],
        submission_dir: str,
    ) -> ValidationResult:
        """config dict에서 validator 정보를 추출하여 실행

        validators[].file 있으면 → 파일 경로 로딩 (새 방식)
        validators[].module 있으면 → importlib 로딩 (기존 방식)
        """
        validators = config.get("validators", [])
        if not validators:
            raise ValueError("config에 validators 항목이 없습니다.")
        v = validators[0]

        if "file" in v:
            # 새 방식: 파일 경로 기반 로딩
            question_dir = config.get("_question_dir", "")
            file_path = (Path(question_dir) / v["file"]).resolve()
            validator = self.load_validator_from_file(
                file_path=str(file_path),
                class_name=v["class"],
                submission_dir=submission_dir,
                config=config,
            )
        else:
            # 기존 방식: 모듈 경로 기반 로딩
            validator = self.load_validator(
                module_path=v["module"],
                class_name=v["class"],
                submission_dir=submission_dir,
                config=config,
            )

        result = validator.validate()
        self._results.append(result)
        return result

    @property
    def results(self) -> List[ValidationResult]:
        return self._results

    @property
    def total_earned(self) -> int:
        return sum(r.earned_points for r in self._results)

    @property
    def total_possible(self) -> int:
        return sum(r.total_points for r in self._results)

    def summary_markdown(self, student_id: str = "") -> str:
        """전체 결과 요약 Markdown"""
        lines = [
            "# 전체 채점 결과",
            "",
        ]
        if student_id:
            lines.append(f"**학생 ID:** {student_id}")
            lines.append("")

        lines.append("| 미션 | 점수 | 결과 |")
        lines.append("|------|------|------|")
        for r in self._results:
            status = "PASS" if r.is_passed else "FAIL"
            lines.append(f"| {r.mission_name} | {r.earned_points}/{r.total_points} | {status} |")

        lines.append("")
        lines.append(f"**총점:** {self.total_earned}/{self.total_possible}")
        all_passed = all(r.is_passed for r in self._results)
        lines.append(f"**최종 결과:** {'PASS' if all_passed else 'FAIL'}")
        return "\n".join(lines)
