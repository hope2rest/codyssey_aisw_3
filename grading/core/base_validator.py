"""BaseValidator: Template Method 패턴 추상 베이스 클래스"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

from .checklist import Checklist
from .validation_result import ValidationResult


class BaseValidator(ABC):
    """모든 Validator가 상속하는 추상 베이스 클래스.

    Template Method 패턴:
        setup() → build_checklist() → validate() → teardown()
    """

    def __init__(self, submission_dir: str, config: Optional[Dict[str, Any]] = None):
        self.submission_dir = Path(submission_dir).resolve()
        self.config = config or {}
        self.checklist = Checklist()
        self._context: Dict[str, Any] = {}

    @property
    @abstractmethod
    def mission_id(self) -> str:
        """미션 고유 ID (예: aiml_level1_q1_svd)"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """미션 이름"""
        ...

    @property
    def passing_score(self) -> float:
        return self.config.get("passing_score", 0.95)

    def _resolve_file(self, default_name: str, override_key: str) -> Path:
        """config에 override 경로가 있으면 사용, 없으면 submission_dir/default_name"""
        override = self.config.get(override_key)
        if override:
            return Path(override).resolve()
        return self.submission_dir / default_name

    def setup(self) -> None:
        """데이터 로드 및 참조 답안 생성. 서브클래스에서 오버라이드."""
        pass

    @abstractmethod
    def build_checklist(self) -> None:
        """CheckItem 등록. 서브클래스에서 반드시 구현."""
        ...

    def teardown(self) -> None:
        """정리 작업. 서브클래스에서 필요 시 오버라이드."""
        pass

    def validate(self) -> ValidationResult:
        """Template Method: 전체 검증 프로세스 실행"""
        try:
            self.setup()
            self.build_checklist()
            self.checklist.execute_all()
            self.teardown()
        except FileNotFoundError as e:
            # 제출물 누락 시 0점 결과 반환
            from .check_item import CheckItem, CheckStatus
            error_item = CheckItem(
                id="0",
                description="제출물 파일 확인",
                points=0,
                validator=lambda: False,
            )
            error_item.status = CheckStatus.ERROR
            error_item.error_message = f"파일 없음: {e}"
            return ValidationResult(
                mission_id=self.mission_id,
                mission_name=self.name,
                items=[error_item],
                total_points=100,
                earned_points=0,
                pass_count=0,
                fail_count=1,
                passing_threshold=self.passing_score,
                is_passed=False,
            )

        return ValidationResult(
            mission_id=self.mission_id,
            mission_name=self.name,
            items=self.checklist.items,
            total_points=self.checklist.total_points,
            earned_points=self.checklist.earned_points,
            pass_count=self.checklist.pass_count,
            fail_count=self.checklist.fail_count,
            passing_threshold=self.passing_score,
            is_passed=self.checklist.is_passed(self.passing_score),
        )
