"""CheckItem 데이터클래스 및 CheckStatus enum"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any


class CheckStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class CheckItem:
    id: str
    description: str
    points: int
    validator: Callable[[], Any]
    hint: str = ""
    ai_trap: str = ""
    status: CheckStatus = CheckStatus.PENDING
    error_message: str = ""
    earned_points: int = 0

    def execute(self) -> 'CheckItem':
        """validator 호출 → bool이면 pass/fail, int 반환 시 부분 점수 지원"""
        try:
            result = self.validator()
            if isinstance(result, bool):
                if result:
                    self.status = CheckStatus.PASSED
                    self.earned_points = self.points
                else:
                    self.status = CheckStatus.FAILED
                    self.earned_points = 0
            elif isinstance(result, int):
                # 부분 점수: 0 이상 max points 이하
                self.earned_points = max(0, min(result, self.points))
                self.status = CheckStatus.PASSED if self.earned_points > 0 else CheckStatus.FAILED
            elif isinstance(result, tuple) and len(result) == 2:
                # (bool_or_int, message) 형태
                value, message = result
                self.error_message = str(message)
                if isinstance(value, bool):
                    self.status = CheckStatus.PASSED if value else CheckStatus.FAILED
                    self.earned_points = self.points if value else 0
                elif isinstance(value, int):
                    self.earned_points = max(0, min(value, self.points))
                    self.status = CheckStatus.PASSED if self.earned_points > 0 else CheckStatus.FAILED
            else:
                self.status = CheckStatus.FAILED
                self.earned_points = 0
        except Exception as e:
            self.status = CheckStatus.ERROR
            self.error_message = str(e)
            self.earned_points = 0
        return self
