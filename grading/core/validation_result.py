"""ValidationResult: JSON 및 Markdown 리포트 생성"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .check_item import CheckItem, CheckStatus


@dataclass
class ValidationResult:
    mission_id: str
    mission_name: str
    items: List[CheckItem]
    total_points: int
    earned_points: int
    pass_count: int
    fail_count: int
    passing_threshold: float
    is_passed: bool
    student_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "student_id": self.student_id,
            "timestamp": self.timestamp,
            "total_points": self.total_points,
            "earned_points": self.earned_points,
            "score_ratio": round(self.earned_points / self.total_points, 4)
            if self.total_points > 0 else 0,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "passing_threshold": self.passing_threshold,
            "is_passed": self.is_passed,
            "checks": [
                {
                    "id": item.id,
                    "description": item.description,
                    "points": item.points,
                    "earned_points": item.earned_points,
                    "status": item.status.value,
                    "hint": item.hint,
                    "ai_trap": item.ai_trap,
                    "error_message": item.error_message,
                }
                for item in self.items
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.mission_name} 채점 결과",
            "",
        ]
        if self.student_id:
            lines.append(f"**학생 ID:** {self.student_id}")
        lines.append(f"**채점 시각:** {self.timestamp}")
        lines.append("")
        lines.append("=" * 55)
        lines.append("")

        for item in self.items:
            if item.status == CheckStatus.PASSED:
                mark = "[PASS]"
            elif item.status == CheckStatus.FAILED:
                mark = "[FAIL]"
            else:
                mark = "[ERROR]"

            points_str = f"{item.earned_points}/{item.points}점"
            lines.append(f"  {mark} {item.id}. {item.description} ({points_str})")

            if item.status != CheckStatus.PASSED:
                if item.error_message:
                    lines.append(f"         -> {item.error_message}")
                if item.hint:
                    lines.append(f"         힌트: {item.hint}")
                if item.ai_trap:
                    lines.append(f"         AI 함정: {item.ai_trap}")

        lines.append("")
        lines.append("=" * 55)
        lines.append(f"  점수: {self.earned_points}/{self.total_points}")
        lines.append(f"  합격 기준: {self.passing_threshold * 100:.0f}%")
        result_str = "PASS" if self.is_passed else "FAIL"
        lines.append(f"  결과: {result_str}")
        lines.append("=" * 55)

        return "\n".join(lines)
