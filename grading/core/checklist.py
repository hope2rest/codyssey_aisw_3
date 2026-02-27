"""Checklist: CheckItem 리스트 관리 및 점수 집계"""
from typing import List, Optional
from .check_item import CheckItem, CheckStatus


class Checklist:
    def __init__(self):
        self._items: List[CheckItem] = []

    def add_item(self, item: CheckItem) -> None:
        self._items.append(item)

    def execute_all(self) -> List[CheckItem]:
        for item in self._items:
            item.execute()
        return self._items

    @property
    def items(self) -> List[CheckItem]:
        return self._items

    @property
    def total_points(self) -> int:
        return sum(item.points for item in self._items)

    @property
    def earned_points(self) -> int:
        return sum(item.earned_points for item in self._items)

    @property
    def pass_count(self) -> int:
        return sum(1 for item in self._items if item.status == CheckStatus.PASSED)

    @property
    def fail_count(self) -> int:
        return sum(1 for item in self._items
                   if item.status in (CheckStatus.FAILED, CheckStatus.ERROR))

    @property
    def score_ratio(self) -> float:
        if self.total_points == 0:
            return 0.0
        return self.earned_points / self.total_points

    def is_passed(self, threshold: float = 0.95) -> bool:
        return self.score_ratio >= threshold
