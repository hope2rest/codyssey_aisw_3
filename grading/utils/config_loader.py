"""YAML 설정 로더 — questions/ 자동 탐색 방식"""
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import yaml


# 프로젝트 루트 (grading/utils/ 의 2단계 상위)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 탐색 대상 디렉토리
_QUESTIONS_DIR = _PROJECT_ROOT / "questions"
_MISSIONS_DIR = Path(__file__).resolve().parent.parent / "missions"

# 캐시
_config_cache: Dict[str, Dict[str, Any]] = {}
_config_path_cache: Dict[str, Path] = {}
_scanned = False


def _scan_configs() -> None:
    """questions/*/config.yaml + grading/missions/ fallback 스캔 (최초 1회)"""
    global _scanned
    if _scanned:
        return

    # 1) questions/ 폴더 우선 스캔
    if _QUESTIONS_DIR.exists():
        for cfg_path in sorted(_QUESTIONS_DIR.glob("*/config.yaml")):
            _register_config(cfg_path)

    # 2) grading/missions/ fallback 스캔 (하위 호환)
    if _MISSIONS_DIR.exists():
        for cfg_path in sorted(_MISSIONS_DIR.rglob("config.yaml")):
            _register_config(cfg_path)

    _scanned = True


def _register_config(cfg_path: Path) -> None:
    """config.yaml 하나를 읽어 캐시에 등록 (중복 mission_id는 첫 등록 우선)"""
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        mid = data.get("mission_id")
        if mid and mid not in _config_cache:
            # data_dir을 절대 경로로 해석하여 저장
            data_dir = data.get("data_dir", ".")
            data["_question_dir"] = str(cfg_path.parent)
            data["_data_path"] = str((cfg_path.parent / data_dir).resolve())
            data["_config_path"] = str(cfg_path)
            _config_cache[mid] = data
            _config_path_cache[mid] = cfg_path
    except Exception:
        pass  # 잘못된 YAML은 무시


# ──────────────────────────────────────────────
# 기존 API (하위 호환)
# ──────────────────────────────────────────────

def get_config_path(mission_id: str) -> Path:
    """미션 ID로 config.yaml 경로 반환"""
    _scan_configs()
    path = _config_path_cache.get(mission_id)
    if path is None:
        raise ValueError(
            f"알 수 없는 미션 ID: {mission_id}\n"
            f"사용 가능: {', '.join(_config_cache.keys())}"
        )
    return path


def load_config(mission_id: str) -> Dict[str, Any]:
    """미션 ID로 YAML 설정 로드"""
    _scan_configs()
    cfg = _config_cache.get(mission_id)
    if cfg is None:
        raise ValueError(
            f"알 수 없는 미션 ID: {mission_id}\n"
            f"사용 가능: {', '.join(_config_cache.keys())}"
        )
    return dict(cfg)  # 사본 반환


def get_all_mission_ids() -> List[str]:
    """동적으로 발견된 전체 미션 ID 목록"""
    _scan_configs()
    return list(_config_cache.keys())


class _LazyMissionIds:
    """ALL_MISSION_IDS를 lazy 평가하는 프록시 객체"""
    def __iter__(self):
        return iter(get_all_mission_ids())

    def __len__(self):
        return len(get_all_mission_ids())

    def __contains__(self, item):
        return item in get_all_mission_ids()

    def __repr__(self):
        return repr(get_all_mission_ids())

    def __list__(self):
        return get_all_mission_ids()


ALL_MISSION_IDS = _LazyMissionIds()


# ──────────────────────────────────────────────
# 신규 API: Plug-and-Play 지원
# ──────────────────────────────────────────────

_STAGES = ["advanced", "basic", "intro"]


def build_question_map() -> Dict[Tuple[str, str], Tuple[str, Path]]:
    """(stage, qnum) → (mission_id, data_path) 매핑 생성

    모든 stage에 대해 qnum을 매핑한다.
    """
    _scan_configs()
    result: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    for mid, cfg in _config_cache.items():
        qnum = cfg.get("qnum")
        if qnum is None:
            continue
        data_path = Path(cfg["_data_path"])
        for stage in _STAGES:
            result[(stage, str(qnum))] = (mid, data_path)
    return result


def build_no_solution_required() -> Set[Tuple[str, str]]:
    """solution이 필요 없는 문항의 (stage, qnum) 세트 반환"""
    _scan_configs()
    result: Set[Tuple[str, str]] = set()
    for cfg in _config_cache.values():
        if not cfg.get("solution_required", True):
            qnum = cfg.get("qnum")
            if qnum is not None:
                for stage in _STAGES:
                    result.add((stage, str(qnum)))
    return result


def build_submission_dirs() -> Dict[str, Path]:
    """mission_id → data_path 매핑 반환"""
    _scan_configs()
    return {
        mid: Path(cfg["_data_path"])
        for mid, cfg in _config_cache.items()
    }
