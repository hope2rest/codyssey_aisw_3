"""YAML 설정 로더"""
from pathlib import Path
from typing import Dict, Any

import yaml


# 미션 ID → 설정 파일 경로 매핑
_MISSIONS_DIR = Path(__file__).resolve().parent.parent / "missions"

_MISSION_MAP = {
    "aiml_level1_q1_svd": "aiml/level1/q1_svd/config.yaml",
    "aiml_level1_q2_tfidf": "aiml/level1/q2_tfidf/config.yaml",
    "aiml_level2_q3_cv": "aiml/level2/q3_cv/config.yaml",
    "aiml_level2_q4_sentiment": "aiml/level2/q4_sentiment/config.yaml",
    "aiml_level3_q5_detection": "aiml/level3/q5_detection/config.yaml",
}

ALL_MISSION_IDS = list(_MISSION_MAP.keys())


def get_config_path(mission_id: str) -> Path:
    """미션 ID로 config.yaml 경로 반환"""
    rel = _MISSION_MAP.get(mission_id)
    if rel is None:
        raise ValueError(
            f"알 수 없는 미션 ID: {mission_id}\n"
            f"사용 가능: {', '.join(_MISSION_MAP.keys())}"
        )
    return _MISSIONS_DIR / rel


def load_config(mission_id: str) -> Dict[str, Any]:
    """미션 ID로 YAML 설정 로드"""
    path = get_config_path(mission_id)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
