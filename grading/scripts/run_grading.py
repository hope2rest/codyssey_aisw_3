"""단일 문항 채점 CLI

사용법:
    python -m grading.scripts.run_grading \
        --student-id S001 \
        --mission-id aiml_level1_q1_svd \
        --submission-dir ./q1_svd
"""
import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading.core.grader import Grader
from grading.utils.config_loader import load_config, get_all_mission_ids


def main():
    parser = argparse.ArgumentParser(description="AI/ML 단일 문항 채점")
    parser.add_argument(
        "--student-id", default="anonymous", help="학생 ID"
    )
    parser.add_argument(
        "--mission-id",
        required=True,
        choices=get_all_mission_ids(),
        help="미션 ID",
    )
    parser.add_argument(
        "--submission-dir",
        required=True,
        help="제출물 디렉토리 경로",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="결과 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="결과 Markdown 파일 경로 (선택)",
    )
    args = parser.parse_args()

    # 설정 로드
    config = load_config(args.mission_id)

    # 채점 실행
    grader = Grader()
    result = grader.run_from_config(
        config=config,
        submission_dir=args.submission_dir,
    )
    result.student_id = args.student_id

    # Markdown 출력
    print(result.to_markdown())

    # JSON 저장
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        print(f"\nJSON 결과 저장: {output_path}")

    # Markdown 저장
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_markdown())
        print(f"Markdown 결과 저장: {output_path}")

    # 합격 여부로 exit code 결정
    sys.exit(0 if result.is_passed else 1)


if __name__ == "__main__":
    main()
