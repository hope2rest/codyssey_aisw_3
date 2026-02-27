"""전체 5문항 일괄 채점 CLI

사용법:
    python -m grading.scripts.run_all \
        --student-id S001 \
        --submission-dir .
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
from grading.utils.config_loader import load_config, ALL_MISSION_IDS

# 미션 ID → 제출물 하위 디렉토리 매핑
SUBMISSION_DIRS = {
    "aiml_level1_q1_svd": "q1_svd",
    "aiml_level1_q2_tfidf": "q2_tfidf",
    "aiml_level2_q3_cv": "q3_CV",
    "aiml_level2_q4_sentiment": "q4_Sentiment",
    "aiml_level3_q5_detection": "q5_detection",
}


def main():
    parser = argparse.ArgumentParser(description="AI/ML 전체 5문항 일괄 채점")
    parser.add_argument(
        "--student-id", default="anonymous", help="학생 ID"
    )
    parser.add_argument(
        "--submission-dir",
        default=".",
        help="제출물 루트 디렉토리 (기본: 현재 디렉토리)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="전체 결과 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="전체 결과 Markdown 파일 경로 (선택)",
    )
    args = parser.parse_args()

    base_dir = Path(args.submission_dir).resolve()
    grader = Grader()

    print("=" * 60)
    print("        AI/ML 퀴즈 자동 채점 시스템 - 전체 채점")
    print("=" * 60)
    print(f"  학생 ID: {args.student_id}")
    print(f"  제출 디렉토리: {base_dir}")
    print("=" * 60)
    print()

    for mission_id in ALL_MISSION_IDS:
        sub_dir_name = SUBMISSION_DIRS[mission_id]
        sub_dir = base_dir / sub_dir_name

        if not sub_dir.exists():
            print(f"[SKIP] {mission_id}: 디렉토리 없음 ({sub_dir})")
            print()
            continue

        try:
            config = load_config(mission_id)
            result = grader.run_from_config(config=config, submission_dir=str(sub_dir))
            result.student_id = args.student_id
            print(result.to_markdown())
            print()
        except Exception as e:
            print(f"[ERROR] {mission_id}: {e}")
            print()

    # 전체 요약
    summary = grader.summary_markdown(student_id=args.student_id)
    print(summary)

    # JSON 저장
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = {
            "student_id": args.student_id,
            "total_earned": grader.total_earned,
            "total_possible": grader.total_possible,
            "missions": [r.to_dict() for r in grader.results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, ensure_ascii=False, indent=2, fp=f)
        print(f"\nJSON 결과 저장: {output_path}")

    # Markdown 저장
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_md = []
        for r in grader.results:
            full_md.append(r.to_markdown())
            full_md.append("")
        full_md.append(summary)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_md))
        print(f"Markdown 결과 저장: {output_path}")

    # 전체 합격 여부
    all_passed = all(r.is_passed for r in grader.results)
    total = grader.total_earned
    possible = grader.total_possible
    print(f"\n최종: {total}/{possible}점 - {'ALL PASS' if all_passed else 'FAIL'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
