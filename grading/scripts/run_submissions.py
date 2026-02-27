"""제출물 기반 일괄 채점 CLI

submissions/ 폴더에 학생 제출물을 모아두면, 파일명에서
단계/문항/학번을 자동 파싱하고 채점 후 results/ 폴더에 출력한다.

파일명 규칙:
    result_{stage}_q{num}_{student_id}.json
    solution_{stage}_q{num}_{student_id}.py

사용법:
    python -m grading.scripts.run_submissions \
        --data-root . \
        --submissions-dir ./submissions \
        --output-dir ./results
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grading.core.grader import Grader
from grading.utils.config_loader import load_config

# 파일명 정규식
RESULT_RE = re.compile(
    r'^result_(?P<stage>\w+)_q(?P<qnum>\d+)_(?P<sid>.+)\.json$'
)
SOLUTION_RE = re.compile(
    r'^solution_(?P<stage>\w+)_q(?P<qnum>\d+)_(?P<sid>.+)\.py$'
)

# (stage, qnum) → (mission_id, data_subdir)
QUESTION_MAP = {
    ("advanced", "1"): ("aiml_level1_q1_svd", "q1_svd"),
    ("advanced", "2"): ("aiml_level1_q2_tfidf", "q2_tfidf"),
    ("advanced", "3"): ("aiml_level2_q3_cv", "q3_CV"),
    ("advanced", "4"): ("aiml_level2_q4_sentiment", "q4_Sentiment"),
    ("advanced", "5"): ("aiml_level3_q5_detection", "q5_detection"),
    ("basic", "1"): ("aiml_level1_q1_svd", "q1_svd"),
    ("basic", "2"): ("aiml_level1_q2_tfidf", "q2_tfidf"),
    ("basic", "3"): ("aiml_level2_q3_cv", "q3_CV"),
    ("basic", "4"): ("aiml_level2_q4_sentiment", "q4_Sentiment"),
    ("basic", "5"): ("aiml_level3_q5_detection", "q5_detection"),
    ("intro", "1"): ("aiml_level1_q1_svd", "q1_svd"),
    ("intro", "2"): ("aiml_level1_q2_tfidf", "q2_tfidf"),
    ("intro", "3"): ("aiml_level2_q3_cv", "q3_CV"),
    ("intro", "4"): ("aiml_level2_q4_sentiment", "q4_Sentiment"),
    ("intro", "5"): ("aiml_level3_q5_detection", "q5_detection"),
}

# solution이 필요 없는 문항 (코드 분석 없음)
NO_SOLUTION_REQUIRED = {
    ("advanced", "2"), ("basic", "2"), ("intro", "2"),
}


def scan_submissions(submissions_dir: Path):
    """submissions 폴더 스캔 → {(stage, qnum, sid): {result_file, solution_file}} 그룹핑"""
    groups = defaultdict(dict)

    for f in sorted(submissions_dir.iterdir()):
        if not f.is_file():
            continue

        m = RESULT_RE.match(f.name)
        if m:
            key = (m.group("stage"), m.group("qnum"), m.group("sid"))
            groups[key]["result_file"] = f
            continue

        m = SOLUTION_RE.match(f.name)
        if m:
            key = (m.group("stage"), m.group("qnum"), m.group("sid"))
            groups[key]["solution_file"] = f
            continue

    return dict(groups)


def main():
    parser = argparse.ArgumentParser(
        description="제출물 기반 일괄 채점 시스템"
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="원본 데이터 루트 디렉토리 (q1_svd/ 등이 있는 곳)",
    )
    parser.add_argument(
        "--submissions-dir",
        default="./submissions",
        help="학생 제출물 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="채점 결과 출력 디렉토리",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    submissions_dir = Path(args.submissions_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not submissions_dir.exists():
        print(f"[ERROR] submissions 디렉토리 없음: {submissions_dir}")
        sys.exit(1)

    print("=" * 60)
    print("     제출물 기반 일괄 채점 시스템")
    print("=" * 60)
    print(f"  데이터 루트: {data_root}")
    print(f"  제출물 디렉토리: {submissions_dir}")
    print(f"  출력 디렉토리: {output_dir}")
    print("=" * 60)
    print()

    # 1. 스캔
    groups = scan_submissions(submissions_dir)
    if not groups:
        print("[INFO] 제출물이 없습니다.")
        sys.exit(0)

    print(f"[INFO] {len(groups)}개 제출물 발견\n")

    # 통계
    success_count = 0
    fail_count = 0
    skip_count = 0
    error_count = 0
    summary_rows = []

    # 2. 각 제출물 채점
    for (stage, qnum, sid), files in sorted(groups.items()):
        label = f"{stage}_q{qnum}_{sid}"
        result_file = files.get("result_file")
        solution_file = files.get("solution_file")

        # 매핑 확인
        question_key = (stage, qnum)
        if question_key not in QUESTION_MAP:
            print(f"[SKIP] {label}: 알 수 없는 문항 ({stage}, q{qnum})")
            skip_count += 1
            summary_rows.append((label, "SKIP", "알 수 없는 문항"))
            print()
            continue

        mission_id, data_subdir = QUESTION_MAP[question_key]

        # result 필수 확인
        if not result_file:
            print(f"[SKIP] {label}: result 파일 누락")
            skip_count += 1
            summary_rows.append((label, "SKIP", "result 파일 누락"))
            print()
            continue

        # solution 필수 확인 (Q2 제외)
        if question_key not in NO_SOLUTION_REQUIRED and not solution_file:
            print(f"[SKIP] {label}: solution 파일 누락")
            skip_count += 1
            summary_rows.append((label, "SKIP", "solution 파일 누락"))
            print()
            continue

        # 데이터 디렉토리 확인
        submission_dir = data_root / data_subdir
        if not submission_dir.exists():
            print(f"[SKIP] {label}: 데이터 디렉토리 없음 ({submission_dir})")
            skip_count += 1
            summary_rows.append((label, "SKIP", "데이터 디렉토리 없음"))
            print()
            continue

        # config 로드 + override 주입
        try:
            config = load_config(mission_id)
            config["result_file_override"] = str(result_file)
            if solution_file:
                config["solution_file_override"] = str(solution_file)

            # 채점 실행
            grader = Grader()
            result = grader.run_from_config(
                config=config,
                submission_dir=str(submission_dir),
            )
            result.student_id = sid

            # 결과 판정
            status = "verified" if result.is_passed else "failed"
            out_base = f"result_{stage}_q{qnum}_{sid}_{status}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # result JSON 저장
            with open(output_dir / f"{out_base}.json", "w", encoding="utf-8") as f:
                f.write(result.to_json())

            # report MD 저장
            with open(output_dir / f"{out_base}.md", "w", encoding="utf-8") as f:
                f.write(result.to_markdown())

            # 콘솔 출력
            mark = "PASS" if result.is_passed else "FAIL"
            score = f"{result.earned_points}/{result.total_points}"
            print(f"[{mark}] {label}: {score} -> {out_base}.json")

            if result.is_passed:
                success_count += 1
            else:
                fail_count += 1
            summary_rows.append((label, mark, score))

        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            error_count += 1
            summary_rows.append((label, "ERROR", str(e)))

        print()

    # 3. 요약 출력
    print("=" * 60)
    print("                    채점 요약")
    print("=" * 60)
    print(f"  전체: {len(groups)}건")
    print(f"  합격(PASS): {success_count}건")
    print(f"  불합격(FAIL): {fail_count}건")
    print(f"  건너뜀(SKIP): {skip_count}건")
    print(f"  오류(ERROR): {error_count}건")
    print("=" * 60)
    print()

    if summary_rows:
        print("| 제출물 | 결과 | 점수/사유 |")
        print("|--------|------|-----------|")
        for label, status, detail in summary_rows:
            print(f"| {label} | {status} | {detail} |")
        print()

    print(f"결과 저장 위치: {output_dir}")
    sys.exit(0 if error_count == 0 and fail_count == 0 else 1)


if __name__ == "__main__":
    main()
