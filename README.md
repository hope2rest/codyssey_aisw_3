# AI/ML 시험 자동 채점 시스템

플러그인 기반 채점 프레임워크를 사용하여 AI/ML 5개 문항(총 500점)을 자동으로 채점합니다.
학생이 제출한 결과물을 `submissions/` 폴더에 모아두면, 파일명에서 단계/문항/학번을 자동 파싱하고
채점 후 `results/` 폴더에 검증 결과 리포트를 생성합니다.

---

## 채점 프로세스 한눈에 보기

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 학생이 제출한 결과물을 submissions/ 폴더에 넣는다                  │
│                                                                 │
│     submissions/                                                │
│     ├── result_advanced_q1_260227001.json    ← 결과 파일         │
│     ├── solution_advanced_q1_260227001.py    ← 솔루션 코드        │
│     ├── result_advanced_q2_260227001.json                       │
│     ├── solution_advanced_q2_260227001.py                       │
│     └── ...                                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 채점 명령어 실행                                               │
│                                                                 │
│     grade                        ← 전체 채점                     │
│     grade q3                     ← Q3만 채점                     │
│     grade 260227001              ← 특정 학생만 채점               │
│     grade advanced_q3_260227001  ← 특정 학생의 특정 문항           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 시스템이 자동으로 처리                                          │
│                                                                 │
│     파일명 파싱 → (단계, 문항, 학번) 추출                            │
│     → 원본 데이터(q1_svd/ 등)와 대조하여 채점                        │
│     → 합격/불합격 판정                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. results/ 폴더에 결과 생성                                      │
│                                                                 │
│     results/                                                    │
│     ├── result_advanced_q1_260227001_verified.json  ← 기계 판독  │
│     ├── result_advanced_q1_260227001_verified.md    ← 사람 판독  │
│     ├── result_advanced_q2_260227001_failed.json                │
│     ├── result_advanced_q2_260227001_failed.md                  │
│     └── ...                                                     │
│                                                                 │
│     파일명 규칙: result_{단계}_q{번호}_{학번}_{verified|failed}     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 문항 구성

| 문항 | 주제 | 난이도 | 배점 | Pass 기준 |
|------|------|--------|------|-----------|
| Q1 | SVD 기반 차원축소 복원분석 | Level 1 | 100 | 100% |
| Q2 | TF-IDF 코사인유사도 문서검색 | Level 1 | 100 | 95% |
| Q3 | 이미지 객체카운팅 규칙기반 한계분석 | Level 2 | 100 | 85% |
| Q4 | 고객리뷰 감성분석 | Level 2 | 100 | 95% |
| Q5 | 부품결함검출 딥러닝기초 전이학습 | Level 3 | 100 | 95% |

각 문항의 상세 요구사항은 해당 폴더의 문제.md 파일을 참고하세요.

---

## 시작하기

```bash
git clone https://github.com/hope2rest/codyssey_aisw_3.git
cd codyssey_aisw_3
pip install -r requirements.txt
```

클론하면 `submissions/` 폴더가 빈 상태로 포함되어 있습니다.
제출물을 넣고 `grade` 명령어를 실행하면 `results/` 폴더가 자동 생성됩니다.

---

## 제출 파일명 규칙

학생은 **솔루션 코드**와 **결과 JSON** 두 파일을 아래 형식으로 제출합니다.

```
솔루션:  solution_{단계}_q{번호}_{학번}.py
결과:    result_{단계}_q{번호}_{학번}.json
```

| 항목 | 값 | 설명 |
|------|----|------|
| 단계 | `advanced` | 심화 과정 |
|      | `basic` | 기초 과정 |
|      | `intro` | 입학연수 |
| 번호 | `1` ~ `5` | 문항 번호 |
| 학번 | 예: `20210001` | 본인 학번 |

### 문항별 제출 파일 예시

| 문항 | 솔루션 파일 | 결과 파일 |
|------|------------|----------|
| Q1 | `solution_advanced_q1_20210001.py` | `result_advanced_q1_20210001.json` |
| Q2 | `solution_advanced_q2_20210001.py` | `result_advanced_q2_20210001.json` |
| Q3 | `solution_advanced_q3_20210001.py` | `result_advanced_q3_20210001.json` |
| Q4 | `solution_advanced_q4_20210001.py` | `result_advanced_q4_20210001.json` |
| Q5 | `solution_advanced_q5_20210001.py` | `result_advanced_q5_20210001.json` |

> Q2는 코드 분석 항목이 없어 솔루션 파일 없이도 채점 가능하지만, 일관성을 위해 함께 제출을 권장합니다.

---

## 사용법

### 빠른 채점 (`grade` 명령어)

프로젝트 루트(`codyssey_aisw/`)에서 `grade.bat`을 사용합니다.

```bash
grade                          # 전체 제출물 채점
grade q3                       # Q3 문항만 채점
grade 260227001                # 특정 학생만 채점
grade advanced_q3_260227001    # 특정 학생의 특정 문항만 채점
```

> `grade` 뒤의 인자는 부분 문자열 매칭입니다. `q3`을 입력하면 파일명에 `q3`이 포함된 제출물만 채점됩니다.

실행 결과 예시:
```
============================================================
     제출물 기반 일괄 채점 시스템
============================================================

[INFO] 5개 제출물 발견

[PASS] advanced_q1_260227001: 100/100 -> result_advanced_q1_260227001_verified.json
[FAIL] advanced_q2_260227001: 70/100 -> result_advanced_q2_260227001_failed.json
[SKIP] advanced_q3_260227001: solution 파일 누락

============================================================
                    채점 요약
============================================================
  전체: 5건
  합격(PASS): 1건
  불합격(FAIL): 1건
  건너뜀(SKIP): 1건
============================================================
```

### 전체 명령어 (grade.bat 없이 직접 실행할 경우)

```bash
# 전체 채점
python -m grading.scripts.run_submissions

# 필터 지정
python -m grading.scripts.run_submissions --filter q3
```

---

## 결과 확인 방법

채점이 완료되면 `results/` 폴더에 학생별/문항별 파일이 생성됩니다.

### result.json (기계 판독용)

```json
{
  "mission_id": "aiml_level1_q1_svd",
  "mission_name": "Q1. SVD 기반 차원축소 복원분석",
  "student_id": "260227001",
  "total_points": 100,
  "earned_points": 100,
  "score_ratio": 1.0,
  "is_passed": true,
  "checks": [
    {"id": "1", "description": "optimal_k 일치", "status": "passed", "earned_points": 20},
    {"id": "2", "description": "cumulative_variance_at_k 정확도", "status": "passed", "earned_points": 20},
    ...
  ]
}
```

### report.md (사람 판독용)

```
# Q1. SVD 기반 차원축소 복원분석 채점 결과

학생 ID: 260227001

=======================================================

  [PASS] 1. optimal_k 일치 (20/20점)
  [PASS] 2. cumulative_variance_at_k 정확도 (20/20점)
  [PASS] 3. reconstruction_mse 정확도 (20/20점)
  [PASS] 4. top_5_singular_values 정확도 (20/20점)
  [PASS] 5. explained_variance_ratio_top5 정확도 (20/20점)

=======================================================
  점수: 100/100
  합격 기준: 100%
  결과: PASS
=======================================================
```

---

## 프로젝트 구조

```
codyssey_aisw/
├── grading/
│   ├── core/                          # 핵심 프레임워크
│   │   ├── base_validator.py          # Template Method 추상 클래스 + _resolve_file()
│   │   ├── check_item.py             # CheckItem + CheckStatus (검증 단위)
│   │   ├── checklist.py              # 점수 집계
│   │   ├── grader.py                 # 동적 Validator 로더
│   │   └── validation_result.py      # JSON/Markdown 리포트
│   ├── plugins/aiml/validators/      # 문항별 Validator 플러그인
│   │   ├── q1_svd_validator.py
│   │   ├── q2_tfidf_validator.py
│   │   ├── q3_cv_validator.py
│   │   ├── q4_sentiment_validator.py
│   │   └── q5_detection_validator.py
│   ├── missions/aiml/                # 미션 설정 (YAML)
│   │   ├── level1/ (q1_svd, q2_tfidf)
│   │   ├── level2/ (q3_cv, q4_sentiment)
│   │   └── level3/ (q5_detection)
│   ├── utils/config_loader.py        # YAML 설정 로더
│   └── scripts/                      # CLI 스크립트
│       ├── run_submissions.py        # ★ 제출물 일괄 채점 (submissions/ → results/)
│       ├── run_grading.py            # 단일 문항 채점
│       └── run_all.py               # 전체 5문항 채점
├── q1_svd/                           # Q1 원본 데이터 + 문제.md
├── q2_tfidf/                         # Q2 원본 데이터 + 문제.md
├── q3_CV/                            # Q3 원본 데이터 + 문제.md
├── q4_Sentiment/                     # Q4 원본 데이터 + 문제.md
├── q5_detection/                     # Q5 원본 데이터 + 문제.md
├── submissions/                      # ★ 학생 제출물 (여기에 파일을 넣으세요)
├── results/                          # ★ 채점 결과 (자동 생성)
├── requirements.txt
└── README.md
```

---

## 아키텍처

### Template Method 패턴

각 Validator는 `BaseValidator`를 상속하며 다음 순서로 실행됩니다:

1. `setup()` — 데이터 로드, 참조 답안 생성, 학생 결과 JSON 로드
2. `build_checklist()` — 검증 항목(CheckItem)을 체크리스트에 등록
3. `validate()` — 모든 CheckItem 실행 후 ValidationResult 반환
4. `teardown()` — 정리 작업

### 파일 경로 결정 (`_resolve_file`)

`BaseValidator._resolve_file()` 헬퍼가 파일 경로를 결정합니다:

- `config`에 override 키가 있으면 해당 경로 사용 (submissions/ 기반 채점)
- 없으면 기존 `submission_dir/기본파일명` 사용 (하위 호환)

이를 통해 기존 `run_grading.py`, `run_all.py`와 새 `run_submissions.py` 모두 동일한 Validator를 사용합니다.

### CheckItem 부분 점수

- `validator()`가 `bool` 반환 → pass(만점) / fail(0점)
- `validator()`가 `int` 반환 → 부분 점수 (0 ~ max points)
- `validator()`가 `(value, message)` 튜플 반환 → 점수 + 메시지

### 동적 플러그인 로딩

`config.yaml`의 `validators` 항목에서 모듈 경로와 클래스명을 지정하면,
`Grader`가 `importlib`로 동적 로드하여 실행합니다.

---

## AI 함정(Trap) 목록

채점 시 AI가 흔히 실수하는 항목들입니다:

| 문항 | 함정 | 설명 |
|------|------|------|
| Q1 | ddof=0 | 모집단 표준편차 사용 필수 (ddof=1이 아님) |
| Q2 | NFC 순서 | NFC 정규화 → lowercase → 특수문자 제거 순서 |
| Q2 | 빈 쿼리 | 불용어만으로 구성된 쿼리 → 유사도 0.0 |
| Q3 | MAE=0 | 과적합 의심으로 0점 처리 |
| Q3 | test_01 | 이미지 없는 항목 제외 필수 |
| Q4 | 데이터 누수 | fit_transform(train) + transform(test) 분리 |
| Q5 | 파일명 | q6_solution.py / result_q6.json 사용 (내부 매핑) |
| Q5 | keras/torch | sklearn만 사용 가능 |

---

## 새 문항 추가 방법

1. `grading/plugins/aiml/validators/`에 새 Validator 클래스 작성 (`BaseValidator` 상속)
2. `grading/missions/aiml/`에 `config.yaml` 추가
3. `grading/utils/config_loader.py`의 `_MISSION_MAP`에 매핑 추가
4. `grading/scripts/run_all.py`의 `SUBMISSION_DIRS`에 디렉토리 매핑 추가
5. `grading/scripts/run_submissions.py`의 `QUESTION_MAP`에 매핑 추가
