# AI/ML 퀴즈 자동 채점 시스템

플러그인 기반 채점 프레임워크를 사용하여 AI/ML 5개 문항(총 500점)을 자동으로 채점합니다.

## 문항 구성

| 문항 | 주제 | 난이도 | 배점 | Pass 기준 |
|------|------|--------|------|-----------|
| Q1 | SVD 기반 차원축소 복원분석 | Level 1 | 100 | 100% |
| Q2 | TF-IDF 코사인유사도 문서검색 | Level 1 | 100 | 95% |
| Q3 | 이미지 객체카운팅 규칙기반 한계분석 | Level 2 | 100 | 85% |
| Q4 | 고객리뷰 감성분석 | Level 2 | 100 | 95% |
| Q5 | 부품결함검출 딥러닝기초 전이학습 | Level 3 | 100 | 95% |

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 단일 문항 채점

```bash
python -m grading.scripts.run_grading \
  --student-id S001 \
  --mission-id aiml_level1_q1_svd \
  --submission-dir ./q1_svd
```

### 전체 5문항 일괄 채점

```bash
python -m grading.scripts.run_all \
  --student-id S001 \
  --submission-dir .
```

### 결과 파일 저장

```bash
# JSON + Markdown 리포트 저장
python -m grading.scripts.run_all \
  --student-id S001 \
  --submission-dir . \
  --output-json results/report.json \
  --output-md results/report.md
```

### 사용 가능한 미션 ID

| 미션 ID | 문항 |
|---------|------|
| `aiml_level1_q1_svd` | Q1 SVD |
| `aiml_level1_q2_tfidf` | Q2 TF-IDF |
| `aiml_level2_q3_cv` | Q3 CV |
| `aiml_level2_q4_sentiment` | Q4 감성분석 |
| `aiml_level3_q5_detection` | Q5 결함검출 |

## 프로젝트 구조

```
codyssey_aisw/
├── grading/
│   ├── core/                      # 핵심 프레임워크
│   │   ├── check_item.py          # CheckItem + CheckStatus (검증 단위)
│   │   ├── checklist.py           # 점수 집계
│   │   ├── base_validator.py      # Template Method 추상 클래스
│   │   ├── grader.py              # 동적 Validator 로더
│   │   └── validation_result.py   # JSON/Markdown 리포트
│   ├── plugins/aiml/validators/   # 문항별 Validator 플러그인
│   │   ├── q1_svd_validator.py
│   │   ├── q2_tfidf_validator.py
│   │   ├── q3_cv_validator.py
│   │   ├── q4_sentiment_validator.py
│   │   └── q5_detection_validator.py
│   ├── missions/aiml/             # 미션 설정 (YAML)
│   │   ├── level1/ (q1_svd, q2_tfidf)
│   │   ├── level2/ (q3_cv, q4_sentiment)
│   │   └── level3/ (q5_detection)
│   ├── utils/config_loader.py     # YAML 설정 로더
│   └── scripts/                   # CLI 스크립트
│       ├── run_grading.py         # 단일 문항 채점
│       └── run_all.py             # 전체 일괄 채점
├── q1_svd/                        # Q1 제출물
├── q2_tfidf/                      # Q2 제출물
├── q3_CV/                         # Q3 제출물
├── q4_Sentiment/                  # Q4 제출물
├── q5_detection/                  # Q5 제출물
├── requirements.txt
└── README.md
```

## 아키텍처

### Template Method 패턴

각 Validator는 `BaseValidator`를 상속하며 다음 순서로 실행됩니다:

1. `setup()` - 데이터 로드, 참조 답안 생성, 학생 결과 JSON 로드
2. `build_checklist()` - 검증 항목(CheckItem)을 체크리스트에 등록
3. `validate()` - 모든 CheckItem 실행 후 ValidationResult 반환
4. `teardown()` - 정리 작업

### CheckItem 부분 점수

- `validator()`가 `bool` 반환: pass(만점) / fail(0점)
- `validator()`가 `int` 반환: 부분 점수 (0 ~ max points)
- `validator()`가 `(value, message)` 튜플 반환: 점수 + 메시지

### 동적 플러그인 로딩

`config.yaml`의 `validators` 항목에서 모듈 경로와 클래스명을 지정하면,
`Grader`가 `importlib`로 동적 로드하여 실행합니다.

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
| Q5 | 파일명 | q6_solution.py / result_q6.json 사용 |
| Q5 | keras/torch | sklearn만 사용 가능 |

## 제출물 구조

각 문항 디렉토리에 다음 파일이 필요합니다:

| 문항 | 솔루션 파일 | 결과 파일 |
|------|------------|----------|
| Q1 | `q1_solution.py` | `result_q1.json` |
| Q2 | `q2_solution.py` | `result_q2.json` |
| Q3 | `q3_solution.py` | `result_q3.json` |
| Q4 | `q4_solution.py` | `result_q4.json` |
| Q5 | `q6_solution.py` | `result_q6.json` |

## 채점 결과 예시

### Markdown 리포트

```
# Q1. SVD 기반 차원축소 복원분석 채점 결과

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

### JSON 리포트

```json
{
  "mission_id": "aiml_level1_q1_svd",
  "earned_points": 100,
  "total_points": 100,
  "is_passed": true,
  "checks": [
    {"id": "1", "status": "passed", "earned_points": 20},
    ...
  ]
}
```

## 새 문항 추가 방법

1. `grading/plugins/aiml/validators/`에 새 Validator 클래스 작성 (`BaseValidator` 상속)
2. `grading/missions/aiml/`에 `config.yaml` 추가
3. `grading/utils/config_loader.py`의 `_MISSION_MAP`에 매핑 추가
4. `grading/scripts/run_all.py`의 `SUBMISSION_DIRS`에 디렉토리 매핑 추가
