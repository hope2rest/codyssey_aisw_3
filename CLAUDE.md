# AI/ML 시험 자동 채점 시스템 — 프로젝트 컨텍스트

## 프로젝트 개요

AI/ML 5개 문항(총 500점) 자동 채점 시스템. Plug-and-Play 구조로 `questions/` 폴더에 문제 폴더만 추가하면 자동 인식.

## 디렉토리 구조

```
codyssey_aisw/
├── grading/
│   ├── core/
│   │   ├── base_validator.py    # Template Method 추상 클래스
│   │   ├── check_item.py        # CheckItem + CheckStatus
│   │   ├── checklist.py         # 점수 집계
│   │   ├── grader.py            # Validator 로더 (file/module 방식)
│   │   └── validation_result.py # JSON/MD 리포트
│   ├── plugins/aiml/validators/ # (비어 있음 — questions/로 이동 완료)
│   ├── utils/config_loader.py   # questions/*/config.yaml 자동 스캔
│   └── scripts/
│       ├── run_submissions.py   # submissions/ → results/ 일괄 채점
│       ├── run_grading.py       # 단일 문항 채점
│       └── run_all.py           # 전체 5문항 채점
├── questions/                   # ★ 문제 정의 (Plug-and-Play)
│   ├── q1_svd/                  # config.yaml, validator.py, 문제.md, data/
│   ├── q2_tfidf/
│   ├── q3_cv/
│   ├── q4_sentiment/
│   └── q5_detection/
├── submissions/                 # 학생 제출물
├── results/                     # 채점 결과 (자동 생성)
├── grade.bat                    # Windows 채점 단축 명령어
└── requirements.txt
```

## Plug-and-Play 시스템 (구현 완료)

### config_loader.py 핵심 함수
- `get_all_mission_ids()` → 동적 미션 ID 목록
- `load_config(mission_id)` → YAML 설정 로드 (캐시됨)
- `build_question_map()` → `(stage, qnum) → (mission_id, data_path)`
- `build_no_solution_required()` → solution 불필요 문항 세트
- `build_submission_dirs()` → `mission_id → data_path`

### 스캔 순서
1. `questions/*/config.yaml` 우선
2. `grading/missions/` fallback (하위 호환)

### config.yaml 형식
```yaml
mission_id: aiml_level1_q1_svd
name: "Q1. SVD 기반 차원축소 복원분석"
qnum: 1                    # 제출 파일명의 q{N}에 대응
passing_score: 1.0
data_dir: data              # 데이터 하위 폴더 (question 폴더 기준 상대경로)
solution_file: q1_solution.py
result_file: result_q1.json
solution_required: true     # false면 솔루션 없이도 채점
validators:
  - file: validator.py      # 파일 경로 방식 (새)
    class: Q1SvdValidator
  # - module: grading.plugins... # 모듈 방식 (기존, 호환)
```

### Validator 로딩
- `file:` → `importlib.util.spec_from_file_location` (새 방식)
- `module:` → `importlib.import_module` (기존 방식)

## 문항별 현황

| 문항 | mission_id | qnum | pass기준 | 솔루션필수 | 비고 |
|------|-----------|------|---------|-----------|------|
| Q1 SVD | aiml_level1_q1_svd | 1 | 100% | O | |
| Q2 TF-IDF | aiml_level1_q2_tfidf | 2 | 95% | X | |
| Q3 CV | aiml_level2_q3_cv | 3 | 85% | O | |
| Q4 감성분석 | aiml_level2_q4_sentiment | 4 | 95% | O | |
| Q5 결함검출 | aiml_level3_q5_detection | 5 | 95% | O | |

## AI 함정 분석 결과 (Sonnet 테스트 완료)

### Sonnet 채점 결과 (함정 강화 전)
| 문항 | 점수 | 결과 | 함정 발동 |
|------|------|------|----------|
| Q1 | 20/100 | FAIL | O — 상수열 제거/NaN 처리 미흡 |
| Q2 | 26/100 | FAIL | O — NFC 정규화 누락 (vocab 3개 차이) |
| Q3 | 85/100 | PASS | 부분 — conv2d filter2D 감지 (-15점) |
| Q4 | 100/100 | PASS | X — 문제지가 완전한 레시피 |
| Q5 | 100/100 | PASS | X — 문제지가 완전한 레시피 |

### Sonnet 채점 결과 (함정 강화 후)
| 문항 | 점수 | 결과 | 함정 발동 |
|------|------|------|----------|
| Q4 | 100/100 | PASS | X — Sonnet이 데이터 탐색으로 NFD/NaN 자동 감지 |
| Q5 | 100/100 | PASS | X — Sonnet이 표준 ML 베스트 프랙티스 적용 |

### 효과적인 함정 패턴 (Q1/Q2에서 검증됨)
1. **데이터에 숨긴 함정**: NaN, 상수열, NFD 인코딩 → 문제에 미언급
2. **레시피 삭제**: 수학 공식만 주고 전처리/파라미터 생략
3. **엄격한 수치 검증**: 참조값과 1e-4 오차 비교

### Q4/Q5 함정이 무효한 근본 원인
- **Q4**: 데이터셋이 단순 (1000건, 짧은 텍스트) → 어떤 파라미터든 동일한 accuracy 수렴 (C=0.1~5.0 모두 acc=0.9833)
- **Q5**: 문제 구조가 단계별 명확 → 표준 ML 파이프라인으로 자연스럽게 풀림
- **공통**: Sonnet은 코드 작성 전 반드시 데이터를 탐색하여 인코딩/NaN/패딩 이슈를 자동 감지
- **Q1/Q2와의 차이**: Q1/Q2는 수학적으로 정밀한 요구사항(ddof=0, NFC 순서)이 있어 한 단계 잘못되면 결과가 크게 달라짐

## Q4/Q5 함정 강화 (구현 완료, 효과 없음)

### Q4 구현 내역
1. **reviews.csv NFD 인코딩 삽입** — 25개 리뷰 텍스트를 NFD로 변환
2. **reviews.csv NaN label 삽입** — 8개 행에 빈 label 추가 (총 1008행)
3. **문제지 파라미터 힌트 삭제** — TfidfVectorizer/LogisticRegression 파라미터, 오버샘플링 방법 제거
4. **validator 수치 검증 강화** — ML accuracy/F1 허용 오차 0.05→0.01, rule_based accuracy 참조값 비교(±0.03)

### Q5 구현 내역
1. **inspection_log.csv NFD defect_type 삽입** — 15개 행
2. **inspection_log.csv 비패딩 part_id 삽입** — 10개 행 (예: "0069"→"69")
3. **문제지 힌트 삭제** — NFC 언급, Forward Pass 수도코드, class_weight='balanced' 제거
4. **validator 임계값 상향** — rule_based 0.65~0.85, ML acc>0.93/f1>0.85, NN acc>0.9, pretrained acc>0.93

### 추가 실험 (Q4, 모두 실패)
- `sublinear_tf=True` → 짧은 텍스트(tf=1)라 효과 없음
- `C` 값 변경(0.1~5.0) → 모두 동일한 accuracy
- `stratify=True` in train_test_split → accuracy 1.0이 되나 SHAP 검증 깨짐 (negative values → 0.0)

### 결론 (1차)
Q4/Q5의 1차 함정 강화(NFD/NaN/파라미터 삭제)는 Sonnet에 무효. 효과적인 함정을 만들려면 Q1/Q2처럼 **AI의 '개선' 본능이 오히려 틀린 결과를 만드는** 설계가 필요.

## Q4 구두점 함정 (2차 강화, 구현 완료)

### 메커니즘
1. `reviews.csv`에 감성 단어+구두점 긍정 리뷰 50개 분산 삽입 (총 1058행, dropna 후 1050행)
2. 감성 사전: `"좋습니다"` (O), `"좋습니다!"` (X) — 구두점 포함 형태 없음
3. `.split()` → `"좋습니다!"` → 사전 미매칭 → 점수 0 → 부정 예측 (오분류)
4. validator rule_based 참조 accuracy: 0.9524 (50/1050 오분류)
5. AI는 NLP 관행상 구두점 제거 → accuracy 1.0 → 차이 0.0476 > 허용 0.03 → FAIL

### 삽입 리뷰 (50개, label=1.0)
- **단일어**: `좋습니다!` ×6, `추천합니다!` ×5, `만족합니다!` ×5, `훌륭합니다!` ×4, `최고입니다!` ×3, `편리합니다!` ×3, `좋습니다.` ×2, `추천합니다.` ×2
- **중립어+구두점어**: `이 제품 좋습니다!` ×2, `품질이 훌륭합니다!` ×2, `배송 빠릅니다!` ×2, `포장 깔끔하다!` ×2, `서비스 만족합니다!` ×2
- **강조어+구두점어**: `매우 좋습니다!` ×2, `정말 훌륭합니다!` ×2, `진짜 최고입니다!` ×2, `너무 좋습니다!` ×2, `아주 만족합니다!` ×2

### 문제.md 변경
- `텍스트를 공백 기준으로 토큰화` → `텍스트를 토큰 단위로 분리` (토큰화 방법 모호화)
- `오버샘플링은 train 데이터에만 적용` → `불균형 처리는 train 데이터에서만 수행`
- `총 1,000건` → 건수 삭제

### 검증 결과
| 시나리오 | rule_based acc | validator ref | 차이 | Check 1 |
|---------|---------------|---------------|------|---------|
| 정답 (구두점 유지) | 0.9524 | 0.9524 | 0.0 | **PASS** |
| AI (구두점 제거) | 1.0 | 0.9524 | 0.0476 | **FAIL** |

- 정답 솔루션: **100/100** (PASS)
- AI 솔루션: **90/100** (check 1에서 -10점, < 95% pass 기준 → FAIL)
