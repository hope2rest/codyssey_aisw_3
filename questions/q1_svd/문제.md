## 문제 1: SVD 기반 데이터 차원 축소 및 복원 정확도 분석

### [ 시험 정보 ]
| 항목 | 내용 |
|------|------|
| 과정 | AI 올인원 |
| 단계 | AI/SW 심화 |
| 난이도 | 1 |
| 권장 시간 | 15분 |
| 관련 과목 | AI 수학, 데이터 분석, 머신러닝 |
| Pass 기준 | 정답 체크리스트 6개 중 6개 모두 충족 |

---

### [ 문제 ]

당신은 센서 데이터 분석팀의 엔지니어입니다.
공장의 100개 센서에서 수집된 500건의 측정 데이터(`sensor_data.csv`, 500×100 행렬)가 주어집니다.

이 데이터의 차원을 줄여 **핵심 패턴만 보존**하면서도 **노이즈를 제거**하는 것이 목표입니다.
다음 요구사항을 모두 충족하는 Python 스크립트 `q1_solution.py`를 작성하십시오.

---

### [ 요구사항 ]

1. **데이터 로드 및 전처리**
   - `sensor_data.csv`를 로드하세요 (헤더 없음).
   - 각 열(feature)에 대해 **평균 0, 표준편차 1로 표준화**(Standardization)하세요.
   - 표준화에는 **NumPy만 사용**하세요 (`sklearn` 등 외부 라이브러리 사용 금지).
   - 표준편차 계산 시 `ddof=0`을 사용하세요.

2. **SVD 분해**
   - 표준화된 데이터 행렬 X에 대해 `numpy.linalg.svd`를 사용하여 U, S, Vt로 분해하세요.
   - 반드시 `full_matrices=False` 옵션을 사용하세요.

3. **Explained Variance Ratio 계산**
   - 각 특이값에 대한 Explained Variance Ratio를 다음 수식으로 계산하세요:
     ```
     explained_variance_ratio[i] = S[i]² / sum(S²)
     ```

4. **최적 k 결정**
   - Cumulative Explained Variance Ratio가 **처음으로 95% 이상**이 되는 최소 k값을 구하세요.

5. **차원 축소 및 복원**
   - 결정된 k값으로 데이터를 축소하고 다시 복원하세요:
     ```
     X_reduced = U[:, :k] * S[:k]
     X_reconstructed = X_reduced @ Vt[:k, :]
     ```

6. **복원 오차 계산**
   - 원본 표준화 데이터와 복원 데이터 간의 MSE를 계산하세요:
     ```
     MSE = mean((X - X_reconstructed)²)
     ```

---

### [ 제약 사항 ]
- **NumPy만 사용**하여 모든 계산을 수행할 것 (`pandas`는 데이터 로드에만 허용, `sklearn`/`scipy` 금지)
- `numpy.linalg.svd` 호출 시 **`full_matrices=False`** 옵션을 반드시 사용할 것
- 모든 부동소수점 결과는 **소수점 이하 6자리**로 반올림하여 출력할 것

---

### [ 입력 형식 ]

**sensor_data.csv**
- 헤더 없는 500×100 숫자 행렬 (쉼표 구분)
- 각 행 = 1건의 측정, 각 열 = 1개 센서

---

### [ 출력 형식 ]

`result_q1.json` 파일로 다음 구조를 저장하세요:

```json
{
  "optimal_k": 정수,
  "cumulative_variance_at_k": 실수 (소수점 6자리),
  "reconstruction_mse": 실수 (소수점 6자리),
  "top_5_singular_values": [실수 5개],
  "explained_variance_ratio_top5": [실수 5개]
}
```

---

### [ 제출 방식 ]

아래 **두 파일**을 `submissions/` 폴더에 제출하세요.

| 파일 | 파일명 형식 | 예시 |
|------|------------|------|
| 솔루션 코드 | `solution_{단계}_q1_{학번}.py` | `solution_advanced_q1_20210001.py` |
| 결과 JSON | `result_{단계}_q1_{학번}.json` | `result_advanced_q1_20210001.json` |

- **단계**: `advanced` (심화) / `basic` (기초) / `intro` (입학연수)
- **학번**: 본인 학번
- 두 파일 모두 제출해야 채점이 진행됩니다.
