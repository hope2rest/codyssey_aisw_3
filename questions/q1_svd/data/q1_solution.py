"""문제 1 정답"""
import numpy as np, pandas as pd, json

# 데이터 로드 (헤더 없음, 빈 셀 존재)
df = pd.read_csv("sensor_data.csv", header=None)
data = df.values.astype(float)

# 결측치 → 열 평균 대체
for j in range(data.shape[1]):
    col = data[:, j]; mask = np.isnan(col)
    if mask.any(): data[mask, j] = np.nanmean(col)

# 상수 열(std≈0) 제거
stds = np.std(data, axis=0, ddof=0)
data = data[:, stds > 1e-10]

# 표준화
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=0)
X = (data - mean) / std

# SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
evr = (S**2) / np.sum(S**2)
cum = np.cumsum(evr)
k = int(np.argmax(cum >= 0.95) + 1)

# 복원
X_rec = (U[:, :k] * S[:k]) @ Vt[:k, :]
mse = float(np.mean((X - X_rec)**2))

result = {
    "optimal_k": k,
    "cumulative_variance_at_k": round(float(cum[k-1]), 6),
    "reconstruction_mse": round(float(mse), 6),
    "top_5_singular_values": [round(float(s), 6) for s in S[:5]],
    "explained_variance_ratio_top5": [round(float(r), 6) for r in evr[:5]]
}
with open("result_q1.json", "w") as f: json.dump(result, f, indent=2)
print(f"Shape:{X.shape}, k={k}, MSE={mse:.6f}")
print(json.dumps(result, indent=2))