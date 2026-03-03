import numpy as np
import pandas as pd
import json

# 1. Load data: no header, 500x100
df = pd.read_csv("sensor_data.csv", header=None)
X_raw = df.values.astype(float)

# Handle NaN values: replace with column mean (nanmean)
col_means = np.nanmean(X_raw, axis=0)
nan_mask = np.isnan(X_raw)
X_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

# 2. Standardize each column using NumPy only, ddof=0
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0, ddof=0)

# Handle constant columns (std=0): replace 0 with 1 to avoid division by zero
# These columns become all-zero after standardization
std_safe = np.where(std == 0, 1.0, std)
X = (X_raw - mean) / std_safe

# 3. SVD decomposition with full_matrices=False
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# 4. Explained Variance Ratio: evr[i] = S[i]^2 / sum(S^2)
evr = S**2 / np.sum(S**2)

# 5. Find optimal k: first k where cumulative variance >= 95%
cum = np.cumsum(evr)
optimal_k = int(np.argmax(cum >= 0.95) + 1)
cumulative_variance_at_k = round(float(cum[optimal_k - 1]), 6)

# 6. Reconstruct using optimal k
X_reduced = U[:, :optimal_k] * S[:optimal_k]
X_reconstructed = X_reduced @ Vt[:optimal_k, :]

# 7. Calculate MSE between original standardized and reconstructed
mse = round(float(np.mean((X - X_reconstructed)**2)), 6)

# 8. Prepare result
result = {
    "optimal_k": optimal_k,
    "cumulative_variance_at_k": cumulative_variance_at_k,
    "reconstruction_mse": mse,
    "top_5_singular_values": [round(float(s), 6) for s in S[:5]],
    "explained_variance_ratio_top5": [round(float(r), 6) for r in evr[:5]]
}

# 9. Save result to JSON
with open("result_q1.json", "w") as f:
    json.dump(result, f, indent=2)

print("Result saved to result_q1.json")
print(json.dumps(result, indent=2))