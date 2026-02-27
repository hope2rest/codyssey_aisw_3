"""문제 1 검증"""
import numpy as np, pandas as pd, json, sys

def gen_ref():
    df = pd.read_csv("sensor_data.csv", header=None)
    d = df.values.astype(float)
    for j in range(d.shape[1]):
        col=d[:,j]; m=np.isnan(col)
        if m.any(): d[m,j]=np.nanmean(col)
    s=np.std(d,axis=0,ddof=0); d=d[:,s>1e-10]
    m=np.mean(d,axis=0); s=np.std(d,axis=0,ddof=0); X=(d-m)/s
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    evr=(S**2)/np.sum(S**2); cum=np.cumsum(evr)
    k=int(np.argmax(cum>=0.95)+1)
    mse=float(np.mean((X-(U[:,:k]*S[:k])@Vt[:k,:])**2))
    return {"optimal_k":k,"cumulative_variance_at_k":round(float(cum[k-1]),6),
            "reconstruction_mse":round(float(mse),6),
            "top_5_singular_values":[round(float(s),6) for s in S[:5]],
            "explained_variance_ratio_top5":[round(float(r),6) for r in evr[:5]]}

def verify():
    score=0; ref=gen_ref()
    try:
        with open("result_q1.json") as f: ans=json.load(f)
    except: print("FAIL: result_q1.json 없음"); return False
    print("="*55+"\n            문항 1 채점 결과\n"+"="*55)
    checks = [
        ("1. optimal_k", "optimal_k", 20, lambda a,r: a==r),
        ("2. cumulative_variance", "cumulative_variance_at_k", 20, lambda a,r: abs(a-r)<1e-4),
        ("3. reconstruction_mse", "reconstruction_mse", 20, lambda a,r: abs(a-r)<1e-4),
    ]
    for name, key, pts, fn in checks:
        a=ans.get(key); r=ref[key]
        if a is not None and fn(a,r):
            score+=pts; print(f"  [PASS] {name}: {a}")
        else: print(f"  [FAIL] {name}: {a} (기대: {r})")
    # SV
    sv=ans.get("top_5_singular_values",[]); rsv=ref["top_5_singular_values"]
    if len(sv)==5 and all(abs(a-b)<1e-3 for a,b in zip(sv,rsv)):
        score+=20; print(f"  [PASS] 4. top_5_singular_values")
    else: print(f"  [FAIL] 4. top_5_singular_values")
    # EVR
    se=ans.get("explained_variance_ratio_top5",[]); re_=ref["explained_variance_ratio_top5"]
    if len(se)==5 and all(abs(a-b)<1e-4 for a,b in zip(se,re_)):
        score+=20; print(f"  [PASS] 5. explained_variance_ratio_top5")
    else: print(f"  [FAIL] 5. explained_variance_ratio_top5")
    passed=score==100
    print("="*55+f"\n  점수: {score}/100\n  결과: {'✅ PASS' if passed else '❌ FAIL'}\n"+"="*55)
    return passed
if __name__=="__main__": sys.exit(0 if verify() else 1)