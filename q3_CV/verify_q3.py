"""문제 3 검증"""
import numpy as np, json, sys, ast, os, re

def verify():
    score=0
    with open("labels.json") as f: labels=json.load(f)
    valid=[n for n in labels if os.path.exists(os.path.join("images",f"{n}.png"))]
    try:
        with open("result_q3.json","r",encoding="utf-8") as f: ans=json.load(f)
    except: print("FAIL: result_q3.json 없음"); return False
    print("="*55+"\n            문항 3 채점 결과\n"+"="*55)
    # 1. conv2d
    try:
        with open("q3_solution.py","r",encoding="utf-8") as f: code=f.read()
        tree=ast.parse(code); funcs=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
        if "conv2d" in funcs and "filter2D" not in code:
            score+=15; print("  [PASS] 1. conv2d 직접 구현")
        else: print("  [FAIL] 1. conv2d 미구현")
    except: print("  [FAIL] 1. 코드 파싱 오류")
    # 2. valid images only
    preds=ans.get("predictions",{})
    if set(preds.keys())==set(valid):
        score+=15; print(f"  [PASS] 2. {len(preds)}개 이미지만 분석 (test_01 제외)")
    else:
        extra=set(preds.keys())-set(valid); missing=set(valid)-set(preds.keys())
        if extra: print(f"  [FAIL] 2. 존재하지 않는 이미지 포함: {extra}")
        if missing: print(f"  [FAIL] 2. 누락: {missing}")
    # 3. easy MAE (비차단)
    easy={k:v for k,v in labels.items() if k.startswith("easy") and k in valid}
    easy_errs=[abs(preds.get(k,0)-v) for k,v in easy.items()]
    mae=float(np.mean(easy_errs)) if easy_errs else 999
    if mae==0.0:
        print(f"  [FAIL] 3. 과적합 또는 데이터 누수 의심 (MAE: {mae:.2f})")
    elif mae<=3.0:
        score+=15; print(f"  [PASS] 3. easy MAE: {mae:.2f}")
    elif mae<=10.0:
        score+=10; print(f"  [WARN] 3. easy MAE: {mae:.2f} (10점, 부분 점수)")
    elif mae<=30.0:
        score+=5; print(f"  [WARN] 3. easy MAE: {mae:.2f} (5점, 부분 점수)")
    elif mae==999:
        print(f" [FAIL] 3. 예측 없음")
    else: print(f" [FAIL] 3. 오차 기준치 초과 (mae: {mae:.2f})")
    # 4. metrics
    m=ans.get("metrics",{})
    if all(c in m for c in ["easy","medium","hard"]):
        score+=10; print("  [PASS] 4. metrics 3카테고리")
    else: print("  [FAIL] 4. metrics 불완전")
    # 5. failure_reasons (한국어)
    fr=ans.get("failure_reasons",[])
    kr=re.compile(r'[가-힣]')
    if len(fr)>=3 and all(len(r)>=20 and kr.search(r) for r in fr):
        score+=20; print(f"  [PASS] 5. failure_reasons: {len(fr)}개 한국어")
    else: print(f"  [FAIL] 5. failure_reasons")
    # 6. why_learning_based
    wlb=ans.get("why_learning_based","")
    if 30<=len(wlb)<=200 and kr.search(wlb):
        score+=15; print(f"  [PASS] 6. why_learning_based: {len(wlb)}자")
    else: print(f"  [FAIL] 6. why_learning_based")
    # 7. worst_case
    if ans.get("worst_case_image","").startswith("hard"):
        score+=10; print(f"  [PASS] 7. worst_case: {ans['worst_case_image']}")
    else: print(f"  [FAIL] 7. worst_case")
    passed=score>=85
    print("="*55+f"\n  점수: {score}/100\n  결과: {'✅ PASS' if passed else '❌ FAIL'}\n"+"="*55)
    return passed
if __name__=="__main__": sys.exit(0 if verify() else 1)
