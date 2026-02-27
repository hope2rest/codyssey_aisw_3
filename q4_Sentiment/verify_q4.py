"""문항 4 검증 — 원본 JSON 키"""
import numpy as np, pandas as pd, json, sys, re, warnings
warnings.filterwarnings("ignore")

def verify():
    score = 0
    try:
        with open("result_q4.json","r",encoding="utf-8") as f: ans=json.load(f)
    except: print("FAIL: result_q4.json 없음"); return False

    # 기준 생성
    df = pd.read_csv("reviews.csv")
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    X_tr,X_te,y_tr,y_te = train_test_split(df["text"],df["label"],test_size=0.3,random_state=42)
    tr_df = pd.DataFrame({"text":X_tr.values,"label":y_tr.values})
    pos,neg = tr_df[tr_df["label"]==1],tr_df[tr_df["label"]==0]
    neg_o = neg.sample(n=len(pos),replace=True,random_state=42)
    bal = pd.concat([pos,neg_o]).reset_index(drop=True)
    vec = TfidfVectorizer(sublinear_tf=False,smooth_idf=True)
    X_tr_tf = vec.fit_transform(bal["text"]); X_te_tf = vec.transform(X_te)
    m = LogisticRegression(C=1.0,penalty="l2",random_state=42,max_iter=1000)
    m.fit(X_tr_tf,bal["label"])
    ep = m.predict(X_te_tf)
    exp_acc = round(float(accuracy_score(y_te,ep)),4)
    exp_f1 = round(float(f1_score(y_te,ep,average="macro")),4)

    print("="*55+"\n            문항 4 채점 결과\n"+"="*55)

    # 1. rule_based 구조
    rb = ans.get("rule_based",{})
    req = ["accuracy","precision_pos","recall_pos","precision_neg","recall_neg","f1_macro"]
    if all(k in rb for k in req):
        score+=10; print("  [PASS] 1. rule_based 6개 지표")
    else: print("  [FAIL] 1. rule_based 지표 누락")

    # 2. ml_based accuracy
    ml = ans.get("ml_based",{})
    ml_acc = ml.get("accuracy",0)
    if abs(ml_acc - exp_acc) < 0.05:
        score+=15; print(f"  [PASS] 2. ML accuracy: {ml_acc}")
    else: print(f"  [FAIL] 2. ML accuracy: {ml_acc} (기대:{exp_acc})")

    # 3. ml_based f1
    ml_f1 = ml.get("f1_macro",0)
    if abs(ml_f1 - exp_f1) < 0.05:
        score+=15; print(f"  [PASS] 3. ML F1: {ml_f1}")
    else: print(f"  [FAIL] 3. ML F1: {ml_f1} (기대:{exp_f1})")

    # 4. SHAP 부호
    sp = ans.get("shap_top5_positive",[])
    sn = ans.get("shap_top5_negative",[])
    if (len(sp)==5 and len(sn)==5 and
        all(x.get("shap_value",0)>0 for x in sp) and
        all(x.get("shap_value",0)<0 for x in sn)):
        score+=15; print("  [PASS] 4. SHAP 부호 정확")
    else: print("  [FAIL] 4. SHAP 부호/개수 오류")

    # 5. fit/transform 분리
    try:
        with open("q4_solution.py","r",encoding="utf-8") as f: code=f.read()
        if "fit_transform" in code and ".transform(" in code:
            score+=15; print("  [PASS] 5. 데이터 누수 방지")
        else: print("  [FAIL] 5. fit/transform 패턴 미확인")
    except: print("  [FAIL] 5. 코드 파싱 오류")

    # 6. business_summary (긍정/부정 한국어)
    bs = ans.get("business_summary","")
    has_pos = "긍정" in bs
    has_neg = "부정" in bs
    if has_pos and has_neg and len(bs)>=20:
        score+=15; print(f"  [PASS] 6. 비즈니스 요약: '긍정'/'부정' 포함")
    else:
        if not has_pos or not has_neg:
            print(f"  [FAIL] 6. '긍정'({has_pos})/'부정'({has_neg}) 필수 단어 누락")
        else: print(f"  [FAIL] 6. 요약 길이 부족")

    # 7. ml_based 6개 지표
    if all(k in ml for k in req):
        score+=15; print("  [PASS] 7. ml_based 6개 지표")
    else: print("  [FAIL] 7. ml_based 지표 누락")

    passed = score >= 95
    print("="*55+f"\n  점수: {score}/100\n  결과: {'✅ PASS' if passed else '❌ FAIL'}\n"+"="*55)
    return passed

if __name__=="__main__": sys.exit(0 if verify() else 1)
