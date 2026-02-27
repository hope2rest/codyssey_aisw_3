"""문제 5.1 검증(자동차 부품 결함 검출)"""
import json, sys, ast, re, os
import numpy as np
import pandas as pd
import unicodedata
from PIL import Image

def verify():
    score = 0
    try:
        with open("result_q6.json", "r", encoding="utf-8") as f:
            ans = json.load(f)
    except:
        print("FAIL: result_q6.json 없음"); return False

    # ── 정답 산출 (데이터 정제 기준) ──
    LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]

    df = pd.read_csv("inspection_log.csv", dtype={"part_id": str})
    df = df.drop_duplicates(subset="part_id", keep="first")
    df["part_id"] = df["part_id"].apply(lambda x: str(x).zfill(4))
    df["defect_type"] = df["defect_type"].apply(
        lambda x: unicodedata.normalize("NFC", str(x).strip()))
    df = df[df["defect_type"].isin(LABELS)]
    df["inspector_note"] = df["inspector_note"].fillna("")

    # 유효 이미지 필터
    def _valid(pid):
        p = os.path.join("part_images", f"{pid}.png")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
        try:
            img = Image.open(p); img.verify(); return True
        except: return False

    df = df[df["part_id"].apply(_valid)]
    expected_samples = len(df)
    expected_dist = df["defect_type"].value_counts().to_dict()

    print("=" * 55 + "\n            문항 6 채점 결과\n" + "=" * 55)

    # ── 코드 분석 ──
    try:
        with open("q6_solution.py", "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code)
        cls = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    except:
        code = ""; cls = []; funcs = []

    # ─── 1. 클래스 구현 (10점) ───
    if "DefectImageLoader" in cls and "InspectionLogProcessor" in cls:
        score += 10; print("  [PASS] 1. 클래스 구현 (DefectImageLoader + InspectionLogProcessor)")
    else:
        print(f"  [FAIL] 1. 클래스 미구현 (발견: {cls})")

    # ─── 2. 데이터 정제 결과 (10점) ───
    ds = ans.get("data_summary", {})
    tvs = ds.get("total_valid_samples", 0)
    ld = ds.get("label_distribution", {})

    if abs(tvs - expected_samples) <= 2:
        label_ok = True
        for lbl in LABELS:
            exp = expected_dist.get(lbl, 0)
            got = ld.get(lbl, 0)
            if abs(exp - got) > 2:
                label_ok = False
                break
        if label_ok:
            score += 10
            print(f"  [PASS] 2. 데이터 정제 (유효={tvs}, 기대={expected_samples})")
        else:
            print(f"  [FAIL] 2. 레이블 분포 불일치 (기대: {expected_dist}, 결과: {ld})")
    else:
        print(f"  [FAIL] 2. 유효 샘플 수 불일치 (기대: {expected_samples}, 결과: {tvs})")

    # ─── 3. conv2d 직접 구현 (10점) ───
    has_conv2d = "def conv2d" in code
    no_cv2_filter = "filter2D" not in code
    has_sobel = ("sobel" in code.lower() or
                 ("[-1, 0, 1]" in code and "[-2, 0, 2]" in code) or
                 ("[-1,0,1]" in code))

    if has_conv2d and no_cv2_filter and has_sobel:
        score += 10; print("  [PASS] 3. conv2d 직접 구현 + Sobel 커널")
    else:
        reasons = []
        if not has_conv2d: reasons.append("conv2d 함수 없음")
        if not no_cv2_filter: reasons.append("filter2D 사용 감지")
        if not has_sobel: reasons.append("Sobel 커널 미정의")
        print(f"  [FAIL] 3. conv2d 구현 문제 ({', '.join(reasons)})")

    # ─── 4. 규칙 기반 결과 (10점) ───
    rb = ans.get("rule_based", {})
    rb_acc = rb.get("test_accuracy", 0)

    if 0.5 <= rb_acc <= 0.95 and rb.get("method") == "edge_threshold_binary":
        score += 10
        print(f"  [PASS] 4. 규칙 기반 (accuracy={rb_acc:.4f}, 이진분류)")
    else:
        print(f"  [FAIL] 4. 규칙 기반 결과 이상 (accuracy={rb_acc})")

    # ─── 5. ML 기반 결과 (10점) ───
    ml = ans.get("ml_based", {})
    ml_acc = ml.get("test_accuracy", 0)
    ml_f1 = ml.get("test_f1_macro", 0)
    pca_n = ml.get("pca_n_components", 0)

    if ml_acc > 0.85 and ml_f1 > 0.7 and 50 <= pca_n <= 400:
        score += 10
        print(f"  [PASS] 5. ML 기반 (acc={ml_acc:.4f}, f1={ml_f1:.4f}, PCA={pca_n})")
    else:
        print(f"  [FAIL] 5. ML 기반 결과 이상 (acc={ml_acc}, f1={ml_f1}, PCA={pca_n})")

    # ─── 6. NN Forward Pass (10점) ───
    nn = ans.get("nn_forward", {})
    nn_acc = nn.get("test_accuracy", 0)

    has_relu = ("relu" in code.lower() or "maximum(0" in code or "np.maximum(0" in code)
    has_softmax = ("softmax" in code.lower() or "exp(" in code)
    has_weight_load = ("pretrained_nn_weights" in code)

    if nn_acc > 0.8 and has_relu and has_softmax and has_weight_load:
        score += 10
        print(f"  [PASS] 6. NN Forward (acc={nn_acc:.4f}, ReLU+Softmax 구현)")
    else:
        reasons = []
        if nn_acc <= 0.8: reasons.append(f"accuracy 낮음({nn_acc})")
        if not has_relu: reasons.append("ReLU 미구현")
        if not has_softmax: reasons.append("Softmax 미구현")
        if not has_weight_load: reasons.append("가중치 미로드")
        print(f"  [FAIL] 6. NN Forward 문제 ({', '.join(reasons)})")

    # ─── 7. 전이학습 비교 (10점) ───
    pre = ans.get("pretrained", {})
    pre_acc = pre.get("test_accuracy", 0)
    tg = ans.get("transfer_gain", None)

    if pre_acc > 0.85 and tg is not None and isinstance(tg, (int, float)):
        cf1 = pre.get("class_f1", {})
        cm = pre.get("confusion_matrix", [])

        has_all_f1 = all(lbl in cf1 for lbl in LABELS)
        has_cm = len(cm) == 5 and all(len(row) == 5 for row in cm)

        if has_all_f1 and has_cm:
            score += 10
            print(f"  [PASS] 7. 전이학습 (pre_acc={pre_acc:.4f}, gain={tg:+.4f})")
        else:
            print(f"  [FAIL] 7. class_f1 또는 confusion_matrix 불완전")
    else:
        print(f"  [FAIL] 7. 전이학습 결과 이상 (acc={pre_acc}, gain={tg})")

    # ─── 8. 개선 실험 (10점) ───
    imp = ans.get("improvement", {})
    before = imp.get("before_f1", 0)
    after = imp.get("after_f1", 0)
    mic = imp.get("most_improved_class", "")

    if before > 0 and after > 0 and mic in LABELS:
        has_cw = "class_weight" in code and "balanced" in code
        if has_cw:
            score += 10
            print(f"  [PASS] 8. 개선 실험 (F1: {before:.4f}→{after:.4f}, 최개선: {mic})")
        else:
            print(f"  [FAIL] 8. class_weight='balanced' 미사용")
    else:
        print(f"  [FAIL] 8. 개선 실험 결과 이상")

    # ─── 9. 비즈니스 보고서 (10점) ───
    report = ans.get("report", {})
    req_sections = ["purpose", "key_results", "transfer_learning_effect", "improvement_suggestion"]

    all_sections = True
    for sec in req_sections:
        val = report.get(sec, "")
        if not isinstance(val, str) or len(val) < 10:
            all_sections = False
            break

    if all_sections:
        score += 10; print("  [PASS] 9. 비즈니스 보고서 (4섹션 완성)")
    else:
        missing = [s for s in req_sections if not isinstance(report.get(s,""), str) or len(report.get(s,"")) < 10]
        print(f"  [FAIL] 9. 보고서 섹션 누락 또는 내용 부족: {missing}")

    # ─── 10. 라이브러리 적절 사용 (10점) ───
    uses_tfidf = "TfidfVectorizer" in code
    uses_lr = "LogisticRegression" in code
    uses_pca = "PCA" in code
    no_keras = "keras" not in code.lower() and "torch" not in code.lower()

    if uses_tfidf and uses_lr and uses_pca and no_keras:
        score += 10
        print(f"  [PASS] 10. 라이브러리 적절 사용 (sklearn O, keras/torch X)")
    else:
        reasons = []
        if not uses_tfidf: reasons.append("TfidfVectorizer 미사용")
        if not uses_lr: reasons.append("LogisticRegression 미사용")
        if not no_keras: reasons.append("keras/torch 사용 감지")
        print(f"  [FAIL] 10. 라이브러리 사용 문제 ({', '.join(reasons)})")

    passed = score >= 95
    print("=" * 55 + f"\n  점수: {score}/100\n  결과: {'✅ PASS' if passed else '❌ FAIL'}\n" + "=" * 55)
    return passed

if __name__ == "__main__":
    sys.exit(0 if verify() else 1)