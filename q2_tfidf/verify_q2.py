"""문제 2 검증"""
import numpy as np, re, json, sys, unicodedata

def verify():
    score = 0
    with open("stopwords.txt","r",encoding="utf-8") as f:
        sw = set(l.strip() for l in f if l.strip())
    def pp(t):
        t = unicodedata.normalize('NFC', t).lower()
        return [x for x in re.sub(r"[^가-힣a-z0-9\s]","",t).split() if x not in sw and len(x)>1]
    with open("documents.txt","r",encoding="utf-8") as f:
        docs = [l.strip() for l in f if l.strip()]
    tok = [pp(d) for d in docs]
    vocab = sorted(set(w for d in tok for w in d))
    w2i = {w:i for i,w in enumerate(vocab)}
    N, V = len(docs), len(vocab)
    tf = np.zeros((N,V))
    for i,d in enumerate(tok):
        if not d: continue
        for w in d: tf[i,w2i[w]] += 1
        tf[i] /= len(d)
    df_v = np.sum(tf>0,axis=0)
    idf = np.log((N+1)/(df_v+1))+1
    tfidf = tf*idf
    def cs(a,b):
        na,nb = np.linalg.norm(a),np.linalg.norm(b)
        return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))
    def qtf(q):
        t=pp(q); qf=np.zeros(V)
        if not t: return qf
        for x in t:
            if x in w2i: qf[w2i[x]]+=1
        qf/=len(t); return qf*idf
    def ref_search(q):
        qv=qtf(q); s=[(i,cs(qv,tfidf[i])) for i in range(N)]
        s.sort(key=lambda x:(-x[1],x[0])); top=s[:3]
        if all(v==0 for _,v in top):
            return [{"doc_index":i,"similarity":0.0} for i in range(3)]
        return [{"doc_index":i,"similarity":round(v,6)} for i,v in top]
    with open("queries.txt","r",encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip()]
    try:
        with open("result_q2.json","r",encoding="utf-8") as f: ans=json.load(f)
    except: print("FAIL: result_q2.json 없음"); return False

    print("="*55+"\n            문항 2 채점 결과\n"+"="*55)
    # 1. num_documents
    if ans.get("num_documents")==N:
        score+=10; print(f"  [PASS] 1. num_documents: {N}")
    else:
        print(f"  [FAIL] 1. num_documents: {ans.get('num_documents')} (기대:{N})")
        if ans.get("num_documents",0)>N: print("         → 빈 줄을 문서로 포함")
    # 2. vocab_size (NFD+전처리순서 핵심 검증)
    if ans.get("vocab_size")==V:
        score+=20; print(f"  [PASS] 2. vocab_size: {V}")
    else:
        diff = (ans.get("vocab_size",0) or 0) - V
        print(f"  [FAIL] 2. vocab_size: {ans.get('vocab_size')} (기대:{V}, 차이:{diff:+d})")
        if diff > 0: print("         → NFC 미적용 또는 전처리 순서 오류 가능성")
    # 3. matrix_shape
    sh = ans.get("tfidf_matrix_shape",[0,0])
    if sh == [N,V]:
        score+=5; print(f"  [PASS] 3. matrix_shape: {sh}")
    else: print(f"  [FAIL] 3. matrix_shape: {sh} (기대:[{N},{V}])")
    # 4-8. 쿼리별 검색 결과
    qr = ans.get("search_results",[])
    qp = 0
    for qi,q in enumerate(queries):
        ref = ref_search(q)
        sub = qr[qi].get("top3",[]) if qi<len(qr) else []
        si = [x["doc_index"] for x in sub]
        ri = [x["doc_index"] for x in ref]
        ss = [round(x.get("similarity",0),6) for x in sub]
        rs = [x["similarity"] for x in ref]
        idx_ok = si == ri
        sim_ok = len(ss)==len(rs) and all(abs(a-b)<1e-3 for a,b in zip(ss,rs))
        if idx_ok and sim_ok:
            qp += 1; print(f"  [PASS] 쿼리{qi+1}: → {si} (sim:{rs[0]:.4f})")
        else:
            detail = ""
            if not idx_ok: detail += f" idx({si}≠{ri})"
            if not sim_ok: detail += f" sim불일치"
            # 함정 힌트(제공 여부는 확인 필요)
            if qi==3 and ans.get("vocab_size",0)!=V:
                detail += " ← NFC 미적용으로 빅데이터/분석 누락 가능"
            if qi==4 and any(x.get("similarity",1)!=0 for x in sub):
                detail += " ← 한국어 불용어 쿼리: 유사도 0이어야 함"
            print(f"  [FAIL] 쿼리{qi+1}:{detail}")
    score += int(qp / len(queries) * 65)
    passed = score >= 95
    print("="*55+f"\n  점수: {score}/100\n  결과: {'✅ PASS' if passed else '❌ FAIL'}\n"+"="*55)
    return passed

if __name__=="__main__": sys.exit(0 if verify() else 1)
