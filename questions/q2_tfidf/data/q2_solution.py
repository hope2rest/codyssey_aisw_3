"""문제 2 정답"""
import numpy as np, re, json, unicodedata

# 불용어 로드 (strip + 중복/빈줄 제거)
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(l.strip() for l in f if l.strip())

def preprocess(text):
    text = unicodedata.normalize('NFC', text)   # [함정A] NFD→NFC
    text = text.lower()                          # [함정C] 소문자 먼저
    text = re.sub(r"[^가-힣a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]  # [함정C] 소문자 후 불용어
    return [t for t in tokens if len(t) > 1]

# 문서 로드 (빈 줄 무시)
with open("documents.txt", "r", encoding="utf-8") as f:
    docs = [l.strip() for l in f if l.strip()]

tokenized = [preprocess(d) for d in docs]
N = len(docs)
print(f"문서 수: {N}")

# 빈 문서 확인 [함정B]
for i, t in enumerate(tokenized):
    if len(t) == 0:
        print(f"  [주의] 문서 {i}: 전처리 후 토큰 0개")

vocab = sorted(set(w for d in tokenized for w in d))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"어휘 수: {V}")

# TF (빈 문서는 0벡터) [함정B]
tf = np.zeros((N, V))
for i, d in enumerate(tokenized):
    if len(d) == 0: continue  # 빈 문서 → TF 0벡터
    for w in d:
        tf[i, w2i[w]] += 1
    tf[i] /= len(d)

# Smooth IDF
df_vec = np.sum(tf > 0, axis=0)
idf = np.log((N + 1) / (df_vec + 1)) + 1
tfidf = tf * idf

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def query_tfidf(q):
    tokens = preprocess(q)
    qtf = np.zeros(V)
    if len(tokens) == 0: return qtf
    for t in tokens:
        if t in w2i: qtf[w2i[t]] += 1
    qtf /= len(tokens)
    return qtf * idf

def search(query, top_n=3):
    qv = query_tfidf(query)
    sims = [(i, cos_sim(qv, tfidf[i])) for i in range(N)]
    sims.sort(key=lambda x: (-x[1], x[0]))
    top = sims[:top_n]
    if all(s == 0 for _, s in top):
        return [{"doc_index": i, "similarity": 0.0} for i in range(top_n)]
    return [{"doc_index": i, "similarity": round(s, 6)} for i, s in top]

with open("queries.txt", "r", encoding="utf-8") as f:
    queries = [l.strip() for l in f if l.strip()]

sr = []
for q in queries:
    top3 = search(q)
    sr.append({"query": q, "top3": top3})
    print(f"쿼리: '{q[:25]}...' → {[t['doc_index'] for t in top3]}")

result = {
    "num_documents": N,
    "vocab_size": V,
    "tfidf_matrix_shape": [N, V],
    "search_results": sr
}
with open("result_q2.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("result_q2.json 저장 완료!")
