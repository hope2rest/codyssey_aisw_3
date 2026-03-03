import re
import json
import numpy as np
from collections import Counter

DATA_DIR = "C:/Users/ks.lee/Desktop/aisw/aisw/codyssey_aisw/questions/q2_tfidf/data/"

# Load stopwords
with open(DATA_DIR + "stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

# Load documents - skip empty lines
with open(DATA_DIR + "documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# Load queries
with open(DATA_DIR + "queries.txt", "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip()]

print(f"Documents: {len(documents)}")
print(f"Stopwords count: {len(stopwords)}")
print(f"Queries: {queries}")


def preprocess(text):
    # (a) lowercase
    text = text.lower()
    # (b) remove everything except Korean (가-힣), English letters, digits, whitespace
    text = re.sub(r'[^\uAC00-\uD7A3a-z0-9\s]', ' ', text)
    # (c) split by whitespace
    tokens = text.split()
    # (d) remove stopwords
    tokens = [t for t in tokens if t not in stopwords]
    # (e) remove tokens with length <= 1
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# Preprocess all documents
tokenized_docs = [preprocess(doc) for doc in documents]
N = len(documents)

print(f"\nN (doc count): {N}")
for i, t in enumerate(tokenized_docs):
    if len(t) == 0:
        print(f"  Warning: doc {i} has 0 tokens after preprocessing")

# Build vocabulary sorted alphabetically
vocab_set = set()
for tokens in tokenized_docs:
    vocab_set.update(tokens)

vocab = sorted(vocab_set)
vocab_size = len(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}

print(f"Vocab size: {vocab_size}")

# Compute TF matrix: shape (N, vocab_size)
tf_matrix = np.zeros((N, vocab_size), dtype=np.float64)
for i, tokens in enumerate(tokenized_docs):
    if len(tokens) == 0:
        continue
    total = len(tokens)
    counts = Counter(tokens)
    for word, cnt in counts.items():
        if word in word2idx:
            tf_matrix[i, word2idx[word]] = cnt / total

# Compute IDF: log((N+1)/(df+1)) + 1
df = np.sum(tf_matrix > 0, axis=0)  # document frequency per term
idf = np.log((N + 1) / (df + 1)) + 1  # smooth IDF

# TF-IDF matrix
tfidf_matrix = tf_matrix * idf
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def search(query_text, top_n=3):
    # Preprocess query the same way
    q_tokens = preprocess(query_text)
    print(f"  Query tokens: {q_tokens}")

    # Build query TF vector using corpus vocab
    q_tf = np.zeros(vocab_size, dtype=np.float64)
    if len(q_tokens) > 0:
        total = len(q_tokens)
        counts = Counter(q_tokens)
        for word, cnt in counts.items():
            if word in word2idx:
                q_tf[word2idx[word]] = cnt / total

    # Apply corpus IDF (do NOT recompute with query)
    q_tfidf = q_tf * idf

    # Compute cosine similarity with each document
    sims = []
    for doc_idx in range(N):
        sim = cosine_similarity(q_tfidf, tfidf_matrix[doc_idx])
        sims.append((doc_idx, sim))

    # Sort: descending similarity, then ascending doc_index for ties
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:top_n]


# Run search for each query and collect results
search_results = []
for query in queries:
    print(f"\nQuery: '{query}'")
    top3 = search(query, top_n=3)
    print(f"  Top3: {top3}")
    result = {
        "query": query,
        "top3": [
            {"doc_index": int(doc_idx), "similarity": round(float(sim), 6)}
            for doc_idx, sim in top3
        ]
    }
    search_results.append(result)

# Build output JSON
output = {
    "vocab_size": vocab_size,
    "tfidf_matrix_shape": list(tfidf_matrix.shape),
    "search_results": search_results
}

output_path = DATA_DIR + "result_q2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved result to: {output_path}")
print(json.dumps(output, ensure_ascii=False, indent=2))
