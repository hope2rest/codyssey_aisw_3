"""Q2 TF-IDF 검증 Validator (8항목, 100점)"""
import json
import re
import unicodedata
import numpy as np
from pathlib import Path

from grading.core.base_validator import BaseValidator
from grading.core.check_item import CheckItem


class Q2TfidfValidator(BaseValidator):
    mission_id = "aiml_level1_q2_tfidf"
    name = "Q2. TF-IDF 코사인유사도 문서검색"

    def setup(self):
        """참조 답안 계산 (verify_q2.py 로직)"""
        data_dir = self.submission_dir

        # 불용어 로드
        with open(data_dir / "stopwords.txt", "r", encoding="utf-8") as f:
            sw = set(l.strip() for l in f if l.strip())

        def pp(t):
            t = unicodedata.normalize("NFC", t).lower()
            return [
                x for x in re.sub(r"[^가-힣a-z0-9\s]", "", t).split()
                if x not in sw and len(x) > 1
            ]

        # 문서 로드
        with open(data_dir / "documents.txt", "r", encoding="utf-8") as f:
            docs = [l.strip() for l in f if l.strip()]

        tok = [pp(d) for d in docs]
        vocab = sorted(set(w for d in tok for w in d))
        w2i = {w: i for i, w in enumerate(vocab)}
        N, V = len(docs), len(vocab)

        # TF-IDF 행렬 생성
        tf = np.zeros((N, V))
        for i, d in enumerate(tok):
            if not d:
                continue
            for w in d:
                tf[i, w2i[w]] += 1
            tf[i] /= len(d)
        df_v = np.sum(tf > 0, axis=0)
        idf = np.log((N + 1) / (df_v + 1)) + 1
        tfidf = tf * idf

        def cs(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

        def qtf(q):
            t = pp(q)
            qf = np.zeros(V)
            if not t:
                return qf
            for x in t:
                if x in w2i:
                    qf[w2i[x]] += 1
            qf /= len(t)
            return qf * idf

        def ref_search(q):
            qv = qtf(q)
            s = [(i, cs(qv, tfidf[i])) for i in range(N)]
            s.sort(key=lambda x: (-x[1], x[0]))
            top = s[:3]
            if all(v == 0 for _, v in top):
                return [{"doc_index": i, "similarity": 0.0} for i in range(3)]
            return [{"doc_index": i, "similarity": round(v, 6)} for i, v in top]

        # 쿼리 로드
        with open(data_dir / "queries.txt", "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip()]

        self._ref_N = N
        self._ref_V = V
        self._queries = queries
        self._ref_search = ref_search

        # 학생 결과 로드
        with open(data_dir / "result_q2.json", "r", encoding="utf-8") as f:
            self._ans = json.load(f)

    def build_checklist(self):
        ans = self._ans
        N = self._ref_N
        V = self._ref_V
        queries = self._queries
        ref_search = self._ref_search

        # 1. num_documents (10점)
        def check_num_docs():
            if ans.get("num_documents") == N:
                return True
            got = ans.get("num_documents", 0)
            msg = f"결과: {got}, 기대: {N}"
            if got and got > N:
                msg += " (빈 줄을 문서로 포함했을 수 있음)"
            return (False, msg)

        self.checklist.add_item(CheckItem(
            id="1",
            description="num_documents 일치",
            points=10,
            validator=check_num_docs,
            hint="빈 줄은 문서로 포함하지 마세요.",
        ))

        # 2. vocab_size (20점) - NFC 전처리 핵심 검증
        def check_vocab_size():
            if ans.get("vocab_size") == V:
                return True
            diff = (ans.get("vocab_size", 0) or 0) - V
            msg = f"결과: {ans.get('vocab_size')}, 기대: {V}, 차이: {diff:+d}"
            if diff > 0:
                msg += " (NFC 미적용 또는 전처리 순서 오류 가능성)"
            return (False, msg)

        self.checklist.add_item(CheckItem(
            id="2",
            description="vocab_size 일치 (NFC 전처리 검증)",
            points=20,
            validator=check_vocab_size,
            hint="NFC 정규화를 lowercase 전에 적용하세요.",
            ai_trap="NFC 정규화 순서: normalize → lowercase → 특수문자 제거",
        ))

        # 3. tfidf_matrix_shape (5점)
        def check_shape():
            sh = ans.get("tfidf_matrix_shape", [0, 0])
            if sh == [N, V]:
                return True
            return (False, f"결과: {sh}, 기대: [{N}, {V}]")

        self.checklist.add_item(CheckItem(
            id="3",
            description="tfidf_matrix_shape 일치",
            points=5,
            validator=check_shape,
        ))

        # 4-8. 쿼리별 검색 결과 (각 13점, 총 65점)
        qr = ans.get("search_results", [])
        for qi, q in enumerate(queries):
            def make_checker(qi_=qi, q_=q):
                def check_query():
                    ref = ref_search(q_)
                    sub = qr[qi_].get("top3", []) if qi_ < len(qr) else []
                    si = [x["doc_index"] for x in sub]
                    ri = [x["doc_index"] for x in ref]
                    ss = [round(x.get("similarity", 0), 6) for x in sub]
                    rs = [x["similarity"] for x in ref]
                    idx_ok = si == ri
                    sim_ok = (
                        len(ss) == len(rs)
                        and all(abs(a - b) < 1e-3 for a, b in zip(ss, rs))
                    )
                    if idx_ok and sim_ok:
                        return True
                    detail = ""
                    if not idx_ok:
                        detail += f" idx({si} != {ri})"
                    if not sim_ok:
                        detail += " sim 불일치"
                    return (False, detail.strip())
                return check_query

            hints = {
                3: "NFC 미적용 시 '빅데이터'/'분석' 누락 가능",
                4: "한국어 불용어만으로 구성된 쿼리 → 유사도 0이어야 함",
            }
            traps = {
                3: "NFC 정규화 미적용으로 한국어 토큰 분리 오류",
                4: "불용어 필터링 후 빈 쿼리 처리 필요",
            }

            self.checklist.add_item(CheckItem(
                id=str(qi + 4),
                description=f"쿼리{qi + 1} 검색 결과",
                points=13,
                validator=make_checker(),
                hint=hints.get(qi, ""),
                ai_trap=traps.get(qi, ""),
            ))
