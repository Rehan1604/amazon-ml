import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

def tfidf_topk(s1c, qc, col, K, n_threads=4):
    """For each record in qc, find the K most similar S1 records (same country)
    by comparing column `col` in 3-letter pieces (TF-IDF).
    Returns columns: entity_id, s1_id, sim, rank."""
    out = []
    for c in qc.country.unique():
        a, b = s1c[s1c.country == c], qc[qc.country == c]
        if len(a) == 0 or len(b) == 0:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                              sublinear_tf=True, dtype=np.float32)
        try:
            A = vec.fit_transform(a[col])
        except ValueError:          # all texts empty for this country
            continue
        B = vec.transform(b[col])
        C = sp_matmul_topn(B, A.T.tocsr(), top_n=K, threshold=0.05,
                           sort=True, n_threads=n_threads).tocsr()
        rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr))
        out.append(pd.DataFrame({"entity_id": b.entity_id.values[rows],
                                 "s1_id": a.entity_id.values[C.indices],
                                 "sim": C.data, "rank": rank}))
    if not out:
        return pd.DataFrame(columns=["entity_id", "s1_id", "sim", "rank"])
    return pd.concat(out, ignore_index=True)

def generate_candidates(s1c, qc, k_name=10, k_addr=10):
    """Shortlist = top-k similar names + top-k similar addresses.
    Returns one row per (S2/S3 record, S1 candidate) with the TF-IDF scores."""
    n = tfidf_topk(s1c, qc, "core", k_name).rename(columns={"sim": "name_sim", "rank": "name_rank"})
    a = tfidf_topk(s1c, qc, "addr", k_addr).rename(columns={"sim": "addr_sim", "rank": "addr_rank"})
    c = n.merge(a, on=["entity_id", "s1_id"], how="outer")
    c[["name_sim", "addr_sim"]] = c[["name_sim", "addr_sim"]].fillna(0)
    c[["name_rank", "addr_rank"]] = c[["name_rank", "addr_rank"]].fillna(99)
    return c