import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

N_THREADS = max(1, os.cpu_count() - 2)   # use almost all CPU cores

def tfidf_topk(s1c, qc, col, K, n_threads=N_THREADS):
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
    """Shortlist = top-k similar names + top-k similar addresses (for small samples)."""
    n = tfidf_topk(s1c, qc, "core", k_name).rename(columns={"sim": "name_sim", "rank": "name_rank"})
    a = tfidf_topk(s1c, qc, "addr", k_addr).rename(columns={"sim": "addr_sim", "rank": "addr_rank"})
    c = n.merge(a, on=["entity_id", "s1_id"], how="outer")
    c[["name_sim", "addr_sim"]] = c[["name_sim", "addr_sim"]].fillna(0)
    c[["name_rank", "addr_rank"]] = c[["name_rank", "addr_rank"]].fillna(99)
    return c

def _fit(texts):
    """Prepare TF-IDF for the S1 side once (reused for every chunk)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                          sublinear_tf=True, dtype=np.float32)
    A = vec.fit_transform(texts)
    return vec, A.T.tocsr()

def generate_candidates_big(s1c, qc, out_prefix, k_name=10, k_addr=10,
                            chunk=500_000, n_threads=N_THREADS):
    """Same shortlist as generate_candidates, but for FULL data:
    works in chunks of records and saves each chunk to disk.
    Returns the list of saved file paths."""
    files = []
    for c in qc.country.unique():
        a = s1c[s1c.country == c].reset_index(drop=True)
        b_all = qc[qc.country == c].reset_index(drop=True)
        if len(a) == 0 or len(b_all) == 0:
            continue
        fitted = {}
        for col in ["core", "addr"]:
            try:
                fitted[col] = _fit(a[col])
            except ValueError:
                fitted[col] = None
        for start in range(0, len(b_all), chunk):
            t = time.time()
            b = b_all.iloc[start:start + chunk]
            parts = []
            for col, K, nm in [("core", k_name, "name"), ("addr", k_addr, "addr")]:
                if fitted[col] is None:
                    continue
                vec, AT = fitted[col]
                C = sp_matmul_topn(vec.transform(b[col]), AT, top_n=K, threshold=0.05,
                                   sort=True, n_threads=n_threads).tocsr()
                rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
                rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr))
                parts.append(pd.DataFrame({"entity_id": b.entity_id.values[rows],
                                           "s1_id": a.entity_id.values[C.indices],
                                           f"{nm}_sim": C.data, f"{nm}_rank": rank}))
            if not parts:
                continue
            df = parts[0] if len(parts) == 1 else parts[0].merge(
                parts[1], on=["entity_id", "s1_id"], how="outer")
            for nm in ["name", "addr"]:
                if f"{nm}_sim" not in df:
                    df[f"{nm}_sim"], df[f"{nm}_rank"] = 0.0, 99
            df[["name_sim", "addr_sim"]] = df[["name_sim", "addr_sim"]].fillna(0)
            df[["name_rank", "addr_rank"]] = df[["name_rank", "addr_rank"]].fillna(99)
            path = f"{out_prefix}_{c}_{start}.pkl"
            df.to_pickle(path)
            files.append(path)
            print(f"{c} rows {start:,}-{start + len(b):,}: {len(df):,} pairs, "
                  f"{round(time.time() - t)}s", flush=True)
    return files

def keep_rarest(B, m):
    """Keep only the m highest-weight (= rarest) pieces of each record."""
    B = B.tocsr().copy()
    ip, data = B.indptr, B.data
    for i in range(B.shape[0]):
        s, e = ip[i], ip[i + 1]
        if e - s > m:
            row = data[s:e]
            cut = np.partition(row, e - s - m)[e - s - m]
            row[row < cut] = 0
    B.eliminate_zeros()
    return B

def _make_vec(kind):
    if kind == "word":   # whole words; keep words seen once (e.g. house numbers)
        return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                           sublinear_tf=True, dtype=np.float32)

# (column, kind, how many rarest pieces/words to search with)
FAST_SEARCHES = [("core", "char", 5), ("core", "word", 3), ("addr", "word", 4)]

def generate_candidates_fast(s1c, qc, out_prefix, K=10, chunk=250_000, n_threads=N_THREADS):
    """FINAL blocking: 3 fast searches (name pieces, name words, address words),
    top-K each, combined. Works in chunks saved to disk. Returns file paths."""
    files = []
    for c in qc.country.unique():
        a = s1c[s1c.country == c].reset_index(drop=True)
        b_all = qc[qc.country == c].reset_index(drop=True)
        if len(a) == 0 or len(b_all) == 0:
            continue
        fitted = []
        for col, kind, m in FAST_SEARCHES:
            vec = _make_vec(kind)
            try:
                A = vec.fit_transform(a[col])
                fitted.append((vec, A.T.tocsr(), col, m))
            except ValueError:
                pass
        for start in range(0, len(b_all), chunk):
            t = time.time()
            b = b_all.iloc[start:start + chunk]
            parts = []
            for vec, AT, col, m in fitted:
                B = keep_rarest(vec.transform(b[col]), m)
                C = sp_matmul_topn(B, AT, top_n=K, threshold=0.01, sort=True,
                                   n_threads=n_threads).tocsr()
                rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
                parts.append(pd.DataFrame({"entity_id": b.entity_id.values[rows],
                                           "s1_id": a.entity_id.values[C.indices]}))
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            path = f"{out_prefix}_{c}_{start}.pkl"
            df.to_pickle(path)
            files.append(path)
            print(f"{c} rows {start:,}-{start + len(b):,} of {len(b_all):,}: "
                  f"{len(df):,} pairs, {round(time.time() - t)}s", flush=True)
    return files