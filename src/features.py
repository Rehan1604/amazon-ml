import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

def _sim(a, b, scorer):
    """Similarity score (0-100) for each pair (a[i], b[i]), using all CPU cores."""
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1).astype(np.float32)

def _jaccard(a, b):
    """Share of words the two texts have in common (0-1)."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out

def build_features(cands, s1c, qc):
    """For every candidate pair, compute similarity numbers the model learns from.
    Returns (table with entity_id, s1_id + features, list of feature names)."""
    q = qc[["entity_id", "core", "addr", "nums"]]
    s = s1c[["entity_id", "core", "addr", "nums"]].rename(columns={
        "entity_id": "s1_id", "core": "s1_core", "addr": "s1_addr", "nums": "s1_nums"})
    d = cands.merge(q, on="entity_id").merge(s, on="s1_id")

    f = pd.DataFrame(index=d.index)
    # 1. Scores from the blocking step
    # 2. Name similarity, several ways (each catches different differences)
    f["name_set"] = _sim(d.core, d.s1_core, fuzz.token_set_ratio)       # ignores extra words
    f["name_sort"] = _sim(d.core, d.s1_core, fuzz.token_sort_ratio)     # ignores word order
    f["name_partial"] = _sim(d.core, d.s1_core, fuzz.partial_ratio)     # one inside the other
    f["name_nospace"] = _sim(d.core.str.replace(" ", ""), d.s1_core.str.replace(" ", ""), fuzz.ratio)  # website names
    f["name_jw"] = _sim(d.core, d.s1_core, JaroWinkler.normalized_similarity)  # typos
    f["name_jac"] = _jaccard(d.core, d.s1_core)
    f["name_len_diff"] = (d.core.str.len() - d.s1_core.str.len()).abs().astype(np.float32)
    # 3. Address similarity
    f["addr_set"] = _sim(d.addr, d.s1_addr, fuzz.token_set_ratio)
    f["addr_sort"] = _sim(d.addr, d.s1_addr, fuzz.token_sort_ratio)
    f["addr_jac"] = _jaccard(d.addr, d.s1_addr)
    # 4. Address numbers (house no, PIN): very strong clues
    f["num_jac"] = _jaccard(d.nums, d.s1_nums)
    first_q = d.nums.str.split().str[0].fillna("")
    first_s = d.s1_nums.str.split().str[0].fillna("")
    f["num_first_eq"] = ((first_q == first_s) & (first_q != "")).astype(np.int8)
    f["q_no_num"] = (d.nums == "").astype(np.int8)
    f["s1_no_num"] = (d.s1_nums == "").astype(np.int8)
    f["q_no_addr"] = (d.addr == "").astype(np.int8)
    f["s1_no_addr"] = (d.s1_addr == "").astype(np.int8)
    f["is_s3"] = d.entity_id.str.startswith("S3").astype(np.int8)
    # 5. Compared with the OTHER candidates of the same record
    f["combo"] = f.name_set + f.addr_set
    for c in ["name_set", "addr_set", "name_jw", "addr_jac", "combo"]:
        f[c + "_gap"] = f.groupby(d.entity_id)[c].transform("max") - f[c]
    f["n_cands"] = d.groupby("entity_id")["entity_id"].transform("size").astype(np.float32)

    out = pd.concat([d[["entity_id", "s1_id"]], f], axis=1)
    return out, list(f.columns)