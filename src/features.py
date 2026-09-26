import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# ---------- extra cleaning applied inside the features ----------
LEGAL_FUZZY = ["private", "limited", "llp", "llc", "incorporated", "corporation", "company",
               "pvt", "ltd", "inc", "corp", "elelpi", "elelsi", "praivet", "limitet"]
LEGAL_EXACT = {"pra", "li", "co", "sa", "sas", "sarl", "lp", "plc", "pllc", "opc", "the", "and", "of"}
HONOR = {"m", "s", "ms", "mr", "mrs", "dr", "sri", "shri", "shree", "smt", "messrs"}
JUNK = {"null", "none", "nan"}

def _norm_core(s):
    """Remove prefixes (mr, dr, sri...) and legal words even when misspelled (plrhivate, praivet)."""
    w = s.split()
    while len(w) > 1 and w[0] in HONOR:
        w = w[1:]
    out = []
    for t in w:
        if t in LEGAL_EXACT:
            continue
        if len(t) >= 4 and max(fuzz.ratio(t, l) for l in LEGAL_FUZZY) >= 80:
            continue
        out.append(t)
    return " ".join(out) if out else " ".join(w)

def _norm_addr(s):
    """Remove leading zeros from numbers ('009105' -> '9105') and junk words ('null')."""
    out = []
    for t in s.split():
        if t in JUNK:
            continue
        if t.isdigit():
            t = str(int(t))
        out.append(t)
    return " ".join(out)

def _map_unique(series, fn):
    """Apply fn once per unique value (much faster)."""
    u = pd.unique(series)
    return series.map(dict(zip(u, [fn(x) for x in u])))

# ---------- helpers ----------
def _sim(a, b, scorer):
    """Similarity score (0-100) for each pair (a[i], b[i]), using all CPU cores."""
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1).astype(np.float32)

def _jaccard(a, b):
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out

def _token_align(a, b):
    """Word-by-word: how well each word finds a partner in the other name.
    A typo keeps high similarity; a swapped word (sibling trap) gives a low one."""
    n = len(a)
    res = np.zeros((n, 6), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        ta, tb = x.split(), y.split()
        if not ta or not tb:
            res[i] = (0, 0, len(ta), 0, len(tb), 0)
            continue
        best_a = [max(fuzz.ratio(u, v) for v in tb) for u in ta]
        best_b = [max(fuzz.ratio(v, u) for u in ta) for v in tb]
        res[i] = (min(best_a), sum(best_a) / len(best_a), sum(s < 80 for s in best_a),
                  min(best_b), sum(s < 80 for s in best_b), fuzz.ratio(ta[0], tb[0]))
    return res

def _num_feats(a, b):
    """Numbers in the two addresses (leading zeros already removed)."""
    n = len(a)
    res = np.zeros((n, 4), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        na = {t for t in x.split() if t.isdigit()}
        nb = {t for t in y.split() if t.isdigit()}
        if not na or not nb:
            res[i] = (0, 0, len(na), 0)
            continue
        common = na & nb
        close = any(len(u) == len(v) and len(u) >= 2 and abs(int(u) - int(v)) <= 20
                    for u in na - common for v in nb - common)
        res[i] = (len(common) / len(na | nb), max((len(t) for t in common), default=0),
                  len(na - nb), float(close))
    return res

# ---------- main ----------
def build_features(cands, s1c, qc):
    """For every candidate pair, compute similarity numbers the model learns from.
    Returns (table with entity_id, s1_id + features, list of feature names)."""
    q = qc[["entity_id", "core", "addr"]]
    s = s1c[["entity_id", "core", "addr"]].rename(columns={
        "entity_id": "s1_id", "core": "s1_core", "addr": "s1_addr"})
    d = cands[["entity_id", "s1_id"]].merge(q, on="entity_id").merge(s, on="s1_id")

    core = _map_unique(d.core, _norm_core)
    s1_core = _map_unique(d.s1_core, _norm_core)
    addr = _map_unique(d.addr, _norm_addr)
    s1_addr = _map_unique(d.s1_addr, _norm_addr)

    f = pd.DataFrame(index=d.index)
    # 1. Name similarity (several ways)
    f["name_set"] = _sim(core, s1_core, fuzz.token_set_ratio)
    f["name_sort"] = _sim(core, s1_core, fuzz.token_sort_ratio)
    f["name_partial"] = _sim(core, s1_core, fuzz.partial_ratio)
    f["name_nospace"] = _sim(core.str.replace(" ", ""), s1_core.str.replace(" ", ""), fuzz.ratio)
    f["name_jw"] = _sim(core, s1_core, JaroWinkler.normalized_similarity)
    f["name_jac"] = _jaccard(core, s1_core)
    f["name_len_diff"] = (core.str.len() - s1_core.str.len()).abs().astype(np.float32)
    # 2. Word-by-word alignment (catches swapped words = sibling traps)
    ta = _token_align(core.values, s1_core.values)
    for j, c in enumerate(["tok_min", "tok_mean", "tok_nbad", "tok_min_rev", "tok_nbad_rev", "tok_first"]):
        f[c] = ta[:, j]
    # 3. Address similarity
    f["addr_set"] = _sim(addr, s1_addr, fuzz.token_set_ratio)
    f["addr_sort"] = _sim(addr, s1_addr, fuzz.token_sort_ratio)
    f["addr_jac"] = _jaccard(addr, s1_addr)
    # 4. Numbers (house number, PIN): exact vs 'close but different'
    nf = _num_feats(addr.values, s1_addr.values)
    for j, c in enumerate(["num_jac", "num_longest_shared", "num_q_missing", "num_close_diff"]):
        f[c] = nf[:, j]
    f["q_no_addr"] = (addr == "").astype(np.int8)
    f["s1_no_addr"] = (s1_addr == "").astype(np.int8)
    f["is_s3"] = d.entity_id.str.startswith("S3").astype(np.int8)
    # 5. Compared with the OTHER candidates of the same record
    f["combo"] = f.name_set + f.addr_set
    for c in ["name_set", "addr_set", "name_jw", "addr_jac", "combo", "tok_min"]:
        f[c + "_gap"] = f.groupby(d.entity_id)[c].transform("max") - f[c]
    f["n_cands"] = d.groupby("entity_id")["entity_id"].transform("size").astype(np.float32)

    out = pd.concat([d[["entity_id", "s1_id"]], f], axis=1)
    return out, list(f.columns)