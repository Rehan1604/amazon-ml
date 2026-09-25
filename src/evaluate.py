import numpy as np
import pandas as pd

def load_truth(gt):
    """Ground truth table -> dict {S1 id: set of matched S2/S3 ids} (empty set = singleton)."""
    return {s: (set(m.split(",")) if m else set())
            for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}

def split_ids(s1_ids, val_frac=0.2, seed=42):
    """Fixed train/validation split of S1 ids. Same result on every laptop."""
    ids = np.array(sorted(s1_ids))
    mask = np.random.default_rng(seed).random(len(ids)) < val_frac
    return set(ids[~mask]), set(ids[mask])

def f05_one(pred, true):
    """F0.5 for one S1 entity, following the competition rules."""
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0                      # correct singleton
    if not true or not pred:
        return 0.0                      # false match on singleton, or missed everything
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def f05_score(pred_dict, truth_dict, ids):
    """Macro-average F0.5 over the given S1 ids."""
    return float(np.mean([f05_one(pred_dict.get(i, ()), truth_dict[i]) for i in ids]))

def write_tsv(pred_dict, s1_ids, path, col="matched_entity_ids"):
    """Write a submission file: one row per S1 id, comma-separated S2/S3 ids, no duplicates.
    col = 'matched_entity_ids' for matching_results.tsv, 'candidate_entity_ids' for candidate_pairs.tsv"""
    rows = []
    for i in s1_ids:
        ids = [x for x in pred_dict.get(i, []) if x.startswith(("S2-", "S3-"))]
        rows.append((i, ",".join(dict.fromkeys(ids))))
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(
        path, sep="\t", index=False, lineterminator="\n")

def pairs_to_pred(pairs, threshold=0.5):
    """pairs: DataFrame with columns s1_id, other_id, prob.
    Keeps only the best S1 for each S2/S3 record (one-owner rule), then applies the threshold.
    Returns dict {S1 id: [matched ids]}."""
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("other_id")
    best = best[best.prob >= threshold]
    return best.groupby("s1_id")["other_id"].apply(list).to_dict()

def tune_threshold(pairs, truth, ids, thresholds=None):
    """Try several thresholds on validation pairs and print the F0.5 of each.
    Returns (best_threshold, best_score)."""
    if thresholds is None:
        thresholds = [x / 100 for x in range(30, 96, 5)]
    best_t, best_s = None, -1
    for t in thresholds:
        s = f05_score(pairs_to_pred(pairs, t), truth, ids)
        print(f"threshold {t:.2f} -> F0.5 {s:.4f}")
        if s > best_s:
            best_t, best_s = t, s
    print("BEST:", best_t, round(best_s, 4))
    return best_t, best_s