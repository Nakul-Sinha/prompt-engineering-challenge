"""Diagnose the CV/test gap: CV variance, train-vs-test distribution shift,
and the difficulty signal (best-vs-2nd-best distance gap)."""
import sys
import numpy as np
sys.path.insert(0, "research")
from features import load_rows, build_row_field, smiles_vec, FIELDS, OPTIONS_KEY

tr = load_rows("dataset/public/train.csv")
te = load_rows("dataset/public/test.csv")
print(f"train={len(tr)} test={len(te)}")


def l1_scalar_atom(cc_vec, cand_vec):
    return np.abs(cc_vec[:15] - cand_vec[:15]).sum()  # 6 scalar + 9 atom


def field_difficulty(rows, field, have_truth):
    """For each row: distance from corrupted to each candidate's support card.
    Returns arrays of (min_dist, gap best-vs-2nd, truth_rank if available)."""
    mins, gaps, truth_ranks, truth_is_min = [], [], [], []
    for r in rows:
        cc = r["corrupted_card"]
        cc_vec = smiles_vec(cc["smiles_features"])
        sup = {s["candidate_token"]: s for s in r["support_cards"]
               if s["repair_field"] == field}
        corrupt = cc[field]
        toks, dists = [], []
        for tok in r[OPTIONS_KEY[field]]:
            if tok == corrupt:
                continue
            s = sup.get(tok)
            if s is None:
                d = 1e9
            else:
                d = l1_scalar_atom(cc_vec, smiles_vec(s["smiles_features"]))
            toks.append(tok); dists.append(d)
        dists = np.array(dists)
        order = dists.argsort()
        mins.append(dists[order[0]])
        gaps.append(dists[order[1]] - dists[order[0]] if len(dists) > 1 else 0)
        if have_truth:
            truth = r["answer"][field]
            ti = toks.index(truth)
            rank = int((dists < dists[ti]).sum())
            truth_ranks.append(rank)
            truth_is_min.append(rank == 0)
    return np.array(mins), np.array(gaps), np.array(truth_ranks), np.array(truth_is_min)


for f in FIELDS:
    mtr, gtr, rk, im = field_difficulty(tr, f, True)
    mte, gte, _, _ = field_difficulty(te, f, False)
    print(f"\n=== {f} ===")
    print(f"  TRAIN min_dist  mean={mtr.mean():.2f} med={np.median(mtr):.1f}  "
          f"TEST min_dist mean={mte.mean():.2f} med={np.median(mte):.1f}")
    print(f"  TRAIN gap(best-2nd) mean={gtr.mean():.2f} med={np.median(gtr):.1f}  "
          f"TEST gap mean={gte.mean():.2f} med={np.median(gte):.1f}")
    print(f"  TRAIN truth-is-nearest(L1 scalar+atom) = {im.mean():.3f}  "
          f"truth_rank mean={rk.mean():.2f}")
