"""Greedy ensemble search over saved OOF rank-normalised scores.

Per field, greedily build an integer-weighted blend of model OOFs that maximises
SHIFT-CV top-1 accuracy (the faithful test proxy), then report clean+shift and the
combined all-three. Low capacity (integer weights, few models) -> low overfit risk.
"""
import sys, argparse
import numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from featx import FIELDS


def acc_for_blend(scores_list, weights, bounds, isc, truth_idx, n):
    total = np.zeros_like(scores_list[0])
    for w, s in zip(weights, scores_list):
        total += w * s
    correct = np.zeros(n, bool)
    for ri in range(n):
        seg = slice(bounds[ri], bounds[ri+1])
        sc = total[seg].copy()
        sc[isc[seg].astype(bool)] = -1e9
        correct[ri] = (int(np.argmax(sc)) == truth_idx[ri])
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="research/oof_scores.npz")
    ap.add_argument("--models", default="lgbm_rank,lgbm_clf,xgb_rank,xgb_clf,cat_rank,cat_clf")
    ap.add_argument("--max_w", type=int, default=4, help="max total weight per field")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    models = args.models.split(",")
    n = None
    chosen = {}
    field_shift_correct = {}
    field_clean_correct = {}
    for f in FIELDS:
        bounds = d[f"{f}__bounds"]; isc = d[f"{f}__isc"]; truth_idx = d[f"{f}__truth_idx"]
        n = len(bounds) - 1
        shift = {m: d[f"{f}__shift__{m}"] for m in models if f"{f}__shift__{m}" in d}
        clean = {m: d[f"{f}__clean__{m}"] for m in models if f"{f}__clean__{m}" in d}
        avail = list(shift.keys())
        # greedy forward selection on shift accuracy
        weights = {m: 0 for m in avail}
        best_acc = 0.0
        for step in range(args.max_w):
            best_m, best_a = None, best_acc
            for m in avail:
                w2 = dict(weights); w2[m] += 1
                sl = [shift[mm] for mm in avail]; ws = [w2[mm] for mm in avail]
                a = acc_for_blend(sl, ws, bounds, isc, truth_idx, n).mean()
                if a > best_a + 1e-9:
                    best_a, best_m = a, m
            if best_m is None:
                break
            weights[best_m] += 1; best_acc = best_a
        chosen[f] = {m: w for m, w in weights.items() if w > 0}
        # eval chosen on shift + clean
        sl_s = [shift[m] for m in avail]; ws = [weights[m] for m in avail]
        cs = acc_for_blend(sl_s, ws, bounds, isc, truth_idx, n)
        sl_c = [clean[m] for m in avail]
        cc = acc_for_blend(sl_c, ws, bounds, isc, truth_idx, n)
        field_shift_correct[f] = cs; field_clean_correct[f] = cc
        # also single best shift model for reference
        singles = {m: acc_for_blend([shift[m]], [1], bounds, isc, truth_idx, n).mean() for m in avail}
        bestsingle = max(singles, key=singles.get)
        print(f"{f:16s} chosen={chosen[f]}  shift_acc={cs.mean():.3f} clean_acc={cc.mean():.3f}  "
              f"(best single shift: {bestsingle}={singles[bestsingle]:.3f})", flush=True)

    for tag, fc in [("SHIFT", field_shift_correct), ("CLEAN", field_clean_correct)]:
        at = np.ones(n, bool)
        for f in FIELDS: at &= fc[f]
        fa = np.mean([fc[f].mean() for f in FIELDS])
        print(f"{tag} blended ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
    print("CHOSEN WEIGHTS:", chosen, flush=True)


if __name__ == "__main__":
    main()
