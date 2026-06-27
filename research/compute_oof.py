"""Compute and SAVE per-(model,field) OOF rank-normalised scores under clean and
shift CV, so ensemble subsets/weights can be searched offline. Saves one npz."""
import sys, json, csv, time, argparse
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import rankdata
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from featx import build_row_field, FIELDS, OPTIONS_KEY
from bakeoff import (load_rows, build_field, TRAINERS, oof_for_model, rank_norm)
csv.field_size_limit(10**8)
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm_rank,lgbm_clf,xgb_rank,xgb_clf,cat_rank,cat_clf")
    ap.add_argument("--n_aug", type=int, default=2)
    ap.add_argument("--out", default="research/oof_scores.npz")
    args = ap.parse_args()
    models = args.models.split(",")
    rows = load_rows("dataset/public/train.csv")
    n = len(rows)
    save = {}
    for f in FIELDS:
        # meta from clean build
        clean = build_field(rows, f, None)
        glen = [len(clean[ri][1]) for ri in range(n)]
        bounds = np.concatenate([[0], np.cumsum(glen)])
        isc = np.concatenate([clean[ri][4] for ri in range(n)])
        save[f"{f}__bounds"] = bounds
        save[f"{f}__isc"] = isc
        # store, per row, index of truth within its segment
        truth_idx = np.full(n, -1)
        for ri in range(n):
            X, y, toks, truth, isc_r = clean[ri]
            for j, t in enumerate(toks):
                if t == truth:
                    truth_idx[ri] = j; break
        save[f"{f}__truth_idx"] = truth_idx
        for mode, pv in [("clean", False), ("shift", True)]:
            for mk in models:
                t0 = time.time()
                oof, _, bnd = oof_for_model(rows, f, mk, args.n_aug, pv)
                rn = rank_norm(oof, bnd, n)
                save[f"{f}__{mode}__{mk}"] = rn.astype(np.float32)
                print(f"  {f:16s} {mode:5s} {mk:10s} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(args.out, **save)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
