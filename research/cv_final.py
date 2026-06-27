"""5-fold GroupKFold CV mirroring solution.py's seed-bagged ensemble exactly."""
import sys, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata

sys.path.insert(0, ".")
from solution import (load_rows, build_field_dataset, ranker_params, clf_params,
                      per_row_rank_avg, FIELDS, SEEDS)

ROOT = "dataset/public"


def oof(rows, field, n_splits=5):
    X, y, g, meta, _ = build_field_dataset(rows, field, with_labels=True)
    bounds = np.concatenate([[0], np.cumsum(g)])
    n = len(rows)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    score_cols = [np.zeros(len(y)) for _ in range(2 * len(SEEDS))]
    for tr, va in kf.split(np.arange(n)):
        trm = np.zeros(len(y), bool); vam = np.zeros(len(y), bool); tg = []
        for ri in tr: trm[bounds[ri]:bounds[ri+1]] = True; tg.append(g[ri])
        for ri in va: vam[bounds[ri]:bounds[ri+1]] = True
        k = 0
        for seed in SEEDS:
            rk = lgb.LGBMRanker(**ranker_params(seed)); rk.fit(X[trm], y[trm].astype(int), group=tg)
            score_cols[k][vam] = rk.predict(X[vam]); k += 1
            cl = lgb.LGBMClassifier(**clf_params(seed)); cl.fit(X[trm], y[trm].astype(int))
            score_cols[k][vam] = cl.predict_proba(X[vam])[:, 1]; k += 1
    ens = per_row_rank_avg(score_cols, bounds, n)
    correct = np.zeros(n, bool)
    for ri, toks, truth, isc in meta:
        s = ens[bounds[ri]:bounds[ri+1]].copy(); s[isc.astype(bool)] = -np.inf
        correct[ri] = (toks[int(np.argmax(s))] == truth)
    return correct


def main():
    t0 = time.time()
    rows = load_rows(ROOT + "/train.csv")
    cc, accs = {}, {}
    for f in FIELDS:
        c = oof(rows, f); cc[f] = c; accs[f] = c.mean()
        print(f"  {f:18s} top-1 = {c.mean():.4f}")
    at = np.ones(len(rows), bool)
    for f in FIELDS: at &= cc[f]
    fa = np.mean(list(accs.values()))
    print(f"all-three = {at.mean():.4f}  field_acc = {fa:.4f}  SCORE = {0.97*at.mean()+0.03*fa:.4f}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
