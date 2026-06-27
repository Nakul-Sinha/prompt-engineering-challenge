"""CV with an ensemble (lambdarank + binary classifier) per field, plus an
optional cross-field stacking stage for name_type. OOF-honest."""
import sys, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata

sys.path.insert(0, "research")
from features import load_rows, build_row_field, FIELDS, OPTIONS_KEY

SEED = 42


def build_field_dataset(rows, field):
    Xs, ys, groups, meta = [], [], [], []
    names = None
    for ri, r in enumerate(rows):
        opts = r[OPTIONS_KEY[field]]
        X, toks, names, is_corrupt = build_row_field(r["corrupted_card"], r["support_cards"], opts, field)
        truth = r["answer"][field] if "answer" in r else None
        y = np.array([1.0 if t == truth else 0.0 for t in toks])
        Xs.append(X); ys.append(y); groups.append(len(toks))
        meta.append((ri, toks, truth, is_corrupt))
    return np.vstack(Xs), np.concatenate(ys), np.array(groups), meta, names


def ranker_params():
    return dict(objective="lambdarank", metric="ndcg", n_estimators=400,
                learning_rate=0.04, num_leaves=31, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)


def clf_params():
    return dict(objective="binary", n_estimators=400, learning_rate=0.04,
                num_leaves=31, min_child_samples=30, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbosity=-1)


def per_row_rank(pred, bounds, ri):
    seg = pred[bounds[ri]:bounds[ri+1]]
    return rankdata(seg) / len(seg)


def oof_scores(rows, field, n_splits=5):
    """Return oof ensemble score array + meta + bounds (group-honest)."""
    X, y, groups, meta, names = build_field_dataset(rows, field)
    bounds = np.concatenate([[0], np.cumsum(groups)])
    n_rows = len(rows)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_rank = np.zeros(len(y))
    oof_clf = np.zeros(len(y))
    for tr_idx, va_idx in kf.split(np.arange(n_rows)):
        tr_mask = np.zeros(len(y), bool); va_mask = np.zeros(len(y), bool)
        tr_group = []
        for ri in tr_idx:
            tr_mask[bounds[ri]:bounds[ri+1]] = True; tr_group.append(groups[ri])
        for ri in va_idx:
            va_mask[bounds[ri]:bounds[ri+1]] = True
        rk = lgb.LGBMRanker(**ranker_params())
        rk.fit(X[tr_mask], y[tr_mask].astype(int), group=tr_group)
        oof_rank[va_mask] = rk.predict(X[va_mask])
        cl = lgb.LGBMClassifier(**clf_params())
        cl.fit(X[tr_mask], y[tr_mask].astype(int))
        oof_clf[va_mask] = cl.predict_proba(X[va_mask])[:, 1]
    # per-row rank-normalize each then average
    ens = np.zeros(len(y))
    for ri in range(n_rows):
        s = slice(bounds[ri], bounds[ri+1])
        r1 = rankdata(oof_rank[s]); r2 = rankdata(oof_clf[s])
        n = (bounds[ri+1]-bounds[ri])
        ens[s] = (r1 + r2) / (2*n)
    return ens, meta, bounds, names


def eval_field(ens, meta, bounds):
    correct = np.zeros(len(meta), bool)
    for ri, toks, truth, is_corrupt in meta:
        s = ens[bounds[ri]:bounds[ri+1]].copy()
        s[is_corrupt.astype(bool)] = -np.inf
        correct[ri] = (toks[int(np.argmax(s))] == truth)
    return correct


def main():
    t0 = time.time()
    rows = load_rows("dataset/public/train.csv")
    print(f"loaded {len(rows)} rows")
    correct = {}
    accs = {}
    for f in FIELDS:
        ens, meta, bounds, names = oof_scores(rows, f)
        c = eval_field(ens, meta, bounds)
        correct[f] = c; accs[f] = c.mean()
        print(f"  {f:18s} top-1 acc = {c.mean():.4f}")
    all_three = np.ones(len(rows), bool)
    for f in FIELDS: all_three &= correct[f]
    fa = np.mean([accs[f] for f in FIELDS])
    comp = 0.97*all_three.mean() + 0.03*fa
    print(f"all-three = {all_three.mean():.4f}  field_acc = {fa:.4f}  SCORE = {comp:.4f}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
