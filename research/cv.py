"""5-fold GroupKFold (group = row) CV for the per-field LightGBM ranker.

Reports per-field top-1 accuracy and the expected competition score
(0.97 * all-three-exact + 0.03 * mean field accuracy).
"""
import sys, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.insert(0, "research")
from features import load_rows, build_row_field, FIELDS, OPTIONS_KEY

SEED = 42


def build_field_dataset(rows, field):
    """Stack all rows for one field. Returns X, y, groups(query lengths), row_ptr."""
    Xs, ys, groups, meta = [], [], [], []
    for ri, r in enumerate(rows):
        opts = r[OPTIONS_KEY[field]]
        X, toks, names, is_corrupt = build_row_field(r["corrupted_card"], r["support_cards"], opts, field)
        truth = r["answer"][field]
        y = np.array([1.0 if t == truth else 0.0 for t in toks])
        Xs.append(X); ys.append(y); groups.append(len(toks))
        meta.append((ri, toks, truth, is_corrupt))
    return np.vstack(Xs), np.concatenate(ys), np.array(groups), meta, names


def lgb_params():
    return dict(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def run_field_cv(rows, field, n_splits=5):
    X, y, groups, meta, names = build_field_dataset(rows, field)
    # group boundaries
    bounds = np.concatenate([[0], np.cumsum(groups)])
    n_rows = len(rows)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(y))
    row_correct = np.zeros(n_rows, dtype=bool)

    for tr_idx, va_idx in kf.split(np.arange(n_rows)):
        tr_mask = np.zeros(len(y), dtype=bool)
        va_mask = np.zeros(len(y), dtype=bool)
        tr_group, va_group = [], []
        for ri in tr_idx:
            tr_mask[bounds[ri]:bounds[ri+1]] = True
            tr_group.append(groups[ri])
        for ri in va_idx:
            va_mask[bounds[ri]:bounds[ri+1]] = True
            va_group.append(groups[ri])
        model = lgb.LGBMRanker(**lgb_params())
        model.fit(X[tr_mask], y[tr_mask].astype(int), group=tr_group)
        oof_pred[va_mask] = model.predict(X[va_mask])

    # evaluate top-1 per row, excluding the corrupted token (answer is never it)
    correct = 0
    for ri, toks, truth, is_corrupt in meta:
        seg = slice(bounds[ri], bounds[ri+1])
        scores = oof_pred[seg].copy()
        scores[is_corrupt.astype(bool)] = -np.inf  # never pick corrupted token
        pred_tok = toks[int(np.argmax(scores))]
        ok = (pred_tok == truth)
        row_correct[ri] = ok
        correct += ok
    acc = correct / n_rows
    return acc, row_correct, names


def main():
    t0 = time.time()
    rows = load_rows("dataset/public/train.csv")
    print(f"loaded {len(rows)} train rows")
    accs = {}
    all_correct = {}
    for f in FIELDS:
        acc, rc, names = run_field_cv(rows, f)
        accs[f] = acc
        all_correct[f] = rc
        print(f"  {f:18s} top-1 acc = {acc:.4f}  (n_feat={len(names)})")
    # combined competition score
    n = len(rows)
    all_three = np.ones(n, dtype=bool)
    for f in FIELDS:
        all_three &= all_correct[f]
    field_acc = np.mean([accs[f] for f in FIELDS])
    comp = 0.97 * all_three.mean() + 0.03 * field_acc
    print(f"all-three-exact = {all_three.mean():.4f}")
    print(f"mean field acc  = {field_acc:.4f}")
    print(f"EXPECTED SCORE  = {comp:.4f}")
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
