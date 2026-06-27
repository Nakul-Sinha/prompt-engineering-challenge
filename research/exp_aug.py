"""Test shift-robust TRAIN augmentation: add perturbed copies of training rows so
the ranker learns scale-invariant boundaries. Eval on clean-CV and shift-CV."""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, "research")
from featx import build_row_field, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
SEED = 42
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)


def load_rows(path):
    out = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        rec = {"id": row["id"], "corrupted_card": json.loads(row["corrupted_card"]),
               "support_cards": json.loads(row["support_cards"]),
               "source_options": json.loads(row["source_options"]),
               "name_type_options": json.loads(row["name_type_options"]),
               "library_options": json.loads(row["library_options"])}
        if row.get("answer_json"): rec["answer"] = json.loads(row["answer_json"])
        out.append(rec)
    return out


def rk():
    return dict(objective="lambdarank", metric="ndcg", n_estimators=350, learning_rate=0.04,
                num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)


def build_field(rows, field, perturb_seed=None):
    """Return per-row list of (X, y, toks, isc)."""
    out = []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(perturb_seed + ri) if perturb_seed is not None else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, rng)
        truth = r["answer"][field]
        y = np.array([1.0 if t == truth else 0.0 for t in toks])
        out.append((X, y, toks, truth, isc))
    return out


def cv(rows, field, n_aug, perturb_val):
    clean = build_field(rows, field, None)
    # precompute augmented copies (different perturbation seeds)
    augs = [build_field(rows, field, 5000 + 777*k) for k in range(n_aug)]
    valp = build_field(rows, field, 9000) if perturb_val else None
    n = len(rows)
    kf = KFold(5, shuffle=True, random_state=SEED)
    correct = np.zeros(n, bool)
    for tr, va in kf.split(np.arange(n)):
        Xtr, ytr, gtr = [], [], []
        for ri in tr:
            for ds in [clean] + augs:
                X, y, *_ = ds[ri]
                Xtr.append(X); ytr.append(y); gtr.append(len(y))
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        m = lgb.LGBMRanker(**rk()); m.fit(Xtr, ytr.astype(int), group=gtr)
        for ri in va:
            ds = valp if perturb_val else clean
            X, y, toks, truth, isc = ds[ri]
            sc = m.predict(X).copy(); sc[isc.astype(bool)] = -np.inf
            correct[ri] = (toks[int(np.argmax(sc))] == truth)
    return correct


def run(rows, n_aug):
    res = {}
    for mode, pv in [("clean", False), ("shift", True)]:
        cc = {f: cv(rows, f, n_aug, pv) for f in FIELDS}
        at = np.ones(len(rows), bool)
        for f in FIELDS: at &= cc[f]
        fa = np.mean([cc[f].mean() for f in FIELDS])
        res[mode] = (cc, at.mean(), 0.97*at.mean()+0.03*fa)
    return res


rows = load_rows("dataset/public/train.csv")
print(f"rows={len(rows)}")
for n_aug in [0, 1, 2, 3]:
    r = run(rows, n_aug)
    c, s = r["clean"], r["shift"]
    print(f"n_aug={n_aug}  CLEAN src/name/lib={c[0]['source_token'].mean():.3f}/{c[0]['name_type_token'].mean():.3f}/{c[0]['library_token'].mean():.3f} "
          f"all3={c[1]:.3f} score={c[2]:.4f}  ||  SHIFT src/name/lib={s[0]['source_token'].mean():.3f}/{s[0]['name_type_token'].mean():.3f}/{s[0]['library_token'].mean():.3f} "
          f"all3={s[1]:.3f} score={s[2]:.4f}")
