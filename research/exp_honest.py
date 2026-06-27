"""Honest robustness check: train with perturbation family A, evaluate on a
DIFFERENT family B. If augmentation only helps when val==train perturbation, the
gain is memorisation; if it helps on unseen B too, it is genuine robustness.

Also calibrates which perturbation best reproduces the real library shift
(train med dist 12.8 -> test 15.0).
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
SEED = 42
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)

TRAIN_P = dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12)
VAL_FAMILIES = {
    "none": None,
    "same(shrink.06/.12)": dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12),
    "strong(shrink.10/.22)": dict(mode="shrink_noise", atom_drop=0.10, count_noise=0.22),
    "grow(.08/.15)": dict(mode="grow_noise", atom_drop=0.08, count_noise=0.15),
    "scalejit(.18)": dict(mode="scale_jitter", count_noise=0.18),
}


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
    return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
                min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)


def build_field(rows, field, seed=None, params=None):
    out = []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(seed + ri) if seed is not None else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, rng, params)
        truth = r["answer"][field]
        y = np.array([1.0 if t == truth else 0.0 for t in toks])
        out.append((X, y, toks, truth, isc))
    return out


def lib_med_dist(rows, seed, params):
    """median candidate L1(scalar+atom) distance for library, for calibration."""
    meds = []
    for ri, r in enumerate(rows):
        cc = smiles_vec(r["corrupted_card"]["smiles_features"])
        sup = {s["candidate_token"]: s for s in r["support_cards"] if s["repair_field"] == "library_token"}
        rng = np.random.default_rng(seed + ri) if seed is not None else None
        from featx import perturb_smiles_vec
        ds = []
        for tok in r["library_options"]:
            if tok == r["corrupted_card"]["library_token"]:
                continue
            s = sup.get(tok)
            if not s:
                continue
            v = smiles_vec(s["smiles_features"])
            if rng is not None and params:
                v = perturb_smiles_vec(v, rng, **params)
            ds.append(np.abs(cc[:15]-v[:15]).sum())
        meds.append(np.median(ds))
    return np.mean(meds)


def cv(rows, field, n_aug, val_params):
    clean = build_field(rows, field, None)
    augs = [build_field(rows, field, 5000+777*k, TRAIN_P) for k in range(n_aug)]
    valp = build_field(rows, field, 9000, val_params) if val_params else clean
    n = len(rows)
    kf = KFold(5, shuffle=True, random_state=SEED)
    correct = np.zeros(n, bool)
    for tr, va in kf.split(np.arange(n)):
        Xtr, ytr, gtr = [], [], []
        for ri in tr:
            for ds in [clean]+augs:
                X, y, *_ = ds[ri]; Xtr.append(X); ytr.append(y); gtr.append(len(y))
        m = lgb.LGBMRanker(**rk()); m.fit(np.vstack(Xtr), np.concatenate(ytr).astype(int), group=gtr)
        for ri in va:
            X, y, toks, truth, isc = valp[ri]
            sc = m.predict(X).copy(); sc[isc.astype(bool)] = -1e9
            correct[ri] = (toks[int(np.argmax(sc))] == truth)
    return correct


rows = load_rows("dataset/public/train.csv")
print(f"rows={len(rows)}")
# calibration: which val family matches real test lib median 15.0 (train clean ~12.8)
print("CALIBRATION lib median dist (real: train 12.8 -> test 15.0):")
print(f"  clean = {lib_med_dist(rows, None, None):.2f}")
for nm, p in VAL_FAMILIES.items():
    if p: print(f"  {nm} = {lib_med_dist(rows, 9000, p):.2f}")
print()
for n_aug in [0, 2]:
    print(f"--- n_aug={n_aug} ---")
    for nm, p in VAL_FAMILIES.items():
        cc = {f: cv(rows, f, n_aug, p) for f in FIELDS}
        at = np.ones(len(rows), bool)
        for f in FIELDS: at &= cc[f]
        fa = np.mean([cc[f].mean() for f in FIELDS])
        print(f"  val={nm:22s} src/name/lib={cc['source_token'].mean():.3f}/{cc['name_type_token'].mean():.3f}/{cc['library_token'].mean():.3f} "
              f"all3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
