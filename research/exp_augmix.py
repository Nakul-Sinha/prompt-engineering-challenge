"""Compare augmentation STRATEGIES, evaluating on val families spanning the real
shift (real lib median 15.0 ~ between 'same' 14.3 and 'strong' 15.6).

Strategies (each adds 2 perturbed copies per row):
  same2   : (shrink .06/.12) x2          [current]
  strong2 : (shrink .09/.18) x2
  mix     : (shrink .06/.12) + (shrink .09/.18)
  mix3    : .05/.10 + .08/.16 + .11/.22   (3 copies, broad)
Val families: same(.06/.12), strong(.10/.22), grow(.08/.15).  We care most about
the val closest to the real 15.0.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, "research")
from featx import build_row_field, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
SEED = 42
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)

STRATS = {
    "same2":   [dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12)]*2,
    "strong2": [dict(mode="shrink_noise", atom_drop=0.09, count_noise=0.18)]*2,
    "mix":     [dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12),
                dict(mode="shrink_noise", atom_drop=0.09, count_noise=0.18)],
    "mix3":    [dict(mode="shrink_noise", atom_drop=0.05, count_noise=0.10),
                dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16),
                dict(mode="shrink_noise", atom_drop=0.11, count_noise=0.22)],
}
VALS = {
    "same(.06/.12)":  dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12),
    "real~(.08/.16)": dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16),
    "strong(.10/.22)":dict(mode="shrink_noise", atom_drop=0.10, count_noise=0.22),
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


def cv(rows, field, strat_params, val_params):
    clean = build_field(rows, field, None)
    augs = [build_field(rows, field, 5000+777*k, p) for k, p in enumerate(strat_params)]
    valp = build_field(rows, field, 9000, val_params)
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
for sname, sp in STRATS.items():
    line = f"{sname:9s}"
    for vname, vp in VALS.items():
        cc = {f: cv(rows, f, sp, vp) for f in FIELDS}
        at = np.ones(len(rows), bool)
        for f in FIELDS: at &= cc[f]
        fa = np.mean([cc[f].mean() for f in FIELDS])
        line += f" | {vname}: all3={at.mean():.3f} sc={0.97*at.mean()+0.03*fa:.4f}"
    print(line, flush=True)
