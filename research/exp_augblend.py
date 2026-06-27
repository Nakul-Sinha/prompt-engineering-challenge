"""Hedge shift-magnitude uncertainty by ENSEMBLING across augmentation strategies.
Train one model with mix3 aug and one with strong2 aug; blend rank-normalised
scores. Evaluate on val families spanning the plausible real shift. Compare to
mix3-alone. Mirrors lgbm rank+clf, seed-bagged.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
SEEDS = [42, 1]
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
MIX3 = [dict(mode="shrink_noise", atom_drop=0.05, count_noise=0.10),
        dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16),
        dict(mode="shrink_noise", atom_drop=0.11, count_noise=0.22)]
STRONG2 = [dict(mode="shrink_noise", atom_drop=0.09, count_noise=0.18)]*2
SEEDSET = [5000, 5777, 6554]
VALS = {"same(.06/.12)": dict(mode="shrink_noise", atom_drop=0.06, count_noise=0.12),
        "real(.08/.16)": dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16),
        "strong(.10/.22)": dict(mode="shrink_noise", atom_drop=0.10, count_noise=0.22)}


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


def rkp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
                        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                        reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
def clp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
                        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                        reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)


def bf(rows, field, seed, params):
    Xs, ys, g = [], [], []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(seed + ri) if seed is not None else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, rng, params)
        truth = r["answer"][field]
        Xs.append((X, toks, truth, isc)); ys.append(np.array([1.0 if t==truth else 0.0 for t in toks])); g.append(len(toks))
    return Xs, ys, g


def models_for(rows, field, tr, strat):
    clean = bf(rows, field, None, None)
    augs = [bf(rows, field, SEEDSET[k], strat[k]) for k in range(len(strat))]
    Xtr, ytr, gtr = [], [], []
    for ri in tr:
        for ds in [clean]+augs:
            Xs, ys, g = ds; Xtr.append(Xs[ri][0]); ytr.append(ys[ri]); gtr.append(g[ri])
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    ms = []
    for s in SEEDS:
        rk = lgb.LGBMRanker(**rkp(s)); rk.fit(Xtr, ytr.astype(int), group=gtr); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**clp(s)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    return ms


def eval_blend(rows, strategies, val_params):
    n = len(rows); kf = KFold(5, shuffle=True, random_state=42)
    fc = {}
    for field in FIELDS:
        valp = bf(rows, field, 9000, val_params)
        correct = np.zeros(n, bool)
        for tr, va in kf.split(np.arange(n)):
            allms = []
            for strat in strategies:
                allms += models_for(rows, field, tr, strat)
            Xs, ys, g = valp
            for ri in va:
                X, toks, truth, isc = Xs[ri]
                acc = np.zeros(len(toks))
                for kind, m in allms:
                    p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                    acc += rankdata(p)
                acc[isc.astype(bool)] = -1e9
                correct[ri] = (toks[int(np.argmax(acc))] == truth)
        fc[field] = correct
    at = np.ones(n, bool)
    for f in FIELDS: at &= fc[f]
    fa = np.mean([fc[f].mean() for f in FIELDS])
    return fc, at.mean(), 0.97*at.mean()+0.03*fa


rows = load_rows("dataset/public/train.csv")
print(f"rows={len(rows)}")
configs = {"mix3": [MIX3], "mix3+strong2": [MIX3, STRONG2]}
for cname, strats in configs.items():
    for vn, vp in VALS.items():
        fc, at, sc = eval_blend(rows, strats, vp)
        print(f"{cname:13s} val={vn:16s} src/name/lib={fc['source_token'].mean():.3f}/{fc['name_type_token'].mean():.3f}/{fc['library_token'].mean():.3f} ALL3={at:.3f} score={sc:.4f}", flush=True)
