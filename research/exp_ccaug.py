"""Does augmenting with a perturbed CORRUPTED CARD (test molecules are smaller)
add robustness? Compare training augmentation that perturbs candidates-only vs
candidates+cc, each evaluated on val that perturbs candidates-only vs candidates+cc.
Real-shift params (.08/.16). Mirrors solution.py's lgbm rank+clf ensemble.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata
sys.path.insert(0, ".")
import solution as S
csv.field_size_limit(10**8)

AUG = [dict(atom_drop=0.05, count_noise=0.10), dict(atom_drop=0.08, count_noise=0.16),
       dict(atom_drop=0.11, count_noise=0.22)]
AUG_SEEDS = [5000, 5777, 6554]
VALP = dict(atom_drop=0.08, count_noise=0.16)


def perturb_cc_card(card, rng, params):
    v = S.perturb_vec(S.smiles_vec(card["smiles_features"]), rng, **params)
    sf = dict(card["smiles_features"])
    for i, k in enumerate(S.SCALARS):
        sf[k] = int(v[i])
    ac = dict(sf["atom_counts"])
    for i, a in enumerate(S.ATOMS):
        ac[a] = int(v[6+i])
    sf["atom_counts"] = ac
    sf["ngram_buckets"] = [int(x) for x in v[15:]]
    return {**card, "smiles_features": sf}


def build_field(rows, field, seed, params, perturb_cc):
    Xs, ys, g = [], [], []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(seed + ri) if seed is not None else None
        cc = r["corrupted_card"]
        if perturb_cc and rng is not None:
            cc = perturb_cc_card(cc, np.random.default_rng((seed or 0) + 100000 + ri), params)
        X, toks, isc = S.build_row_field(cc, r["support_cards"], r[S.OPTIONS_KEY[field]], field, rng, params)
        truth = r["answer"][field]
        Xs.append((X, toks, truth, isc))
        y = np.array([1.0 if t == truth else 0.0 for t in toks]); ys.append(y); g.append(len(toks))
    return Xs, ys, g


def cv(rows, field, train_cc, val_cc):
    clean = build_field(rows, field, None, None, False)
    augs = [build_field(rows, field, AUG_SEEDS[k], AUG[k], train_cc) for k in range(len(AUG))]
    valp = build_field(rows, field, 9000, VALP, val_cc)
    n = len(rows); kf = KFold(5, shuffle=True, random_state=42)
    correct = np.zeros(n, bool)
    for tr, va in kf.split(np.arange(n)):
        Xtr, ytr, gtr = [], [], []
        for ri in tr:
            for ds in [clean] + augs:
                Xs, ys, g = ds
                Xtr.append(Xs[ri][0]); ytr.append(ys[ri]); gtr.append(g[ri])
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        models = []
        for seed in S.SEEDS:
            rk = lgb.LGBMRanker(**S.ranker_params(seed)); rk.fit(Xtr, ytr.astype(int), group=gtr)
            cl = lgb.LGBMClassifier(**S.clf_params(seed)); cl.fit(Xtr, ytr.astype(int))
            models += [("rk", rk), ("cl", cl)]
        Xs, ys, g = valp
        for ri in va:
            X, toks, truth, isc = Xs[ri]
            acc = np.zeros(len(toks))
            for kind, m in models:
                p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                acc += rankdata(p)
            acc[isc.astype(bool)] = -1e9
            correct[ri] = (toks[int(np.argmax(acc))] == truth)
    return correct


rows = S.load_rows("dataset/public/train.csv")
print(f"rows={len(rows)} seeds={S.SEEDS}")
for train_cc in [False, True]:
    for val_cc in [False, True]:
        cc = {f: cv(rows, f, train_cc, val_cc) for f in S.FIELDS}
        at = np.ones(len(rows), bool)
        for f in S.FIELDS: at &= cc[f]
        fa = np.mean([cc[f].mean() for f in S.FIELDS])
        print(f"train_cc={train_cc!s:5s} val_cc={val_cc!s:5s} "
              f"src/name/lib={cc['source_token'].mean():.3f}/{cc['name_type_token'].mean():.3f}/{cc['library_token'].mean():.3f} "
              f"ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
