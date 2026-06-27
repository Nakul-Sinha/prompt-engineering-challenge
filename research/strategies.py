"""Compare genuinely different strategies on a leakage-free clustered CV
(GroupKFold by query-molecule cluster, the honest proxy), and write a test
submission for each so the best can be checked on the real leaderboard.

Strategies (deliberately spread from 'fit-hard' to 'robust/regularised'):
  full_noaug   : full features, lgbm rank+clf, no augmentation (fit the data)
  robust       : scale-invariant features only (drop raw diffs/abs dist/abs vals),
                 heavier regularisation (anti-overfit bet)
  ngram_boost  : full features + extra ngram-fingerprint similarity emphasis
  blind_aug    : full features + blind random-strength augmentation (no test peek)
  ens_full_rob : average of full_noaug + robust
"""
import sys, json, csv, os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
SEED = 42

CFG_FULL = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
CFG_ROBUST = dict(raw_diff=0, absdist=0, cos=1, absval=0, vendor=1, hint=0, rel=1, comp=1)


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


def rp(s, reg=1.0, leaves=31, mcs=30, ne=350):
    return dict(objective="lambdarank", n_estimators=ne, learning_rate=0.04, num_leaves=leaves,
                min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=reg, random_state=s, n_jobs=-1, verbosity=-1)
def cp(s, reg=1.0, leaves=31, mcs=30, ne=350):
    return dict(objective="binary", n_estimators=ne, learning_rate=0.04, num_leaves=leaves,
                min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=reg, random_state=s, n_jobs=-1, verbosity=-1)


def rand_aug_params(rng):
    return dict(mode="shrink_noise", atom_drop=float(rng.uniform(0.02, 0.14)),
                count_noise=float(rng.uniform(0.06, 0.26)))


def build_field(rows, field, cfg, perturb_seed=None, blind=False):
    Xs, ys, g, meta = [], [], [], []
    for ri, r in enumerate(rows):
        rng = None; params = None
        if perturb_seed is not None:
            rng = np.random.default_rng(perturb_seed + ri)
            params = rand_aug_params(rng) if blind else dict(atom_drop=0.08, count_noise=0.16)
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, cfg, rng, params)
        Xs.append(X); g.append(len(toks))
        if "answer" in r:
            truth = r["answer"][field]
            ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
            meta.append((toks, truth, isc))
        else:
            meta.append((toks, None, isc))
    return Xs, (np.concatenate(ys) if ys else None), g, meta


def train_models(Xtr, ytr, gtr, reg, leaves, mcs):
    ms = []
    for s in [42, 1]:
        rk = lgb.LGBMRanker(**rp(s, reg, leaves, mcs)); rk.fit(Xtr, ytr.astype(int), group=gtr); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**cp(s, reg, leaves, mcs)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    return ms


def assemble(rows, field, cfg, aug_seeds=(), blind=False):
    base = build_field(rows, field, cfg)
    sets = [base]
    for sd in aug_seeds:
        sets.append(build_field(rows, field, cfg, sd, blind))
    Xtr, ytr, gtr = [], [], []
    for Xs, y, g, meta in sets:
        Xtr += Xs; ytr.append(y); gtr += g
    return np.vstack(Xtr), np.concatenate(ytr), gtr


STRATS = {
    "full_noaug":  dict(cfg=CFG_FULL, reg=1.0, leaves=31, mcs=30, aug=()),
    "robust":      dict(cfg=CFG_ROBUST, reg=3.0, leaves=15, mcs=50, aug=()),
    "blind_aug":   dict(cfg=CFG_FULL, reg=1.0, leaves=31, mcs=30, aug=(5000, 5777, 6554), blind=True),
}


def predict_and_cv(rows_tr, rows_te, groups):
    n = len(rows_tr)
    gkf = GroupKFold(5)
    splits = list(gkf.split(np.arange(n), groups=groups))
    results = {}
    test_scores = {}  # strat -> field -> per-row rank-normalized test scores
    for sname, sp in STRATS.items():
        cfg = sp["cfg"]
        fc = {}; tscore = {}
        for field in FIELDS:
            # CV
            Xs, ys, g, meta = build_field(rows_tr, field, cfg)
            bounds = np.concatenate([[0], np.cumsum(g)])
            correct = np.zeros(n, bool)
            for trn, val in splits:
                # assemble training (with aug) restricted to trn rows
                trrows = [rows_tr[i] for i in trn]
                Xtr, ytr, gtr = assemble(trrows, field, cfg, sp.get("aug", ()), sp.get("blind", False))
                ms = train_models(Xtr, ytr, gtr, sp["reg"], sp["leaves"], sp["mcs"])
                for ri in val:
                    X = Xs[ri]; toks, truth, isc = meta[ri]
                    acc = np.zeros(len(toks))
                    for kind, m in ms:
                        p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                        acc += rankdata(p)
                    acc[isc.astype(bool)] = -1e9
                    correct[ri] = (toks[int(np.argmax(acc))] == truth)
            fc[field] = correct
            # full-train test prediction
            Xtr, ytr, gtr = assemble(rows_tr, field, cfg, sp.get("aug", ()), sp.get("blind", False))
            ms = train_models(Xtr, ytr, gtr, sp["reg"], sp["leaves"], sp["mcs"])
            Xs_te, _, g_te, meta_te = build_field(rows_te, field, cfg)
            bte = np.concatenate([[0], np.cumsum(g_te)])
            ts = []
            for ri in range(len(rows_te)):
                X = Xs_te[ri]; toks, _, isc = meta_te[ri]
                acc = np.zeros(len(toks))
                for kind, m in ms:
                    p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                    acc += rankdata(p)
                ts.append((toks, isc, acc / (len(ms)*len(toks))))
            tscore[field] = ts
        at = np.ones(n, bool)
        for f in FIELDS: at &= fc[f]
        fa = np.mean([fc[f].mean() for f in FIELDS])
        results[sname] = (fc, at.mean(), 0.97*at.mean()+0.03*fa)
        test_scores[sname] = tscore
        print(f"[{sname}] CV src/name/lib={fc['source_token'].mean():.3f}/{fc['name_type_token'].mean():.3f}/{fc['library_token'].mean():.3f} "
              f"ALL3={at.mean():.3f} score={results[sname][2]:.4f}", flush=True)
    return results, test_scores


def write_sub(rows_te, tscore, path):
    os.makedirs("working", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["id", "answer_json"])
        for ri, r in enumerate(rows_te):
            ans = {}
            for f in FIELDS:
                toks, isc, acc = tscore[f][ri]
                a = acc.copy(); a[isc.astype(bool)] = -1e9
                ans[f] = toks[int(np.argmax(a))]
            w.writerow([r["id"], json.dumps(ans, separators=(",", ":"))])


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    V = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in tr])
    groups = KMeans(n_clusters=120, random_state=SEED, n_init=4).fit_predict(StandardScaler().fit_transform(V))
    print(f"train={len(tr)} test={len(te)}")
    results, tscore = predict_and_cv(tr, te, groups)
    for sname in STRATS:
        write_sub(te, tscore[sname], f"working/submission_{sname}.csv")
        print(f"wrote working/submission_{sname}.csv", flush=True)
    # ensemble full_noaug + robust (blend test scores)
    print("ensemble full_noaug+robust:", flush=True)
    os.makedirs("working", exist_ok=True)
    with open("working/submission_ens_full_robust.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["id", "answer_json"])
        for ri, r in enumerate(te):
            ans = {}
            for f in FIELDS:
                toks, isc, a1 = tscore["full_noaug"][f][ri]
                _, _, a2 = tscore["robust"][f][ri]
                a = (a1 + a2).copy(); a[isc.astype(bool)] = -1e9
                ans[f] = toks[int(np.argmax(a))]
            w.writerow([r["id"], json.dumps(ans, separators=(",", ":"))])
    print("wrote working/submission_ens_full_robust.csv", flush=True)


if __name__ == "__main__":
    main()
