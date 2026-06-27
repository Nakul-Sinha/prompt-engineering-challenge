"""Simulate the OOD gap WITHIN train via extrapolation splits: train on one region
of chemical space, evaluate on a structurally distant region. If this reproduces the
~0.41 LB (vs 0.53 random CV), it captures the gap mechanism, and whichever model
degrades LEAST under extrapolation is the better real-test bet (even if its random
CV is lower).

Splits:
  size    : train on smaller molecules, test on larger (and vice versa), by heavy-atom count
  pca     : split by sign of the 1st PCA component of structure (two halves)
Compares full vs robust feature sets.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)

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


def rp(s, reg, leaves, mcs): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04,
    num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=reg, random_state=s, n_jobs=-1, verbosity=-1)
def cp(s, reg, leaves, mcs): return dict(objective="binary", n_estimators=350, learning_rate=0.04,
    num_leaves=leaves, min_child_samples=mcs, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=reg, random_state=s, n_jobs=-1, verbosity=-1)


def build_field(rows, field, cfg):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, cfg, None)
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def eval_split(rows, tr_idx, va_idx, cfg, reg, leaves, mcs):
    fc = {}
    for field in FIELDS:
        Xs, ys, g, meta = build_field(rows, field, cfg)
        Xtr = np.vstack([Xs[i] for i in tr_idx]); ytr = np.concatenate([ys[i] for i in tr_idx])
        gtr = [g[i] for i in tr_idx]
        ms = []
        for s in [42, 1]:
            rk = lgb.LGBMRanker(**rp(s, reg, leaves, mcs)); rk.fit(Xtr, ytr.astype(int), group=gtr); ms.append(("rk", rk))
            cl = lgb.LGBMClassifier(**cp(s, reg, leaves, mcs)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
        correct = []
        for i in va_idx:
            X = Xs[i]; toks, truth, isc = meta[i]
            acc = np.zeros(len(toks))
            for kind, m in ms:
                p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                acc += rankdata(p)
            acc[isc.astype(bool)] = -1e9
            correct.append(toks[int(np.argmax(acc))] == truth)
        fc[field] = np.array(correct)
    at = np.ones(len(va_idx), bool)
    for f in FIELDS: at &= fc[f]
    fa = np.mean([fc[f].mean() for f in FIELDS])
    return fc, at.mean(), 0.97*at.mean()+0.03*fa


def main():
    rows = load_rows("dataset/public/train.csv")
    n = len(rows)
    V = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in rows])
    size = V[:, 6:15].sum(1)  # heavy-atom count
    pca1 = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(V))[:, 0]
    axes = {
        "size_lo->hi": np.argsort(size),
        "pca_lo->hi": np.argsort(pca1),
    }
    models = {"full": (CFG_FULL, 1.0, 31, 30), "robust": (CFG_ROBUST, 3.0, 15, 50)}
    for aname, order in axes.items():
        half = n // 2
        for direction, (tr_idx, va_idx) in [
            ("train_low/test_high", (order[:half], order[half:])),
            ("train_high/test_low", (order[half:], order[:half])),
        ]:
            for mname, (cfg, reg, leaves, mcs) in models.items():
                fc, at, sc = eval_split(rows, tr_idx, va_idx, cfg, reg, leaves, mcs)
                print(f"[{aname} | {direction} | {mname}] "
                      f"src/name/lib={fc['source_token'].mean():.3f}/{fc['name_type_token'].mean():.3f}/{fc['library_token'].mean():.3f} "
                      f"ALL3={at:.3f} score={sc:.4f}", flush=True)


if __name__ == "__main__":
    main()
