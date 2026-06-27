"""Targeted: does emphasizing the ngram substructure fingerprint help in the
small-molecule (test_low) regime that the real test lives in? Compare feature sets
on the train_high/test_low extrapolation splits (the hard, real-test-like regime).
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

CFGS = {
    "full":         dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1),
    "full+ngdist":  dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1, ngdist=1),
    "full+ng+comp2":dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1, ngdist=1),
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


def rp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)
def cp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)


def build_field(rows, field, cfg):
    Xs, meta, g, ys = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, cfg, None)
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def eval_split(rows, tr_idx, va_idx, cfg):
    fc = {}
    for field in FIELDS:
        Xs, ys, g, meta = build_field(rows, field, cfg)
        Xtr = np.vstack([Xs[i] for i in tr_idx]); ytr = np.concatenate([ys[i] for i in tr_idx])
        gtr = [g[i] for i in tr_idx]
        ms = []
        for s in [42, 1]:
            rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xtr, ytr.astype(int), group=gtr); ms.append(("rk", rk))
            cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
        cor = []
        for i in va_idx:
            X = Xs[i]; toks, truth, isc = meta[i]
            acc = np.zeros(len(toks))
            for kind, m in ms:
                p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                acc += rankdata(p)
            acc[isc.astype(bool)] = -1e9
            cor.append(toks[int(np.argmax(acc))] == truth)
        fc[field] = np.array(cor)
    at = np.ones(len(va_idx), bool)
    for f in FIELDS: at &= fc[f]
    return fc, at.mean()


def main():
    rows = load_rows("dataset/public/train.csv"); n = len(rows)
    V = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in rows])
    size = V[:, 6:15].sum(1)
    pca1 = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(V))[:, 0]
    half = n // 2
    splits = {
        "size_test_low": (np.argsort(size)[half:], np.argsort(size)[:half]),
        "pca_test_low": (np.argsort(pca1)[half:], np.argsort(pca1)[:half]),
    }
    for sname, (tr_idx, va_idx) in splits.items():
        for cname, cfg in CFGS.items():
            fc, at = eval_split(rows, tr_idx, va_idx, cfg)
            print(f"[{sname} | {cname}] src/name/lib={fc['source_token'].mean():.3f}/{fc['name_type_token'].mean():.3f}/{fc['library_token'].mean():.3f} ALL3={at:.3f}", flush=True)


if __name__ == "__main__":
    main()
