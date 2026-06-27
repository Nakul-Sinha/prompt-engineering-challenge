"""Test whether row-grouped CV is optimistic due to structurally-similar QUERY
molecules leaking across folds. Cluster corrupted-card molecules by structure and
run GroupKFold by cluster; compare to plain row KFold. If clustered CV drops toward
the real LB (~0.41), the row-CV was leaking and clustered-CV is the trustworthy proxy.

Uses the current full-feature lgbm rank+clf ensemble, NO augmentation (clean), to
compare apples-to-apples with the v1 model that scored ~0.41 on the LB.
"""
import sys, json, csv, argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
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


def rp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
                       min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
def cp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
                       min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)


def build_field(rows, field):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def run_cv(rows, splits, label):
    n = len(rows)
    fc = {}
    for field in FIELDS:
        Xs, ys, g, meta = build_field(rows, field)
        bounds = np.concatenate([[0], np.cumsum(g)])
        Xall = np.vstack(Xs); yall = np.concatenate(ys)
        correct = np.zeros(n, bool)
        for trn, val in splits:
            trm = np.zeros(len(yall), bool); gtr = []
            for ri in trn: trm[bounds[ri]:bounds[ri+1]] = True; gtr.append(g[ri])
            cols = []
            for s in [42, 1]:
                rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xall[trm], yall[trm].astype(int), group=gtr)
                cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xall[trm], yall[trm].astype(int))
                cols.append(("rk", rk)); cols.append(("cl", cl))
            for ri in val:
                X = Xs[ri]; toks, truth, isc = meta[ri]
                acc = np.zeros(len(toks))
                for kind, m in cols:
                    p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                    acc += rankdata(p)
                acc[isc.astype(bool)] = -1e9
                correct[ri] = (toks[int(np.argmax(acc))] == truth)
        fc[field] = correct
    at = np.ones(n, bool)
    for f in FIELDS: at &= fc[f]
    fa = np.mean([fc[f].mean() for f in FIELDS])
    print(f"[{label}] src/name/lib={fc['source_token'].mean():.3f}/{fc['name_type_token'].mean():.3f}/{fc['library_token'].mean():.3f} "
          f"ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)


def main():
    rows = load_rows("dataset/public/train.csv")
    n = len(rows)
    # cluster corrupted-card molecules
    V = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in rows])
    Vs = StandardScaler().fit_transform(V)
    print(f"rows={n}")
    # plain row KFold (reference, expected ~0.53)
    kf = KFold(5, shuffle=True, random_state=SEED)
    run_cv(rows, list(kf.split(np.arange(n))), "row-KFold")
    # clustered GroupKFold at several granularities
    for ncl in [30, 60, 120, 200]:
        km = KMeans(n_clusters=ncl, random_state=SEED, n_init=4).fit(Vs)
        groups = km.labels_
        gkf = GroupKFold(5)
        run_cv(rows, list(gkf.split(np.arange(n), groups=groups)), f"cluster-GroupKFold(k={ncl})")


if __name__ == "__main__":
    main()
