"""Optimize against the LB-predictive pseudo-test proxy (domain-classifier split:
train on least-test-like train rows, evaluate on most-test-like). Sweep model
regularization + feature subsets PER FIELD and report pseudo-test accuracy
(averaged over a few split fractions for stability). Pick the best per-field config.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)

CFG_FULL = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
CFG_NOABS = dict(raw_diff=1, absdist=1, cos=1, absval=0, vendor=1, hint=1, rel=1, comp=1)
CFG_ROBUST = dict(raw_diff=0, absdist=0, cos=1, absval=0, vendor=1, hint=0, rel=1, comp=1)

CONFIGS = {
    "full_l31_r1":     (CFG_FULL, dict(leaves=31, mcs=30, reg=1.0, ne=350)),
    "full_l15_r3":     (CFG_FULL, dict(leaves=15, mcs=50, reg=3.0, ne=300)),
    "full_l7_r5":      (CFG_FULL, dict(leaves=7, mcs=80, reg=5.0, ne=250)),
    "full_l4_r8":      (CFG_FULL, dict(leaves=4, mcs=100, reg=8.0, ne=250)),
    "noabs_l15_r3":    (CFG_NOABS, dict(leaves=15, mcs=50, reg=3.0, ne=300)),
    "noabs_l7_r5":     (CFG_NOABS, dict(leaves=7, mcs=80, reg=5.0, ne=250)),
    "robust_l15_r3":   (CFG_ROBUST, dict(leaves=15, mcs=50, reg=3.0, ne=300)),
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


def domain_p(tr, te):
    Xtr = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in tr])
    Xte = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in te])
    X = np.vstack([Xtr, Xte]); y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                           n_jobs=-1, verbosity=-1)
    m.fit(X, y.astype(int))
    return m.predict_proba(Xtr)[:, 1]


def rp(s, p): return dict(objective="lambdarank", n_estimators=p["ne"], learning_rate=0.04,
    num_leaves=p["leaves"], min_child_samples=p["mcs"], subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=p["reg"], random_state=s, n_jobs=-1, verbosity=-1)
def cp(s, p): return dict(objective="binary", n_estimators=p["ne"], learning_rate=0.04,
    num_leaves=p["leaves"], min_child_samples=p["mcs"], subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=p["reg"], random_state=s, n_jobs=-1, verbosity=-1)


def build_field(rows, field, cfg):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, cfg, None)
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def field_acc(tr_rows, va_rows, field, cfg, params):
    Xs, ys, g, meta = build_field(tr_rows, field, cfg)
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    Xv, yv, gv, mv = build_field(va_rows, field, cfg)
    ms = []
    for s in [42, 1]:
        rk = lgb.LGBMRanker(**rp(s, params)); rk.fit(Xtr, ytr.astype(int), group=g); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**cp(s, params)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    cor = np.zeros(len(va_rows), bool)
    for i in range(len(va_rows)):
        X = Xv[i]; toks, truth, isc = mv[i]
        acc = np.zeros(len(toks))
        for kind, m in ms:
            p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
            acc += rankdata(p)
        acc[isc.astype(bool)] = -1e9
        cor[i] = (toks[int(np.argmax(acc))] == truth)
    return cor


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    p = domain_p(tr, te)
    order = np.argsort(-p)
    n = len(tr)
    fracs = [0.30, 0.40]  # average over two pseudo-test sizes for stability
    # per field, per config -> mean accuracy across fracs
    best = {}
    for field in FIELDS:
        print(f"=== {field} ===", flush=True)
        scores = {}
        for cname, (cfg, params) in CONFIGS.items():
            accs = []
            for fr in fracs:
                nt = int(n * fr)
                val_idx = set(order[:nt].tolist())
                va = [tr[i] for i in sorted(val_idx)]
                trr = [tr[i] for i in range(n) if i not in val_idx]
                cor = field_acc(trr, va, field, cfg, params)
                accs.append(cor.mean())
            m = float(np.mean(accs))
            scores[cname] = m
            print(f"  {cname:16s} pseudo-test acc={m:.3f}  (fracs {[round(a,3) for a in accs]})", flush=True)
        bestc = max(scores, key=scores.get)
        best[field] = (bestc, scores[bestc])
        print(f"  -> BEST {field}: {bestc} = {scores[bestc]:.3f}", flush=True)
    print("SUMMARY best per-field:", {f: best[f] for f in FIELDS}, flush=True)
    # estimate combined (product as independence proxy)
    prod = 1.0
    for f in FIELDS: prod *= best[f][1]
    print(f"product-of-best per-field (rough all3 estimate) = {prod:.3f}", flush=True)


if __name__ == "__main__":
    main()
