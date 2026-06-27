"""Diagnostic: a deliberately SIMPLER, scale-invariant, regularised model to test
the overfit hypothesis (current full model gets ~0.405 on the real LB despite 0.53
CV). Drops scale-dependent feature blocks (raw signed/abs diffs, absolute values,
absolute block distances); keeps only scale-invariant signals: within-row relational
(rank/z/ratio/is-min of distances), cosine sims, atom-composition fractions, vendor
match. Heavy regularisation, no augmentation. Writes working/submission_robust.csv.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)

# scale-invariant only: drop raw_diff, absdist, absval
CFG = dict(raw_diff=0, absdist=0, cos=1, absval=0, vendor=1, hint=0, rel=1, comp=1)
SEEDS = [42, 1, 7]


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


def rparams(seed):
    return dict(objective="lambdarank", n_estimators=250, learning_rate=0.04, num_leaves=15,
                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=3.0, random_state=seed, n_jobs=-1, verbosity=-1)


def cparams(seed):
    return dict(objective="binary", n_estimators=250, learning_rate=0.04, num_leaves=15,
                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=3.0, random_state=seed, n_jobs=-1, verbosity=-1)


def build(rows, field, labels):
    Xs, ys, g, meta = [], [], [], []
    for ri, r in enumerate(rows):
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        Xs.append(X); g.append(len(toks))
        if labels:
            truth = r["answer"][field]
            ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        meta.append((ri, toks, isc))
    return Xs, (np.concatenate(ys) if labels else None), g, meta


def predict_field(tr, te, field):
    Xs, y, g, _ = build(tr, field, True)
    Xtr = np.vstack(Xs)
    Xs_te, _, g_te, meta = build(te, field, False)
    bounds = np.concatenate([[0], np.cumsum(g_te)]); Xte = np.vstack(Xs_te)
    cols = []
    for s in SEEDS:
        rk = lgb.LGBMRanker(**rparams(s)); rk.fit(Xtr, y.astype(int), group=g)
        cols.append(rk.predict(Xte))
        cl = lgb.LGBMClassifier(**cparams(s)); cl.fit(Xtr, y.astype(int))
        cols.append(cl.predict_proba(Xte)[:, 1])
    ens = np.zeros(len(Xte))
    for ri in range(len(te)):
        sl = slice(bounds[ri], bounds[ri+1]); n = bounds[ri+1]-bounds[ri]
        acc = np.zeros(n)
        for c in cols: acc += rankdata(c[sl])
        ens[sl] = acc/(len(cols)*n)
    preds = {}
    for ri, toks, isc in meta:
        sc = ens[bounds[ri]:bounds[ri+1]].copy(); sc[isc.astype(bool)] = -np.inf
        preds[te[ri]["id"]] = toks[int(np.argmax(sc))]
    return preds


tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
print(f"train={len(tr)} test={len(te)} robust scale-invariant model")
fp = {f: predict_field(tr, te, f) for f in FIELDS}
import os
os.makedirs("working", exist_ok=True)
with open("working/submission_robust.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["id", "answer_json"])
    for r in te:
        rid = r["id"]
        ans = {"source_token": fp["source_token"][rid], "name_type_token": fp["name_type_token"][rid],
               "library_token": fp["library_token"][rid]}
        w.writerow([rid, json.dumps(ans, separators=(",", ":"))])
print("wrote working/submission_robust.csv")
# how different from current submission?
try:
    cur = {r["id"]: json.loads(r["answer_json"]) for r in csv.DictReader(open("working/submission.csv"))}
    diff = sum(any(fp[f][i] != cur[i][f] for f in FIELDS) for i in cur)
    print(f"rows differing from current submission: {diff}/{len(cur)}")
except Exception as e:
    print("compare skipped:", e)
