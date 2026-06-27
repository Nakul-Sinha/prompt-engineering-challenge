"""Covariate-shift importance weighting (domain adaptation). A domain classifier
distinguishes train vs test corrupted-card structure; training rows that look like
test are upweighted so the model focuses on test-relevant regions.

Validation: designate the train rows MOST test-like (top by domain-classifier
score) as a pseudo-test val set; train on the rest with vs without IW; compare
accuracy on the pseudo-test val. If IW helps the pseudo-test, it should help the
real test. Then also writes an IW test submission.
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
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


def domain_scores(tr, te):
    Xtr = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in tr])
    Xte = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in te])
    X = np.vstack([Xtr, Xte]); y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=-1)
    m.fit(X, y.astype(int))
    p_tr = m.predict_proba(Xtr)[:, 1]
    # AUC-ish separation
    p_te = m.predict_proba(Xte)[:, 1]
    print(f"domain classifier: mean p(test) on train={p_tr.mean():.3f}, on test={p_te.mean():.3f}", flush=True)
    return p_tr


def rp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)
def cp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)


def build_field(rows, field):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        truth = r["answer"][field] if "answer" in r else None
        Xs.append(X); g.append(len(toks))
        ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]) if truth else None)
        meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def expand_weights(roww, g):
    return np.concatenate([np.full(g[i], roww[i]) for i in range(len(g))])


def train_eval(tr_rows, va_rows, roww):
    fc = {}
    for field in FIELDS:
        Xs, ys, g, meta = build_field(tr_rows, field)
        Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
        sw = expand_weights(roww, g) if roww is not None else None
        Xv, yv, gv, mv = build_field(va_rows, field)
        ms = []
        for s in [42, 1]:
            rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xtr, ytr.astype(int), group=g, sample_weight=sw); ms.append(("rk", rk))
            cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xtr, ytr.astype(int), sample_weight=sw); ms.append(("cl", cl))
        cor = []
        for i in range(len(va_rows)):
            X = Xv[i]; toks, truth, isc = mv[i]
            acc = np.zeros(len(toks))
            for kind, m in ms:
                p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                acc += rankdata(p)
            acc[isc.astype(bool)] = -1e9
            cor.append(toks[int(np.argmax(acc))] == truth)
        fc[field] = np.array(cor)
    at = np.ones(len(va_rows), bool)
    for f in FIELDS: at &= fc[f]
    return fc, at.mean()


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    p = domain_scores(tr, te)
    order = np.argsort(-p)  # most test-like first
    n = len(tr); ntop = n // 3
    val_idx = set(order[:ntop].tolist())   # pseudo-test = most test-like train rows
    tr_idx = [i for i in range(n) if i not in val_idx]
    va_rows = [tr[i] for i in sorted(val_idx)]
    tr_rows = [tr[i] for i in tr_idx]
    # importance weights for the training subset, toward the val (test-like) region
    pw = p[tr_idx]
    w = np.clip(pw / (1 - pw + 1e-6), 0.1, 10.0)
    w = w / w.mean()
    print(f"pseudo-test val={len(va_rows)} train-sub={len(tr_rows)}; weight range [{w.min():.2f},{w.max():.2f}]", flush=True)
    fc0, a0 = train_eval(tr_rows, va_rows, None)
    print(f"[no-IW]  pseudo-test src/name/lib={fc0['source_token'].mean():.3f}/{fc0['name_type_token'].mean():.3f}/{fc0['library_token'].mean():.3f} ALL3={a0:.3f}", flush=True)
    fc1, a1 = train_eval(tr_rows, va_rows, w)
    print(f"[IW]     pseudo-test src/name/lib={fc1['source_token'].mean():.3f}/{fc1['name_type_token'].mean():.3f}/{fc1['library_token'].mean():.3f} ALL3={a1:.3f}", flush=True)
    print(f"IW effect on pseudo-test ALL3: {a1-a0:+.3f}", flush=True)


if __name__ == "__main__":
    main()
