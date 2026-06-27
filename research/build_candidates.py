"""Build candidate submissions with set-relative features (name) and report honest
all-three on the LB-faithful pseudo-test proxy. Writes test submissions to pre-check.

Configs (per-field feature mode): base | setrel
  baseline    : all base
  setrel_name : name=setrel, source/library=base   (expected best)
  setrel_all  : all setrel
"""
import sys, json, csv, os
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
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


def domain_p(tr, te):
    Xtr = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in tr])
    Xte = np.vstack([smiles_vec(r["corrupted_card"]["smiles_features"]) for r in te])
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, random_state=0, verbosity=-1)
    m.fit(np.vstack([Xtr, Xte]), np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(int))
    return m.predict_proba(Xtr)[:, 1]


def rp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)
def cp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)


def setrel_feats(cv, all_vecs):
    cent = np.mean(all_vecs, axis=0)
    dc = np.abs(cv[:15] - cent[:15]).sum()
    dists = np.array([np.abs(cv[:15] - v[:15]).sum() for v in all_vecs])
    ds = np.sort(dists)
    nearest_other = ds[1] if len(ds) > 1 else 0.0
    dall = np.array([np.abs(cv - v).sum() for v in all_vecs])
    return [dc, nearest_other, float(dists.mean()), float(dall.std())]


def build_field(rows, field, mode, labels=True):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        if mode == "setrel":
            sup = {s["candidate_token"]: smiles_vec(s["smiles_features"]) for s in r["support_cards"] if s["repair_field"] == field}
            all_vecs = [sup[t] for t in toks if t in sup]
            cc = smiles_vec(r["corrupted_card"]["smiles_features"])
            extra = []
            for t in toks:
                cv = sup.get(t, cc)
                extra.append(setrel_feats(cv, all_vecs) if all_vecs else [0, 0, 0, 0])
            X = np.hstack([X, np.array(extra, dtype=np.float64)])
        Xs.append(X); g.append(len(toks))
        if labels:
            truth = r["answer"][field]
            ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
            meta.append((toks, truth, isc))
        else:
            meta.append((toks, None, isc))
    return Xs, (np.concatenate(ys) if labels else None), g, meta


def train_models(Xtr, ytr, g):
    ms = []
    for s in SEEDS:
        rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xtr, ytr.astype(int), group=g); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    return ms


def predict(ms, Xs, meta):
    preds = []; cors = []
    for i in range(len(Xs)):
        X = Xs[i]; toks, truth, isc = meta[i]
        acc = np.zeros(len(toks))
        for kind, m in ms:
            p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
            acc += rankdata(p)
        acc[isc.astype(bool)] = -1e9
        bi = int(np.argmax(acc)); preds.append(toks[bi]); cors.append(truth is not None and toks[bi] == truth)
    return preds, np.array(cors)


CONFIGS = {
    "baseline":    {"source_token": "base", "name_type_token": "base", "library_token": "base"},
    "setrel_name": {"source_token": "base", "name_type_token": "setrel", "library_token": "base"},
    "setrel_all":  {"source_token": "setrel", "name_type_token": "setrel", "library_token": "setrel"},
}


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    p = domain_p(tr, te); order = np.argsort(-p); n = len(tr)
    fr = 0.35; nt = int(n*fr); val = set(order[:nt].tolist())
    va = [tr[i] for i in sorted(val)]; trr = [tr[i] for i in range(n) if i not in val]
    for cname, modes in CONFIGS.items():
        # proxy eval
        cors = {}
        for field in FIELDS:
            Xs, ys, g, meta = build_field(trr, field, modes[field])
            ms = train_models(np.vstack(Xs), ys, g)
            Xv, _, gv, mv = build_field(va, field, modes[field])
            _, cor = predict(ms, Xv, mv)
            cors[field] = cor
        at = np.ones(len(va), bool)
        for f in FIELDS: at &= cors[f]
        fa = np.mean([cors[f].mean() for f in FIELDS])
        print(f"[{cname}] PROXY src/name/lib={cors['source_token'].mean():.3f}/{cors['name_type_token'].mean():.3f}/{cors['library_token'].mean():.3f} "
              f"ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
        # full-train test submission
        os.makedirs("working", exist_ok=True)
        fp = {}
        for field in FIELDS:
            Xs, ys, g, meta = build_field(tr, field, modes[field])
            ms = train_models(np.vstack(Xs), ys, g)
            Xte, _, gte, mte = build_field(te, field, modes[field], labels=False)
            preds, _ = predict(ms, Xte, mte)
            fp[field] = {te[i]["id"]: preds[i] for i in range(len(te))}
        with open(f"working/submission_{cname}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["id", "answer_json"])
            for r in te:
                rid = r["id"]
                ans = {f: fp[f][rid] for f in FIELDS}
                w.writerow([rid, json.dumps(ans, separators=(",", ":"))])
        print(f"wrote working/submission_{cname}.csv", flush=True)


if __name__ == "__main__":
    main()
