"""Test the orthogonal 'distance-to-corrupted-token-exemplar' signal on the
LB-faithful pseudo-test proxy. The corruption picks a structurally-nearby decoy, so
candidates near the corrupted token's exemplar are likelier to be the true answer
(verified: true->corr 20.2 vs decoy 24.0 for source; 14.5 vs 17.0 for name).

Adds, per candidate: L1/L2/ngram/cos distance to the corrupted token's exemplar, plus
within-row rank and is-min of that distance. Evaluates source & name on the proxy.
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


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na*nb)) if na > 0 and nb > 0 else 0.0


def build_field(rows, field, use_corr):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        if use_corr:
            sup = {s["candidate_token"]: smiles_vec(s["smiles_features"]) for s in r["support_cards"] if s["repair_field"] == field}
            corr = r["corrupted_card"][field]
            cvec = sup.get(corr)
            extra = []
            d_sa, d_all, d_ng, c_all = [], [], [], []
            for t in toks:
                v = sup.get(t)
                if cvec is None or v is None:
                    d_sa.append(99.0); d_all.append(999.0); d_ng.append(99.0); c_all.append(0.0)
                else:
                    d_sa.append(np.abs(cvec[:15]-v[:15]).sum())
                    d_all.append(np.abs(cvec-v).sum())
                    d_ng.append(np.abs(cvec[15:]-v[15:]).sum())
                    c_all.append(cos(cvec, v))
            d_sa = np.array(d_sa); d_all = np.array(d_all); d_ng = np.array(d_ng); c_all = np.array(c_all)
            rank_sa = d_sa.argsort().argsort().astype(float)/max(len(d_sa)-1, 1)
            ismin = (d_sa == d_sa.min()).astype(float)
            # is this candidate the corrupted token itself? (already isc) - exclude from signal via large dist handled by model
            extra = np.column_stack([d_sa, d_all, d_ng, c_all, rank_sa, ismin])
            X = np.hstack([X, extra])
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def eval_field(trr, va, field, use_corr):
    Xs, ys, g, meta = build_field(trr, field, use_corr)
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    Xv, yv, gv, mv = build_field(va, field, use_corr)
    ms = []
    for s in [42, 1]:
        rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xtr, ytr.astype(int), group=g); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    cor = np.zeros(len(va), bool)
    for i in range(len(va)):
        X = Xv[i]; toks, truth, isc = mv[i]
        acc = np.zeros(len(toks))
        for kind, m in ms:
            p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
            acc += rankdata(p)
        acc[isc.astype(bool)] = -1e9
        cor[i] = (toks[int(np.argmax(acc))] == truth)
    return cor.mean()


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    p = domain_p(tr, te); order = np.argsort(-p); n = len(tr)
    for fr in [0.30, 0.40]:
        nt = int(n*fr); val = set(order[:nt].tolist())
        va = [tr[i] for i in sorted(val)]; trr = [tr[i] for i in range(n) if i not in val]
        print(f"frac={fr} val={len(va)}", flush=True)
        for field in ["source_token", "name_type_token", "library_token"]:
            a0 = eval_field(trr, va, field, False)
            a1 = eval_field(trr, va, field, True)
            print(f"  {field:16s} base={a0:.3f}  +corrdist={a1:.3f}  delta={a1-a0:+.3f}", flush=True)


if __name__ == "__main__":
    main()
