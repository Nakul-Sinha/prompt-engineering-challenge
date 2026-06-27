"""Hunt for extra name_type (and source) signal, evaluated on the LB-faithful
pseudo-test proxy. Tests:
  base        : current full features
  +setrel     : set-relative features (candidate vs centroid / vs other candidates)
  +xfield_or  : cross-field denoised query using ORACLE source+library exemplars
  +xfield_pred: cross-field denoised query using predicted source+library (honest)
  +both       : setrel + xfield_pred
Reports per-field pseudo-test accuracy (avg over 2 split fractions).
"""
import sys, json, csv
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
sys.path.insert(0, "research")
from featx import build_row_field, smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)
CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
N_SC = 6; N_AT = 9


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
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, random_state=0, verbosity=-1).fit(X, y.astype(int))
    return m.predict_proba(Xtr)[:, 1]


def rp(s): return dict(objective="lambdarank", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)
def cp(s): return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=s, n_jobs=-1, verbosity=-1)


def cand_vecs(r, field):
    sup = {s["candidate_token"]: smiles_vec(s["smiles_features"]) for s in r["support_cards"] if s["repair_field"] == field}
    toks = [t for t in r[OPTIONS_KEY[field]]]
    return toks, sup


def setrel_feats(cand_vec, all_vecs):
    """candidate relative to the set of candidates."""
    cent = np.mean(all_vecs, axis=0)
    dc = np.abs(cand_vec[:15] - cent[:15]).sum()
    dists = [np.abs(cand_vec[:15] - v[:15]).sum() for v in all_vecs]
    dists_sorted = sorted(dists)
    nearest_other = dists_sorted[1] if len(dists_sorted) > 1 else 0.0
    return [dc, nearest_other, float(np.mean(dists)), float(np.std([np.abs(cand_vec - v).sum() for v in all_vecs]))]


def build_field(rows, field, mode, src_pred=None, lib_pred=None):
    """mode in {base,setrel,xfield_or,xfield_pred,both}."""
    Xs, ys, g, meta = [], [], [], []
    for ri, r in enumerate(rows):
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, None)
        cc = smiles_vec(r["corrupted_card"]["smiles_features"])
        # denoised query
        q = cc.copy()
        if mode in ("xfield_or", "xfield_pred", "both"):
            parts = [cc]
            if field != "source_token":
                st, ssup = cand_vecs(r, "source_token")
                if mode == "xfield_or":
                    key = r["answer"]["source_token"]
                else:
                    key = src_pred[ri] if src_pred else None
                if key in ssup: parts.append(ssup[key])
            if field != "library_token":
                lt, lsup = cand_vecs(r, "library_token")
                if mode == "xfield_or":
                    key = r["answer"]["library_token"]
                else:
                    key = lib_pred[ri] if lib_pred else None
                if key in lsup: parts.append(lsup[key])
            q = np.mean(parts, axis=0)
        ctoks, csup = cand_vecs(r, field)
        all_vecs = [csup[t] for t in toks if t in csup]
        extra_rows = []
        for t in toks:
            cv = csup.get(t, cc)
            ex = []
            if mode in ("xfield_or", "xfield_pred", "both"):
                dq = cv[:15] - q[:15]
                ex += [np.abs(dq).sum(), float(np.sqrt((dq**2).sum()))]
            if mode in ("setrel", "both"):
                ex += setrel_feats(cv, all_vecs) if all_vecs else [0, 0, 0, 0]
            extra_rows.append(ex)
        if extra_rows and len(extra_rows[0]) > 0:
            X = np.hstack([X, np.array(extra_rows, dtype=np.float64)])
        truth = r["answer"][field]
        Xs.append(X); ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
        g.append(len(toks)); meta.append((toks, truth, isc))
    return Xs, ys, g, meta


def predict_field_oof(rows_tr, rows_va, field, mode, src_tr=None, lib_tr=None, src_va=None, lib_va=None):
    Xs, ys, g, meta = build_field(rows_tr, field, mode, src_tr, lib_tr)
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    Xv, yv, gv, mv = build_field(rows_va, field, mode, src_va, lib_va)
    ms = []
    for s in [42, 1]:
        rk = lgb.LGBMRanker(**rp(s)); rk.fit(Xtr, ytr.astype(int), group=g); ms.append(("rk", rk))
        cl = lgb.LGBMClassifier(**cp(s)); cl.fit(Xtr, ytr.astype(int)); ms.append(("cl", cl))
    preds = []; cor = np.zeros(len(rows_va), bool)
    for i in range(len(rows_va)):
        X = Xv[i]; toks, truth, isc = mv[i]
        acc = np.zeros(len(toks))
        for kind, m in ms:
            p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
            acc += rankdata(p)
        acc[isc.astype(bool)] = -1e9
        bi = int(np.argmax(acc)); preds.append(toks[bi]); cor[i] = (toks[bi] == truth)
    return preds, cor


def main():
    tr = load_rows("dataset/public/train.csv"); te = load_rows("dataset/public/test.csv")
    p = domain_p(tr, te); order = np.argsort(-p); n = len(tr)
    for fr in [0.35]:
        nt = int(n*fr); val_idx = set(order[:nt].tolist())
        va = [tr[i] for i in sorted(val_idx)]; trr = [tr[i] for i in range(n) if i not in val_idx]
        # base predictions for source/library (needed for xfield_pred)
        src_tr_pred = {}; lib_tr_pred = {}
        # predict source & library on BOTH trr (via self-fit, ok for feature) and va
        src_pred_tr, _ = predict_field_oof(trr, trr, "source_token", "base")
        lib_pred_tr, _ = predict_field_oof(trr, trr, "library_token", "base")
        src_pred_va, _ = predict_field_oof(trr, va, "source_token", "base")
        lib_pred_va, _ = predict_field_oof(trr, va, "library_token", "base")
        print(f"frac={fr} val={len(va)}", flush=True)
        for field in ["name_type_token", "source_token"]:
            for mode in ["base", "setrel", "xfield_or", "xfield_pred", "both"]:
                if mode in ("xfield_pred", "both"):
                    _, cor = predict_field_oof(trr, va, field, mode, src_pred_tr, lib_pred_tr, src_pred_va, lib_pred_va)
                else:
                    _, cor = predict_field_oof(trr, va, field, mode)
                print(f"  {field:16s} {mode:12s} pseudo-test acc={cor.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
