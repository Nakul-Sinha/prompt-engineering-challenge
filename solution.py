import csv
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata

csv.field_size_limit(10**8)

ROOT = Path("./dataset/public")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

SCALARS = ["aromatic_count", "branch_count", "charge_count", "length_bin",
           "ring_digit_count", "stereo_count"]
ATOMS = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
NGRAM_N = 12
FIELDS = ["source_token", "name_type_token", "library_token"]
OPTIONS_KEY = {"source_token": "source_options",
               "name_type_token": "name_type_options",
               "library_token": "library_options"}
N_SCALAR, N_ATOM = len(SCALARS), len(ATOMS)
SEEDS = [42, 1, 7]


def smiles_vec(s):
    v = [s[k] for k in SCALARS]
    v += [s["atom_counts"][k] for k in ATOMS]
    v += list(s["ngram_buckets"])
    return np.asarray(v, dtype=np.float64)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _pair(cc_vec, cand_vec, vmatch, vmiss, hint):
    diff = cand_vec - cc_vec
    ad = np.abs(diff)
    sc_a, at_a, ng_a = ad[:N_SCALAR], ad[N_SCALAR:N_SCALAR+N_ATOM], ad[N_SCALAR+N_ATOM:]
    cc_at, cd_at = cc_vec[N_SCALAR:N_SCALAR+N_ATOM], cand_vec[N_SCALAR:N_SCALAR+N_ATOM]
    cc_ng, cd_ng = cc_vec[N_SCALAR+N_ATOM:], cand_vec[N_SCALAR+N_ATOM:]
    f = []
    for i in range(len(cc_vec)):
        f += [diff[i], ad[i]]
    f += [sc_a.sum(), at_a.sum(), ng_a.sum(), sc_a.sum()+at_a.sum(), ad.sum(),
          float(np.sqrt((sc_a**2).sum()+(at_a**2).sum())), float(np.sqrt((ng_a**2).sum()))]
    f += [_cos(cc_at, cd_at), _cos(cc_ng, cd_ng), _cos(cc_vec[:N_SCALAR], cand_vec[:N_SCALAR]),
          _cos(cc_vec, cand_vec), float(at_a.sum() == 0), float(sc_a.sum() == 0),
          float(sc_a.sum() == 0 and at_a.sum() == 0)]
    for i in range(N_SCALAR):
        f += [cand_vec[i], cc_vec[i]]
    f += [cd_at.sum(), cc_at.sum(), cd_ng.sum(), cc_ng.sum()]
    f += [float(vmatch), float(vmiss), float(hint)]
    cc_tot, cd_tot = max(cc_at.sum(), 1), max(cd_at.sum(), 1)
    fd = cd_at/cd_tot - cc_at/cc_tot
    for i in range(N_ATOM):
        f += [fd[i], abs(fd[i])]
    f += [np.abs(fd).sum(), float(np.sqrt((fd**2).sum())),
          float(cd_at[1]/cd_tot), float(cc_at[1]/cc_tot)]
    return f, (sc_a.sum()+at_a.sum(), ad.sum(), ng_a.sum(), _cos(cc_vec, cand_vec), float(hint))


def _setrel(cand_vec, all_vecs):
    cent = np.mean(all_vecs, axis=0)
    dc = np.abs(cand_vec[:15] - cent[:15]).sum()
    dists = np.array([np.abs(cand_vec[:15] - v[:15]).sum() for v in all_vecs])
    ds = np.sort(dists)
    nearest_other = ds[1] if len(ds) > 1 else 0.0
    dall = np.array([np.abs(cand_vec - v).sum() for v in all_vecs])
    return [dc, nearest_other, float(dists.mean()), float(dall.std())]


def build_row_field(corrupted_card, support_cards, options, field):
    cc_vec = smiles_vec(corrupted_card["smiles_features"])
    cc_vendor = corrupted_card["vendor_family_token"]
    corrupt_tok = corrupted_card[field]
    sup_idx = {s["candidate_token"]: s for s in support_cards if s["repair_field"] == field}
    sup_vec = {t: smiles_vec(s["smiles_features"]) for t, s in sup_idx.items()}
    all_vecs = [sup_vec[t] for t in options if t in sup_vec]

    rows, toks, isc = [], [], []
    L1sa, L1all, L1ng, cosall, hints = [], [], [], [], []
    for tok in options:
        s = sup_idx.get(tok)
        if s is not None:
            cand_vec = sup_vec[tok]
            cv = s["vendor_family_token"]; hint = s["evidence_rank_hint"]
            vmiss = (cv == "vendor_missing"); vmatch = (cv == cc_vendor and not vmiss)
        else:
            cand_vec = cc_vec.copy(); hint = 99; vmiss = True; vmatch = False
        feat, rel = _pair(cc_vec, cand_vec, vmatch, vmiss, hint)
        feat = feat + (_setrel(cand_vec, all_vecs) if all_vecs else [0.0, 0.0, 0.0, 0.0])
        rows.append(feat); toks.append(tok); isc.append(1.0 if tok == corrupt_tok else 0.0)
        L1sa.append(rel[0]); L1all.append(rel[1]); L1ng.append(rel[2]); cosall.append(rel[3]); hints.append(rel[4])

    X = np.asarray(rows, dtype=np.float64)
    isc = np.asarray(isc)

    def rel_block(arr, hi_good=False):
        a = np.asarray(arr, float)
        r = a.argsort().argsort().astype(float) / max(len(a)-1, 1)
        z = (a - a.mean())/a.std() if a.std() > 0 else np.zeros_like(a)
        med = np.median(a); mn = a.min()
        ratio = a/(med if med != 0 else 1); ratio_min = a/(mn if mn != 0 else 1)
        ismin = (a == (a.max() if hi_good else a.min())).astype(float)
        return np.column_stack([r, z, ratio, ratio_min, ismin])
    rel_parts = [rel_block(L1sa), rel_block(L1all), rel_block(L1ng),
                 rel_block(cosall, hi_good=True), rel_block(hints)]
    X = np.hstack([X, np.hstack(rel_parts), isc.reshape(-1, 1)])
    return X, toks, isc


def load_rows(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {"id": row["id"], "corrupted_card": json.loads(row["corrupted_card"]),
                   "support_cards": json.loads(row["support_cards"]),
                   "source_options": json.loads(row["source_options"]),
                   "name_type_options": json.loads(row["name_type_options"]),
                   "library_options": json.loads(row["library_options"])}
            if row.get("answer_json"):
                rec["answer"] = json.loads(row["answer_json"])
            out.append(rec)
    return out


def build_field(rows, field, with_labels=True):
    Xs, ys, g, meta = [], [], [], []
    for r in rows:
        X, toks, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                       r[OPTIONS_KEY[field]], field)
        Xs.append(X); g.append(len(toks))
        if with_labels:
            truth = r["answer"][field]
            ys.append(np.array([1.0 if t == truth else 0.0 for t in toks]))
            meta.append((toks, truth, isc))
        else:
            meta.append((toks, None, isc))
    return Xs, (np.concatenate(ys) if with_labels else None), g, meta


def ranker_params(seed):
    return dict(objective="lambdarank", metric="ndcg", n_estimators=350, learning_rate=0.04,
                num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)


def clf_params(seed):
    return dict(objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=31,
                min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)


def train_predict_field(train_rows, test_rows, field):
    Xs_tr, ytr, gtr, _ = build_field(train_rows, field)
    Xtr = np.vstack(Xs_tr)
    Xs_te, _, g_te, meta_te = build_field(test_rows, field, with_labels=False)
    bounds = np.concatenate([[0], np.cumsum(g_te)])
    Xte = np.vstack(Xs_te)

    score_cols = []
    for seed in SEEDS:
        rk = lgb.LGBMRanker(**ranker_params(seed)); rk.fit(Xtr, ytr.astype(int), group=gtr)
        score_cols.append(rk.predict(Xte))
        cl = lgb.LGBMClassifier(**clf_params(seed)); cl.fit(Xtr, ytr.astype(int))
        score_cols.append(cl.predict_proba(Xte)[:, 1])

    ens = np.zeros(len(Xte))
    for ri in range(len(test_rows)):
        s = slice(bounds[ri], bounds[ri+1]); n = bounds[ri+1]-bounds[ri]
        acc = np.zeros(n)
        for col in score_cols:
            acc += rankdata(col[s])
        ens[s] = acc / (len(score_cols)*n)

    preds = {}
    for ri, (toks, _, isc) in enumerate(meta_te):
        s = ens[bounds[ri]:bounds[ri+1]].copy(); s[isc.astype(bool)] = -np.inf
        preds[test_rows[ri]["id"]] = toks[int(np.argmax(s))]
    return preds


def main():
    print("Loading data...")
    train_rows = load_rows(ROOT / "train.csv")
    test_rows = load_rows(ROOT / "test.csv")
    print(f"train={len(train_rows)} test={len(test_rows)} (seeds={SEEDS})")

    field_preds = {}
    for f in FIELDS:
        print(f"Training + predicting {f} ...")
        field_preds[f] = train_predict_field(train_rows, test_rows, f)

    ids = [r["id"] for r in test_rows]
    out_path = WORK / "submission.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "answer_json"])
        for rid in ids:
            ans = {"source_token": field_preds["source_token"][rid],
                   "name_type_token": field_preds["name_type_token"][rid],
                   "library_token": field_preds["library_token"][rid]}
            w.writerow([rid, json.dumps(ans, separators=(",", ":"))])
    print(f"Wrote {out_path} ({len(ids)} rows)")


if __name__ == "__main__":
    main()
