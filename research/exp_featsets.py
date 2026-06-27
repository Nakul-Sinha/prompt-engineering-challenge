"""Compare feature-set configs under (a) clean 5-fold CV and (b) shift-simulated
CV, to pick the config that is ROBUST to the observed test shift.

Clean CV  : validate on unperturbed val rows (optimistic, = current behaviour).
Shift CV  : validate on perturbed val rows (candidate smiles shrunk + noised)
            to mimic train->test library distance scale increase.
"""
import sys, time, json, csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata

sys.path.insert(0, "research")
from featx import build_row_field, FIELDS, OPTIONS_KEY

csv.field_size_limit(10**8)
SEED = 42


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


def rk_params():
    return dict(objective="lambdarank", metric="ndcg", n_estimators=350,
                learning_rate=0.04, num_leaves=31, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)


def build_field(rows, field, cfg, perturb=False):
    Xs, ys, g, meta = [], [], [], []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(1000+ri) if perturb else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, cfg, rng)
        truth = r["answer"][field]
        ys.append(np.array([1.0 if t==truth else 0.0 for t in toks]))
        Xs.append(X); g.append(len(toks)); meta.append((ri, toks, truth, isc))
    return np.vstack(Xs), np.concatenate(ys), np.array(g), meta, names


def cv_field(rows, field, cfg, perturb_val):
    Xc, y, g, meta, names = build_field(rows, field, cfg, perturb=False)
    bounds = np.concatenate([[0], np.cumsum(g)])
    # perturbed features for validation only
    if perturb_val:
        Xp, _, gp, _, _ = build_field(rows, field, cfg, perturb=True)
    n = len(rows)
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in kf.split(np.arange(n)):
        trm = np.zeros(len(y), bool); tg = []
        for ri in tr: trm[bounds[ri]:bounds[ri+1]] = True; tg.append(g[ri])
        m = lgb.LGBMRanker(**rk_params()); m.fit(Xc[trm], y[trm].astype(int), group=tg)
        for ri in va:
            s = slice(bounds[ri], bounds[ri+1])
            Xv = Xp[s] if perturb_val else Xc[s]
            oof[s] = m.predict(Xv)
    correct = np.zeros(n, bool)
    for ri, toks, truth, isc in meta:
        sc = oof[bounds[ri]:bounds[ri+1]].copy(); sc[isc.astype(bool)] = -np.inf
        correct[ri] = (toks[int(np.argmax(sc))] == truth)
    return correct


CONFIGS = {
    "current(all)":   dict(raw_diff=1,absdist=1,cos=1,absval=1,vendor=1,hint=1,rel=1),
    "+comp":          dict(raw_diff=1,absdist=1,cos=1,absval=1,vendor=1,hint=1,rel=1,comp=1),
    "+ngdist":        dict(raw_diff=1,absdist=1,cos=1,absval=1,vendor=1,hint=1,rel=1,ngdist=1),
    "+comp+ngdist":   dict(raw_diff=1,absdist=1,cos=1,absval=1,vendor=1,hint=1,rel=1,comp=1,ngdist=1),
}


def main():
    rows = load_rows("dataset/public/train.csv")
    print(f"rows={len(rows)}\n")
    for name, cfg in CONFIGS.items():
        res = {}
        for mode, pv in [("clean", False), ("shift", True)]:
            cc = {}
            for f in FIELDS:
                cc[f] = cv_field(rows, f, cfg, pv)
            at = np.ones(len(rows), bool)
            for f in FIELDS: at &= cc[f]
            fa = np.mean([cc[f].mean() for f in FIELDS])
            score = 0.97*at.mean()+0.03*fa
            res[mode] = (cc, at.mean(), score)
        c = res["clean"]; s = res["shift"]
        print(f"{name:22s} CLEAN src/name/lib={c[0]['source_token'].mean():.3f}/{c[0]['name_type_token'].mean():.3f}/{c[0]['library_token'].mean():.3f} "
              f"all3={c[1]:.3f} score={c[2]:.4f}  ||  SHIFT lib={s[0]['library_token'].mean():.3f} all3={s[1]:.3f} score={s[2]:.4f}")


if __name__ == "__main__":
    main()
