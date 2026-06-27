"""Model-family bake-off with shift-robust augmentation, evaluated on clean-CV and
shift-CV (the faithful proxy for the real test). Also evaluates the per-field
ensemble of all model families. Saves OOF score arrays for offline analysis.

Models: LGBM ranker, LGBM clf, XGB ranker, XGB clf, CatBoost ranker, CatBoost clf.
Selection metric: SHIFT all-three / score.
"""
import sys, json, csv, time, argparse
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import rankdata
sys.path.insert(0, "research")
sys.path.insert(0, ".")
from featx import build_row_field, FIELDS, OPTIONS_KEY
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


def build_field(rows, field, perturb_seed=None, params=None):
    out = []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(perturb_seed + ri) if perturb_seed is not None else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, rng, params)
        truth = r["answer"][field]
        y = np.array([1.0 if t == truth else 0.0 for t in toks])
        out.append((X, y, toks, truth, isc))
    return out


# ---- model trainers: each returns predict(X)->score (higher=better) ---------
def train_lgbm_ranker(Xtr, ytr, gtr, seed):
    import lightgbm as lgb
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=350, learning_rate=0.04,
                       num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
    m.fit(Xtr, ytr.astype(int), group=gtr)
    return m.predict


def train_lgbm_clf(Xtr, ytr, gtr, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", n_estimators=350, learning_rate=0.04,
                           num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1)
    m.fit(Xtr, ytr.astype(int))
    return lambda X: m.predict_proba(X)[:, 1]


def train_xgb_ranker(Xtr, ytr, gtr, seed):
    import xgboost as xgb
    m = xgb.XGBRanker(objective="rank:pairwise", n_estimators=350, learning_rate=0.04,
                      max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                      random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr.astype(int), group=gtr)
    return m.predict


def train_xgb_clf(Xtr, ytr, gtr, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=350, learning_rate=0.04,
                          max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr.astype(int))
    return lambda X: m.predict_proba(X)[:, 1]


def train_cat_ranker(Xtr, ytr, gtr, seed):
    from catboost import CatBoost, Pool
    gid = np.repeat(np.arange(len(gtr)), gtr)
    pool = Pool(Xtr, label=ytr.astype(float), group_id=gid)
    m = CatBoost(dict(loss_function="YetiRank", iterations=350, learning_rate=0.05,
                      depth=6, random_seed=seed, verbose=False, thread_count=-1))
    m.fit(pool)
    return lambda X: m.predict(X)


def train_cat_clf(Xtr, ytr, gtr, seed):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=350, learning_rate=0.05, depth=6, random_seed=seed,
                           verbose=False, thread_count=-1)
    m.fit(Xtr, ytr.astype(int))
    return lambda X: m.predict_proba(X)[:, 1]


TRAINERS = {
    "lgbm_rank": train_lgbm_ranker, "lgbm_clf": train_lgbm_clf,
    "xgb_rank": train_xgb_ranker, "xgb_clf": train_xgb_clf,
    "cat_rank": train_cat_ranker, "cat_clf": train_cat_clf,
}


# mix3 augmentation (mild->strong) + real-shift-estimate validation perturbation
AUG_PARAMS = [dict(mode="shrink_noise", atom_drop=0.05, count_noise=0.10),
              dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16),
              dict(mode="shrink_noise", atom_drop=0.11, count_noise=0.22)]
VAL_PARAMS = dict(mode="shrink_noise", atom_drop=0.08, count_noise=0.16)  # ~ real test


def oof_for_model(rows, field, model_key, n_aug, perturb_val, seed=SEED):
    clean = build_field(rows, field, None)
    augs = [build_field(rows, field, 5000 + 777*k, AUG_PARAMS[k]) for k in range(min(n_aug, len(AUG_PARAMS)))]
    valp = build_field(rows, field, 9000, VAL_PARAMS) if perturb_val else None
    n = len(rows)
    # group boundaries for clean
    glen = [len(clean[ri][1]) for ri in range(n)]
    bounds = np.concatenate([[0], np.cumsum(glen)])
    oof = np.zeros(bounds[-1])
    kf = KFold(5, shuffle=True, random_state=SEED)
    for tr, va in kf.split(np.arange(n)):
        Xtr, ytr, gtr = [], [], []
        for ri in tr:
            for ds in [clean] + augs:
                X, y, *_ = ds[ri]
                Xtr.append(X); ytr.append(y); gtr.append(len(y))
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        pred = TRAINERS[model_key](Xtr, ytr, gtr, seed)
        for ri in va:
            ds = valp if perturb_val else clean
            X = ds[ri][0]
            oof[bounds[ri]:bounds[ri+1]] = pred(X)
    return oof, clean, bounds


def eval_oof(oof, clean, bounds, n):
    correct = np.zeros(n, bool)
    for ri in range(n):
        X, y, toks, truth, isc = clean[ri]
        sc = oof[bounds[ri]:bounds[ri+1]].copy(); sc[isc.astype(bool)] = -np.inf
        correct[ri] = (toks[int(np.argmax(sc))] == truth)
    return correct


def rank_norm(oof, bounds, n):
    out = np.zeros_like(oof)
    for ri in range(n):
        s = slice(bounds[ri], bounds[ri+1])
        out[s] = rankdata(oof[s]) / (bounds[ri+1]-bounds[ri])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm_rank,lgbm_clf,xgb_rank,xgb_clf,cat_rank,cat_clf")
    ap.add_argument("--n_aug", type=int, default=2)
    args = ap.parse_args()
    models = args.models.split(",")
    rows = load_rows("dataset/public/train.csv")
    n = len(rows)
    print(f"rows={n} models={models} n_aug={args.n_aug}", flush=True)

    results = {}  # (mode, model, field) -> correct array; plus rank-normed oof for ensemble
    rnorm = {}
    for mode, pv in [("clean", False), ("shift", True)]:
        for mk in models:
            for f in FIELDS:
                t0 = time.time()
                oof, clean, bounds = oof_for_model(rows, f, mk, args.n_aug, pv)
                corr = eval_oof(oof, clean, bounds, n)
                results[(mode, mk, f)] = corr
                rnorm[(mode, mk, f)] = (rank_norm(oof, bounds, n), bounds, clean)
                print(f"  [{mode}] {mk:10s} {f:16s} acc={corr.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        # per-field accs & score for each model
        for mk in models:
            at = np.ones(n, bool)
            for f in FIELDS: at &= results[(mode, mk, f)]
            fa = np.mean([results[(mode, mk, f)].mean() for f in FIELDS])
            print(f"  [{mode}] {mk:10s} ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
        # ensemble of all models (avg rank-normed)
        ens_corr = {}
        for f in FIELDS:
            acc = None; bnd = None; cl = None
            for mk in models:
                rn, bnd, cl = rnorm[(mode, mk, f)]
                acc = rn if acc is None else acc + rn
            correct = np.zeros(n, bool)
            for ri in range(n):
                X, y, toks, truth, isc = cl[ri]
                sc = acc[bnd[ri]:bnd[ri+1]].copy(); sc[isc.astype(bool)] = -np.inf
                correct[ri] = (toks[int(np.argmax(sc))] == truth)
            ens_corr[f] = correct
        at = np.ones(n, bool)
        for f in FIELDS: at &= ens_corr[f]
        fa = np.mean([ens_corr[f].mean() for f in FIELDS])
        print(f"  [{mode}] ENSEMBLE(all) src/name/lib="
              f"{ens_corr['source_token'].mean():.3f}/{ens_corr['name_type_token'].mean():.3f}/{ens_corr['library_token'].mean():.3f} "
              f"ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)


if __name__ == "__main__":
    main()
