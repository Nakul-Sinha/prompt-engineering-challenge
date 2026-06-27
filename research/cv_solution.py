"""Honest shift-CV that mirrors solution.py exactly (mix3 aug, lgbm rank+clf
ensemble, seed-bagged). Validation perturbation = real-shift estimate (.08/.16),
DIFFERENT seed from training aug -> honest. Optionally also perturb the corrupted
card (test molecules are smaller too) to test that variant.
"""
import sys, json, csv, argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import rankdata
sys.path.insert(0, ".")
import solution as S
csv.field_size_limit(10**8)

VAL_PARAMS = dict(atom_drop=0.08, count_noise=0.16)


def build_field_val(rows, field, seed, params, perturb_cc=False):
    """Build features perturbing candidates (and optionally cc) for validation."""
    out = []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(seed + ri)
        cc = dict(r["corrupted_card"])
        if perturb_cc:
            v = S.smiles_vec(cc["smiles_features"])
            v = S.perturb_vec(v, rng, **params)
            sf = dict(cc["smiles_features"])
            sf2 = {**sf}
            sf2["aromatic_count"], sf2["branch_count"], sf2["charge_count"], sf2["length_bin"], \
                sf2["ring_digit_count"], sf2["stereo_count"] = [int(x) for x in v[:6]]
            ac = dict(sf["atom_counts"])
            for i, a in enumerate(S.ATOMS):
                ac[a] = int(v[6+i])
            sf2["atom_counts"] = ac
            sf2["ngram_buckets"] = [int(x) for x in v[15:]]
            cc = {**cc, "smiles_features": sf2}
        X, toks, isc = S.build_row_field(cc, r["support_cards"],
                                         r[S.OPTIONS_KEY[field]], field, rng, params)
        truth = r["answer"][field]
        out.append((X, toks, truth, isc))
    return out


def run(rows, perturb_cc):
    n = len(rows)
    kf = KFold(5, shuffle=True, random_state=42)
    field_correct = {}
    for field in S.FIELDS:
        # training datasets: clean + mix3 augs
        clean = S.build_field(rows, field, None)
        augs = [S.build_field(rows, field, S.AUG_SEEDS[k], S.AUG_PARAMS[k]) for k in range(S.N_AUG)]
        valp = build_field_val(rows, field, 9000, VAL_PARAMS, perturb_cc)
        correct = np.zeros(n, bool)
        for tr, va in kf.split(np.arange(n)):
            Xtr, ytr, gtr = [], [], []
            for ri in tr:
                for ds in [clean] + augs:
                    Xs, y, g, meta = ds
                    Xtr.append(Xs[ri]); ytr.append(y[sum(g[:ri]):sum(g[:ri+1])]); gtr.append(g[ri])
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            cols = []
            for seed in S.SEEDS:
                rk = lgb.LGBMRanker(**S.ranker_params(seed)); rk.fit(Xtr, ytr.astype(int), group=gtr)
                cl = lgb.LGBMClassifier(**S.clf_params(seed)); cl.fit(Xtr, ytr.astype(int))
                cols.append(("rk", seed, rk)); cols.append(("cl", seed, cl))
            for ri in va:
                X, toks, truth, isc = valp[ri]
                acc = np.zeros(len(toks))
                for kind, seed, m in cols:
                    p = m.predict(X) if kind == "rk" else m.predict_proba(X)[:, 1]
                    acc += rankdata(p)
                acc[isc.astype(bool)] = -1e9
                correct[ri] = (toks[int(np.argmax(acc))] == truth)
        field_correct[field] = correct
        print(f"  {field:16s} shift_acc={correct.mean():.3f}", flush=True)
    at = np.ones(n, bool)
    for f in S.FIELDS: at &= field_correct[f]
    fa = np.mean([field_correct[f].mean() for f in S.FIELDS])
    print(f"  perturb_cc={perturb_cc}  ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)


def build_field_with_g(rows, field, seed, params):
    pass


if __name__ == "__main__":
    rows = S.load_rows("dataset/public/train.csv")
    print(f"rows={len(rows)} mirror of solution.py (mix3, rank+clf, seeds={S.SEEDS})")
    for pcc in [False, True]:
        run(rows, pcc)
