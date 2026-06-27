"""Quick validation of JOINT triple inference vs independent per-field argmax.

Pure geometry (no training): for each row, take top-K candidates per field by L1
distance to the corrupted card, form K^3 triples, and pick the triple minimising a
joint-consistency score (pairwise distances among {cc, s_card, n_card, l_card}).
Compare all-three-exact to independent nearest-neighbour. Also under shift-sim.
"""
import sys, json, csv
import numpy as np
sys.path.insert(0, "research")
from featx import smiles_vec, perturb_smiles_vec, FIELDS, OPTIONS_KEY
csv.field_size_limit(10**8)


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


def d15(a, b):
    return np.abs(a[:15] - b[:15]).sum()


def row_cards(r, field, perturb_rng):
    cc = smiles_vec(r["corrupted_card"]["smiles_features"])
    sup = {s["candidate_token"]: s for s in r["support_cards"] if s["repair_field"] == field}
    corrupt = r["corrupted_card"][field]
    cards = {}
    for tok in r[OPTIONS_KEY[field]]:
        if tok == corrupt:
            continue
        s = sup.get(tok)
        v = smiles_vec(s["smiles_features"]) if s else cc.copy()
        if perturb_rng is not None and s is not None:
            v = perturb_smiles_vec(v, perturb_rng)
        cards[tok] = v
    return cc, cards


def eval_mode(rows, perturb, K=5, wjoint=1.0):
    indep_all3 = 0
    joint_all3 = 0
    n = 0
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(2000 + ri) if perturb else None
        cc = smiles_vec(r["corrupted_card"]["smiles_features"])
        cards = {}
        topk = {}
        for f in FIELDS:
            _, c = row_cards(r, f, rng)
            cards[f] = c
            dist = {t: d15(cc, v) for t, v in c.items()}
            topk[f] = sorted(dist, key=dist.get)[:K]
        truth = r["answer"]
        # independent: nearest per field
        indep = {f: min(cards[f], key=lambda t: d15(cc, cards[f][t])) for f in FIELDS}
        indep_all3 += all(indep[f] == truth[f] for f in FIELDS)
        # joint: best triple among topK^3
        best = None; best_sc = 1e18
        for s in topk["source_token"]:
            vs = cards["source_token"][s]
            for nm in topk["name_type_token"]:
                vn = cards["name_type_token"][nm]
                for l in topk["library_token"]:
                    vl = cards["library_token"][l]
                    # marginal closeness + mutual consistency
                    marg = d15(cc, vs) + d15(cc, vn) + d15(cc, vl)
                    mutual = d15(vs, vn) + d15(vs, vl) + d15(vn, vl)
                    sc = marg + wjoint * mutual
                    if sc < best_sc:
                        best_sc = sc; best = (s, nm, l)
        joint_all3 += (best[0] == truth["source_token"] and best[1] == truth["name_type_token"]
                       and best[2] == truth["library_token"])
        n += 1
    return indep_all3 / n, joint_all3 / n


rows = load_rows("dataset/public/train.csv")
print(f"rows={len(rows)}  (geometry-only, no training)")
for perturb in [False, True]:
    tag = "SHIFT" if perturb else "CLEAN"
    for w in [0.0, 0.5, 1.0, 2.0]:
        ia, ja = eval_mode(rows, perturb, K=5, wjoint=w)
        print(f"  {tag}  wjoint={w:.1f}  indep_all3={ia:.3f}  joint_all3={ja:.3f}")
