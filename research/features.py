"""Feature engineering for the anonymized perturbation-card repair task.

For each row we have a corrupted_card (true compound's structure features) and a
set of candidate support_cards, one per candidate token per field. We build, for
every (corrupted_card, candidate) pair, a feature vector describing how well the
candidate matches the query. A learned ranker then picks the best candidate.
"""
import json
import numpy as np

SCALARS = ["aromatic_count", "branch_count", "charge_count", "length_bin",
           "ring_digit_count", "stereo_count"]
ATOMS = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
NGRAM_N = 12
FIELDS = ["source_token", "name_type_token", "library_token"]


def smiles_vec(s):
    """Flatten a smiles_features dict into a numeric vector (order is stable)."""
    v = [s[k] for k in SCALARS]
    v += [s["atom_counts"][k] for k in ATOMS]
    v += list(s["ngram_buckets"])
    return np.asarray(v, dtype=np.float64)


def _pair_features(cc_smiles, cand_smiles, vendor_match, vendor_missing,
                   hint, cc_vec):
    """Feature vector for one (corrupted, candidate) pair."""
    cand_vec = smiles_vec(cand_smiles)
    diff = cand_vec - cc_vec
    absdiff = np.abs(diff)

    feats = []
    names = []

    # raw per-feature signed + abs differences
    allnames = SCALARS + ["atom_" + a for a in ATOMS] + [f"ng{i}" for i in range(NGRAM_N)]
    for i, nm in enumerate(allnames):
        feats.append(diff[i]); names.append("d_" + nm)
        feats.append(absdiff[i]); names.append("ad_" + nm)

    # split into blocks for aggregate distances
    n_scalar = len(SCALARS)
    n_atom = len(ATOMS)
    sc_a = absdiff[:n_scalar]
    at_a = absdiff[n_scalar:n_scalar + n_atom]
    ng_a = absdiff[n_scalar + n_atom:]
    sc_v_c = cc_vec[:n_scalar]; sc_v_d = cand_vec[:n_scalar]
    at_v_c = cc_vec[n_scalar:n_scalar + n_atom]; at_v_d = cand_vec[n_scalar:n_scalar + n_atom]
    ng_v_c = cc_vec[n_scalar + n_atom:]; ng_v_d = cand_vec[n_scalar + n_atom:]

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    # aggregate L1 distances per block
    feats.append(sc_a.sum()); names.append("L1_scalar")
    feats.append(at_a.sum()); names.append("L1_atom")
    feats.append(ng_a.sum()); names.append("L1_ngram")
    feats.append(sc_a.sum() + at_a.sum()); names.append("L1_scalar_atom")
    feats.append(absdiff.sum()); names.append("L1_all")
    # L2
    feats.append(float(np.sqrt((sc_a**2).sum() + (at_a**2).sum()))); names.append("L2_scalar_atom")
    feats.append(float(np.sqrt((ng_a**2).sum()))); names.append("L2_ngram")
    # cosine similarities
    feats.append(cos(at_v_c, at_v_d)); names.append("cos_atom")
    feats.append(cos(ng_v_c, ng_v_d)); names.append("cos_ngram")
    feats.append(cos(sc_v_c, sc_v_d)); names.append("cos_scalar")
    feats.append(cos(cc_vec, cand_vec)); names.append("cos_all")
    # exact-block-match indicators
    feats.append(float(at_a.sum() == 0)); names.append("atom_exact")
    feats.append(float(sc_a.sum() == 0)); names.append("scalar_exact")
    feats.append(float(sc_a.sum() == 0 and at_a.sum() == 0)); names.append("core_exact")
    # total heavy-atom counts
    feats.append(float(at_v_c.sum())); names.append("cc_total_atoms")
    feats.append(float(at_v_d.sum())); names.append("cand_total_atoms")
    feats.append(float(at_v_d.sum() - at_v_c.sum())); names.append("d_total_atoms")
    # vendor / hint signals
    feats.append(float(vendor_match)); names.append("vendor_match")
    feats.append(float(vendor_missing)); names.append("vendor_missing")
    feats.append(float(hint)); names.append("hint")

    # absolute structure values (let the model learn exemplar priors)
    for i, nm in enumerate(SCALARS):
        feats.append(cand_vec[i]); names.append("cand_" + nm)
        feats.append(cc_vec[i]); names.append("cc_" + nm)
    feats.append(float(ng_v_d.sum())); names.append("cand_ngram_sum")
    feats.append(float(ng_v_c.sum())); names.append("cc_ngram_sum")
    feats.append(float(at_v_d.sum() and ng_v_d.sum() / max(at_v_d.sum(), 1)))
    names.append("cand_ngram_per_atom")

    return np.asarray(feats, dtype=np.float64), names


def build_row_field(corrupted_card, support_cards, options, field):
    """Return (X [n_cand, n_feat], cand_tokens [n_cand], feat_names) for one field.

    Candidates are taken from `options` (the authoritative candidate set). Each
    candidate's support card supplies its smiles/vendor/hint; if a candidate has
    no support card (rare), neutral values are used.
    """
    cc_smiles = corrupted_card["smiles_features"]
    cc_vendor = corrupted_card["vendor_family_token"]
    cc_vec = smiles_vec(cc_smiles)
    corrupt_tok = corrupted_card[field]

    # index support cards for this field by candidate token
    sup_idx = {}
    for s in support_cards:
        if s["repair_field"] == field:
            sup_idx[s["candidate_token"]] = s

    rows = []
    toks = []
    is_corrupt = []
    names = None
    for tok in options:
        s = sup_idx.get(tok)
        if s is not None:
            cand_smiles = s["smiles_features"]
            cv = s["vendor_family_token"]
            hint = s["evidence_rank_hint"]
            vendor_missing = (cv == "vendor_missing")
            vendor_match = (cv == cc_vendor and not vendor_missing)
        else:
            # no support card: neutral / far
            cand_smiles = cc_smiles  # zero diff is misleading; use large hint
            hint = 99
            vendor_missing = True
            vendor_match = False
        f, names = _pair_features(cc_smiles, cand_smiles, vendor_match,
                                  vendor_missing, hint, cc_vec)
        rows.append(f)
        toks.append(tok)
        is_corrupt.append(1.0 if tok == corrupt_tok else 0.0)

    X = np.vstack(rows)
    is_corrupt = np.asarray(is_corrupt)
    X = np.hstack([X, is_corrupt.reshape(-1, 1)])
    names = names + ["is_corrupt"]

    # within-row relational features (rank / z-score / is-min) for key distances
    def add_rel(colname):
        j = names.index(colname)
        col = X[:, j]
        order = col.argsort().argsort().astype(np.float64)  # rank 0..n-1
        rel_rank = order / max(len(col) - 1, 1)
        mu, sd = col.mean(), col.std()
        z = (col - mu) / sd if sd > 0 else np.zeros_like(col)
        is_min = (col == col.min()).astype(np.float64)
        return rel_rank, z, is_min

    extra = []
    extra_names = []
    for base in ["L1_scalar_atom", "L1_all", "L1_ngram", "cos_all", "cos_atom", "hint"]:
        rr, z, im = add_rel(base)
        extra.append(rr); extra_names.append("rank_" + base)
        extra.append(z); extra_names.append("z_" + base)
        extra.append(im); extra_names.append("ismin_" + base)
    if extra:
        X = np.hstack([X, np.column_stack(extra)])
        names = names + extra_names

    return X, toks, names, is_corrupt


def load_rows(path):
    import csv
    csv.field_size_limit(10**8)
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "id": row["id"],
                "corrupted_card": json.loads(row["corrupted_card"]),
                "support_cards": json.loads(row["support_cards"]),
                "source_options": json.loads(row["source_options"]),
                "name_type_options": json.loads(row["name_type_options"]),
                "library_options": json.loads(row["library_options"]),
            }
            if row.get("answer_json"):
                rec["answer"] = json.loads(row["answer_json"])
            out.append(rec)
    return out


OPTIONS_KEY = {
    "source_token": "source_options",
    "name_type_token": "name_type_options",
    "library_token": "library_options",
}
