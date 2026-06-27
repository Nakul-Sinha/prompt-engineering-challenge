"""Configurable feature builder + train/test-shift simulation, to find the
feature set that is ROBUST to the observed library distance-scale shift.

Feature groups (toggle via config):
  raw_diff   : per-feature signed+abs diffs
  absdist    : block L1/L2 absolute distances
  cos        : cosine sims + exact-block flags
  absval     : absolute structure values of cc and candidate  (scale-dependent!)
  vendor     : vendor match/missing
  hint       : evidence_rank_hint (raw + rank)
  rel        : within-row rank / z / is-min / ratio-to-median  (scale-INVARIANT)
"""
import numpy as np

SCALARS = ["aromatic_count", "branch_count", "charge_count", "length_bin",
           "ring_digit_count", "stereo_count"]
ATOMS = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
NGRAM_N = 12
FIELDS = ["source_token", "name_type_token", "library_token"]
OPTIONS_KEY = {"source_token": "source_options",
               "name_type_token": "name_type_options",
               "library_token": "library_options"}


def smiles_vec(s):
    v = [s[k] for k in SCALARS]
    v += [s["atom_counts"][k] for k in ATOMS]
    v += list(s["ngram_buckets"])
    return np.asarray(v, dtype=np.float64)


def perturb_smiles_vec(vec, rng, atom_drop=0.06, count_noise=0.12, mode="shrink_noise"):
    """Simulate test-shift. Several families so we can train on one and validate on
    another (honest robustness check)."""
    v = vec.astype(np.float64).copy()
    if mode == "shrink_noise":
        v[6:15] = np.maximum(0, np.round(v[6:15] * (1 - atom_drop)))
        noise = rng.poisson(count_noise, size=v.shape) - rng.poisson(count_noise, size=v.shape)
        v = np.maximum(0, v + noise)
    elif mode == "grow_noise":  # opposite-direction size change + noise
        v[6:15] = np.maximum(0, np.round(v[6:15] * (1 + atom_drop)))
        noise = rng.poisson(count_noise, size=v.shape) - rng.poisson(count_noise, size=v.shape)
        v = np.maximum(0, v + noise)
    elif mode == "scale_jitter":  # multiplicative jitter on all counts (different family)
        fac = rng.normal(1.0, count_noise, size=v.shape)
        v = np.maximum(0, np.round(v * fac))
    return v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def build_row_field(corrupted_card, support_cards, options, field, cfg,
                    perturb_rng=None, perturb_params=None):
    cc_smiles = corrupted_card["smiles_features"]
    cc_vendor = corrupted_card["vendor_family_token"]
    cc_vec = smiles_vec(cc_smiles)
    corrupt_tok = corrupted_card[field]
    sup_idx = {s["candidate_token"]: s for s in support_cards
               if s["repair_field"] == field}

    n_scalar, n_atom = len(SCALARS), len(ATOMS)
    rows, toks, is_corrupt = [], [], []
    # cache distance scalars for relational features
    L1sa_list, L1all_list, L1ng_list, cosall_list, hint_list = [], [], [], [], []
    base_feats = []

    for tok in options:
        s = sup_idx.get(tok)
        if s is not None:
            cand_vec = smiles_vec(s["smiles_features"])
            if perturb_rng is not None:
                pp = perturb_params or {}
                cand_vec = perturb_smiles_vec(cand_vec, perturb_rng, **pp)
            cv = s["vendor_family_token"]
            hint = s["evidence_rank_hint"]
            vmiss = (cv == "vendor_missing")
            vmatch = (cv == cc_vendor and not vmiss)
        else:
            cand_vec = cc_vec.copy(); hint = 99; vmiss = True; vmatch = False

        diff = cand_vec - cc_vec
        ad = np.abs(diff)
        sc_a, at_a, ng_a = ad[:n_scalar], ad[n_scalar:n_scalar+n_atom], ad[n_scalar+n_atom:]
        L1sa = sc_a.sum()+at_a.sum(); L1all = ad.sum(); L1ng = ng_a.sum()
        cosall = cos(cc_vec, cand_vec)

        f, nm = [], []
        if cfg.get("raw_diff"):
            allnm = SCALARS+["atom_"+a for a in ATOMS]+[f"ng{i}" for i in range(NGRAM_N)]
            for i, n in enumerate(allnm):
                f += [diff[i], ad[i]]; nm += ["d_"+n, "ad_"+n]
        if cfg.get("absdist"):
            f += [sc_a.sum(), at_a.sum(), ng_a.sum(), L1sa, L1all,
                  float(np.sqrt((sc_a**2).sum()+(at_a**2).sum())), float(np.sqrt((ng_a**2).sum()))]
            nm += ["L1_sc","L1_at","L1_ng","L1_sa","L1_all","L2_sa","L2_ng"]
        if cfg.get("cos"):
            f += [cos(cc_vec[n_scalar:n_scalar+n_atom], cand_vec[n_scalar:n_scalar+n_atom]),
                  cos(cc_vec[n_scalar+n_atom:], cand_vec[n_scalar+n_atom:]),
                  cos(cc_vec[:n_scalar], cand_vec[:n_scalar]), cosall,
                  float(at_a.sum()==0), float(sc_a.sum()==0), float(sc_a.sum()==0 and at_a.sum()==0)]
            nm += ["cos_at","cos_ng","cos_sc","cos_all","at_exact","sc_exact","core_exact"]
        if cfg.get("absval"):
            for i, n in enumerate(SCALARS):
                f += [cand_vec[i], cc_vec[i]]; nm += ["cand_"+n, "cc_"+n]
            f += [cand_vec[n_scalar:n_scalar+n_atom].sum(), cc_vec[n_scalar:n_scalar+n_atom].sum(),
                  cand_vec[n_scalar+n_atom:].sum(), cc_vec[n_scalar+n_atom:].sum()]
            nm += ["cand_atoms","cc_atoms","cand_ngsum","cc_ngsum"]
        if cfg.get("vendor"):
            f += [float(vmatch), float(vmiss)]; nm += ["vmatch","vmiss"]
        if cfg.get("hint"):
            f += [float(hint)]; nm += ["hint"]
        if cfg.get("comp"):
            # scale-invariant composition: atom fractions (robust to size shift)
            cc_at = cc_vec[n_scalar:n_scalar+n_atom]; cd_at = cand_vec[n_scalar:n_scalar+n_atom]
            cc_tot = max(cc_at.sum(), 1); cd_tot = max(cd_at.sum(), 1)
            cc_frac = cc_at/cc_tot; cd_frac = cd_at/cd_tot
            fd = cd_frac - cc_frac
            for i, a in enumerate(ATOMS):
                f += [fd[i], abs(fd[i])]; nm += ["fd_"+a, "afd_"+a]
            f += [np.abs(fd).sum(), float(np.sqrt((fd**2).sum()))]; nm += ["frac_L1","frac_L2"]
            # heavy-atom ratio C/N/O signatures (scale-invariant)
            f += [cd_frac[1], cc_frac[1]]; nm += ["cand_Cfrac","cc_Cfrac"]
        if cfg.get("ngdist"):
            # scale-invariant ngram distribution distances
            cc_ng = cc_vec[n_scalar+n_atom:]; cd_ng = cand_vec[n_scalar+n_atom:]
            cc_p = cc_ng/max(cc_ng.sum(),1); cd_p = cd_ng/max(cd_ng.sum(),1)
            l1 = np.abs(cc_p-cd_p).sum()
            m = 0.5*(cc_p+cd_p)
            def kl(a,b):
                mask=a>0; return float(np.sum(a[mask]*np.log((a[mask])/(b[mask]+1e-12)+1e-12)))
            jsd = 0.5*kl(cc_p,m)+0.5*kl(cd_p,m)
            # pearson corr of ngram profiles
            if cc_ng.std()>0 and cd_ng.std()>0:
                corr = float(np.corrcoef(cc_ng, cd_ng)[0,1])
            else:
                corr = 0.0
            f += [l1, jsd, corr]; nm += ["ng_pL1","ng_jsd","ng_corr"]

        base_feats.append((f, nm))
        L1sa_list.append(L1sa); L1all_list.append(L1all); L1ng_list.append(L1ng)
        cosall_list.append(cosall); hint_list.append(hint)
        toks.append(tok); is_corrupt.append(1.0 if tok==corrupt_tok else 0.0)

    # relational (scale-invariant) features
    L1sa_arr = np.array(L1sa_list); L1all_arr=np.array(L1all_list)
    L1ng_arr=np.array(L1ng_list); cosall_arr=np.array(cosall_list); hint_arr=np.array(hint_list)

    def rel_block(arr, hi_is_good=False):
        r = arr.argsort().argsort().astype(float)/max(len(arr)-1,1)
        if hi_is_good: r = 1-r
        z = (arr-arr.mean())/arr.std() if arr.std()>0 else np.zeros_like(arr)
        med = np.median(arr); ratio = arr/(med if med!=0 else 1)
        mn = arr.min(); ratio_min = arr/(mn if mn!=0 else 1)
        ismin = (arr==arr.min()).astype(float) if not hi_is_good else (arr==arr.max()).astype(float)
        return np.column_stack([r, z, ratio, ratio_min, ismin])

    rel_parts = []
    rel_names = []
    if cfg.get("rel"):
        for arr, base, hig in [(L1sa_arr,"L1sa",False),(L1all_arr,"L1all",False),
                               (L1ng_arr,"L1ng",False),(cosall_arr,"cosall",True),
                               (hint_arr,"hint",False)]:
            rel_parts.append(rel_block(arr, hig))
            rel_names += [f"{base}_rank",f"{base}_z",f"{base}_ratio",f"{base}_ratiomin",f"{base}_ismin"]

    Xb = np.array([bf[0] for bf in base_feats], dtype=np.float64) if base_feats[0][0] else np.zeros((len(toks),0))
    names = base_feats[0][1] if base_feats[0][0] else []
    parts = [Xb] if Xb.shape[1] else []
    if rel_parts:
        parts.append(np.hstack(rel_parts)); names = names + rel_names
    # is_corrupt always
    isc = np.array(is_corrupt)
    parts.append(isc.reshape(-1,1)); names = names + ["is_corrupt"]
    X = np.hstack(parts)
    return X, toks, names, isc
