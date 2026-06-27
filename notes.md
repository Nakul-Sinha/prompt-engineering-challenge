# Notes — Anonymized Perturbation-Card Metadata Repair

## Challenge facts
- **Task:** row-local token repair. For each row choose the true `source_token`,
  `name_type_token`, `library_token` from that row's candidate option lists.
- **Data:** `train.csv` (600 labeled rows), `test.csv` (300, no `answer_json`),
  `sample_submission.csv`. Each row: `corrupted_card`, `support_cards`,
  `source_options` (32), `name_type_options` (10), `library_options` (27),
  `repair_fields`, `answer_json` (train only).
- **Output:** `./working/submission.csv` with `id,answer_json`; `answer_json` is a
  JSON object with the three string tokens, each from the row's option list.
- **Metric:** `row_score = 0.97*all_three_exact + 0.03*field_accuracy`,
  `score = mean(row_score)`. Higher is better; max 1.0.
  → **Dominated by getting all three fields exactly right.** Maximizing each
  field's top-1 accuracy maximizes expected all-three (fields ~independent
  given features — confirmed: per-field-product ≈ observed all-three).
- **Rules / not allowed:** only the public CSVs; no external lookups, no
  compound/structure/id lookups, no hardcoded task-id→answer maps, no leaderboard
  probing. Tokens are salted **per row** (zero cross-row token overlap), so no
  global token mapping is possible — must solve from each row's own evidence.

## Data structure understanding
- `corrupted_card`: carries the **true compound's** `smiles_features`
  (atom counts, aromatic/ring/branch/stereo/charge counts, length_bin, 12 ngram
  buckets), a `vendor_family_token`, `qc_context`, and the three **corrupted**
  tokens (decoys).
- `support_cards`: exactly one card per candidate token per field
  (27+32+10 = 69). Each has `repair_field`, `candidate_token`,
  `vendor_family_token`, `smiles_features`, `evidence_rank_hint` (0–6).
- `qc_context` is only on the corrupted card (constant across a row's
  candidates) → not useful for ranking candidates.

## Key signals found (on train)
- **Structure similarity is the core signal.** The right candidate's support
  card is structurally close to the corrupted card. Exact feature match is rare
  (anonymization noise) → nearest-neighbour / learned ranking problem.
- **Answer is NEVER the corrupted token** (0/600 for all fields) and the
  corrupted token is always in-options → hard-exclude it at prediction.
- **Vendor match (source):** when corrupted vendor ≠ missing (235/600), the true
  `source` candidate's vendor matches it 81% of the time. Strong for source.
  Weak for name/library (their true vendors are "missing" ~61% of the time).
- **Cross-field vendor linkage is weak** (true name-vendor == true source-vendor
  only 70/600) → no useful stacking from it.
- **`name_type` is the hard field**: vendor mostly missing + weak structure link
  → ~0.72 top-1 is near its achievable ceiling from available evidence.

## Approach (compliant, learned ML)
Per field, build (corrupted, candidate) pair features:
- per-feature signed & abs diffs; block L1/L2 distances (scalar/atom/ngram);
  cosine sims; exact-block indicators; vendor match/missing; evidence hint;
  absolute structure values; within-row rank / z-score / is-min of key
  distances; `is_corrupt` flag.
Train an **ensemble** of LightGBM **LambdaRank** + **binary classifier**,
**seed-bagged** (seeds 42/1/7), average per-row rank-normalised scores, pick the
top non-corrupted candidate.

## Local CV progression (5-fold, group = row, OOF-honest)
| Step | name | source | library | all-three | SCORE |
|---|---|---|---|---|---|
| L1 nearest-neighbour heuristic | 0.63 | 0.64 | 0.78 | ~0.31 | ~0.34 |
| Single LGBM ranker | 0.683 | 0.828 | 0.848 | 0.497 | 0.5054 |
| + exclude corrupt token | 0.700 | 0.845 | 0.850 | 0.507 | 0.5154 |
| + absolute structure features | 0.710 | 0.842 | 0.862 | 0.528 | 0.5366 |
| + ranker/classifier ensemble | 0.720 | 0.838 | 0.863 | 0.535 | 0.5432 |
| **+ seed-bagging (FINAL, solution.py)** | **0.720** | **0.838** | **0.867** | **0.532** | **0.5400** |

Final official local CV = **0.5400** (5-fold, group=row, seed-bagged ensemble,
honest OOF). Seed-bagging ≈ single-seed within CV noise but lowers variance for
the private split.

## Submission history
- **v1** (no shift handling): clean CV 0.54 → **pre-check 0.41** (over-fit to train
  distance scale).
- **v2** (shift-robust: mix3 augmentation, lgbm rank+clf, 3 seeds): honest shift-CV
  **0.525** (all-three 0.517). Stable 0.517–0.523 across every shift-proxy variant
  (candidate-only / candidate+cc perturbation) → expected real test ~0.50–0.52.
  `working/submission.csv` = this model; 300 rows; passes validator; reproduced by an
  isolated smoke test (solution.py + dataset/public only).

## Where the score now stands / remaining ceiling
- Augmentation essentially removed the library shift penalty (shift 0.855 ≈ clean
  0.858). The limiter is now **name_type ≈ 0.72**, which is an information ceiling:
  its candidates' vendors are mostly "missing", structure link is weak, qc_context is
  constant within a row, and oracle cross-field triangulation adds only +0.014.
- To push all-three past ~0.53 one would need a name_type signal that does not appear
  to exist in the public features.

## Next actions / ideas (if more credits)
- Per-field hyperparameter tuning (name_type may want shallower trees).
- Calibrated 2nd-stage stack using predicted source/library to inform name_type
  (low expected gain given weak cross-field linkage).
- More ngram-derived shape features / monotonic constraints.
- H100 only useful for large sweeps; method (signal) is the bottleneck, not compute.

## v2 — Distribution-shift diagnosis & shift-robust training (post 0.41 pre-check)

The pre-submission check scored **0.41** while local 5-fold CV said 0.54 — a real
gap. Diagnosis:

- **library_token is distribution-shifted train→test.** The true library
  candidate's support card sits FURTHER from the corrupted card in test: the whole
  candidate-distance distribution shifts up (train median 12.8 → test 15.0), and
  test molecules are slightly smaller (cc heavy-atoms 30.6 → 28.7). Source/name
  distances barely move. A model keyed to train's absolute distance scale
  over-fits and drops on the shifted test.
- Built a **shift-simulation proxy** (perturb candidate structure features: shrink
  molecule + integer count noise). It reproduces the gap: current model scores
  clean-CV 0.538 but shift-CV 0.45 — matching the real 0.41.
- **Calibration:** real lib median 15.0 sits between sim "same"(.06/.12→14.3) and
  "strong"(.10/.22→15.6); the real shift is near the *strong* end.

**Fix that worked — training-time augmentation.** Add N_AUG perturbed copies of
each training row so the ranker learns scale-invariant boundaries. Shift-CV
0.46 → 0.53. Honest check (train on shift family A, evaluate on a *different*
family B) confirms the robustness is genuine, not memorisation:

| val shift family | no-aug | aug(n=2) | gain |
|---|---|---|---|
| none (clean)      | 0.542 | 0.533 | -0.01 |
| same (.06/.12)    | 0.448 | 0.518 | +0.07 |
| strong (.10/.22)  | 0.345 | 0.442 | +0.10 |
| grow (opposite)   | 0.442 | 0.490 | +0.05 |
| scalejit (other)  | 0.195 | 0.277 | +0.08 |

Augmentation helps on every unseen shift family. Since the real shift is near
"strong", stronger/mixed augmentation is being tuned (exp_augmix).

**Model bake-off (shift-CV, n_aug=2):** lgbm_clf 0.533 (best single), lgbm_rank
0.527, xgb 0.524, cat 0.514; naive all-6 ensemble 0.525 (dilutes — curated >
bloated). Curated weighted blend under search.

**Dead ends:** joint cross-field consistency (true cards cluster at dist 5.9 vs
21.2 random, but the mutual term is redundant with closeness-to-corrupted and
hurts); pure scale-invariant feature sets (stripping absolute features lowers both
clean and shift — richer features carry transferable signal); neural listwise
ranker (~on par with GBDT, far slower, dropped).

## Compliance
- Uses only public CSVs; learned ranking on provided features; no id→answer maps,
  no external lookups, no leaderboard probing, no metadata fingerprinting.
  `solution.py` regenerates `working/submission.csv` end-to-end from
  `dataset/public/`.
