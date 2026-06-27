# Approach: Anonymized Perturbation-Card Metadata Repair

**Recommended time spent:** 9 hours

## Summary
Each row is a corrupted small-molecule perturbation card; the goal is to restore
three row-local categorical tokens (`source_token`, `name_type_token`,
`library_token`) by choosing from that row's candidate lists. The metric is
`0.97*all_three_exact + 0.03*field_accuracy`, so the score is almost entirely
driven by getting all three fields exactly right; I optimise each field's top-1
accuracy (fields are ~independent given the evidence, so this maximises expected
all-three).

Tokens are salted per row (zero cross-row overlap), so there is no global mapping
to memorise, every row is solved from its own evidence. I framed it as a
**learned candidate-ranking** problem: for each field, score every candidate by how
well its support card matches the corrupted card, and pick the best non-corrupted
candidate.

## The key problem: train→test distribution shift
A first model scored ~0.54 on plain 5-fold CV but only **0.41** on the
pre-submission check. Investigation showed a genuine distribution shift: the true
`library` candidate's structure features sit **further** from the corrupted card in
test (candidate-distance median 12.8→15.0) and test molecules are slightly smaller
(corrupted-card heavy-atoms 30.6→28.7). Source/name distances barely move. A model
keyed to train's absolute distance scale over-fits and degrades on test.

I built a **shift-simulation proxy** (perturb candidate structure features: shrink
molecule + integer count noise) that reproduces the gap, the non-robust model
scores ~0.54 clean but ~0.41 to 0.46 under simulated shift, and used it as the
selection metric instead of optimistic clean CV.

## Model architecture
Per field, an **ensemble of LightGBM LambdaRank + binary classifier**, **seed-bagged
over 3 seeds**, trained on clean rows **plus 3 shift-augmented copies**. Per-row
rank-normalised scores from all members are averaged; the top non-corrupted
candidate is selected.

## Preprocessing / features (~130 per candidate)
Pairwise (corrupted_card, candidate support_card) features: per-feature signed/abs
diffs; block L1/L2 distances (scalar/atom/ngram); cosine sims; exact-block flags;
absolute structure values; vendor match/missing; evidence hint; **scale-invariant
atom-composition fractions**; within-row rank/z/ratio/is-min relational features;
and an `is_corrupt` flag.

## Key design decisions
1. **Shift-augmentation (the main win).** Each training row also appears with
   perturbed candidate features at three strengths ("mix3": mild→strong), teaching
   scale-invariant boundaries. On the shift proxy this lifts the metric from ~0.46
   to ~0.53; gains are largest on the weak fields (name 0.67→0.72, library
   0.82→0.85). Validated as genuine: training on one shift family still improves
   accuracy on *unseen* shift families (strong +0.10, opposite-direction +0.05,
   different-transform +0.08).
2. **Exclude the corrupted token.** The answer is never the corrupted token (0/600
   on train), so it is hard-masked at prediction.
3. **Rank+classifier ensemble, seed-bagged.** Reduces variance on the brittle
   all-three metric; LightGBM-only keeps the solution reproducible anywhere.
4. **Selection on a shift proxy, not clean CV.** Clean CV is optimistic under the
   observed shift; all choices were made on the shift-aware metric.

## What worked
- Shift-augmentation, the decisive lever (+0.07 to 0.10 on the realistic proxy).
- Structure-similarity ranking is the core signal; vendor match strongly
  disambiguates `source` (81% match when the corrupted vendor is present).
- Corrupted-token exclusion and atom-composition fractions: clean, reliable gains.

## What did not work / underperformed
- **Joint cross-field inference.** The three true support cards cluster tightly
  (mutual distance 5.9 vs 21.2 for random pairs), but the mutual-consistency term is
  redundant with closeness-to-corrupted and *hurts* triple selection.
- **Stripping to scale-invariant features only.** Removing absolute features lowered
  both clean and shift scores, the richer features carry transferable signal; the
  fix for shift was augmentation, not feature removal.
- **Neural listwise ranker.** ~On par with GBDT but far slower; no gain, dropped.
- **XGBoost/CatBoost diversity.** Model families are equivalent on the shift proxy
  (all 0.52 to 0.53); a diverse blend added only +0.002, not worth extra runtime
  dependencies.
- **`name_type` (~0.72) is near its ceiling**: its candidates' vendors are mostly
  "missing" and its structure link is weak; `qc_context` is constant within a row so
  cannot discriminate candidates.

## Local validation
Honest shift cross-validation (5-fold; group = row; validation perturbation drawn
with a different seed than the training augmentation, so robustness is not
memorised):

| Field | Shift top-1 accuracy |
|---|---|
| source_token | 0.833 |
| name_type_token | 0.720 |
| library_token | 0.855 |
| **all-three-exact** | **0.517** |
| **Shift-CV score** | **0.525** |

vs ~0.41 for the non-robust v1. (Plain clean CV reads ~0.54 but is optimistic under
the real shift.)

## Compliance
Uses only the public CSVs. Pure learned LightGBM ranking on the provided
structure/vendor/evidence features, no compound/structure/id lookups, no hardcoded
task-id→answer maps, no leaderboard probing, no metadata fingerprinting or row-order
leakage. `solution.py` reads `./dataset/public/` and regenerates
`./working/submission.csv` end-to-end (~4 min CPU). The submitted CSV is exactly what
the code produces.
