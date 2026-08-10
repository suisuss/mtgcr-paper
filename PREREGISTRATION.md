# Pre-registration: mtgcr-paper, Round 1

**Date closed:** 2026-07-23 (this file is committed before any outcome-correlated
result is computed; the commit hash is the timestamp)
**Kit:** mtgcr-paper (cloned from paper-template)
**Consumers:** projects/zone-risk-early-warning.md (H1, Bar 2 baseline) and
projects/endpoint-consensus-geometry.md (Q1 functionals, 2a clustering) in the
parent SayHeya research directory. Per the amended charters (2026-07-23), both
projects' researcher degrees of freedom are fixed here, in one document, before
the shared clustering spike runs.

**Lane:** everything in this round is the descriptive lane, plus pre-registered
baseline curves for later predictive work. No confirmatory claim closes on this
round; its purpose is to decide which project graduates to a full pre-registered
study. Spike results are reported with the conventions below regardless of
direction.

## Data (fixed)

- Source artifacts: the actr-paper reproduction kit at commit
  `970ff72837f57292d4ce4d3e06bf61d3bd15eeca` (pinned in KIT_PROVENANCE.json;
  the loader verifies the sibling checkout's HEAD matches and refuses to run
  otherwise).
- Corpora: USS-SGD (primary, n=1000 conversations) and USS-MWOZ (replication,
  n=1000), MiniLM (all-MiniLM-L6-v2) 384-d L2-normalized turn embeddings from
  `data/embeddings/uss_{sgd,mwoz}_minilm.npz` + `_meta.json`.
- Raw turn text: `data/raw/uss_repo/dataset/{SGD,MWOZ}.txt` (tab-separated
  ROLE/text/action/ratings lines, blank-line-separated conversations). The
  parser must verify conversation count and per-conversation turn counts align
  with the meta file; misalignment aborts the run.
- Outcome variable: `overall_median` (median of 3 annotator conversation-level
  ratings) for band definitions; `overall_mean` for rank correlations.
- Bands: **bad** = overall_median <= 2.5; **good** = overall_median >= 3.5;
  middle excluded from band contrasts. Known imbalance (SGD: ~39 bad / ~270
  good) is accepted and reported; this is why band results here are descriptive
  with exact CIs, and why the balanced ToolBench arm (separate spike, gated on
  the pipeline port) remains the charter's primary H1 corpus.

## Spike 1 - shared clustering (zone H1 descriptive arm + endpoint 2a + Bar 2 baseline)

1. **Clustering:** k-means (scikit-learn KMeans, n_init=10), fit unsupervised
   on all turn embeddings of a corpus (labels unseen at fit time). Primary
   k=64, seed 42. Sensitivities: k=16 (seed 42) and seed 123 (k=64), reported
   as max-lift stability only.
2. **Zone lift (descriptive):** per cluster, the share of its band-conversation
   turns that come from bad-band conversations, vs the corpus base rate over
   band turns. Report lift = share/base with Jeffreys 95% intervals on turn
   counts AND a conversation-level bootstrap CI (resample conversations, 1000
   draws, seed 42). A cluster is "high-risk" at lift >= 2x base with the
   bootstrap CI excluding 1x.
3. **Position stratification (endpoint 2a):** cluster composition by normalized
   position tau quartile; endpoint geography = distribution over clusters of
   final-turn assignments, by band. Purely descriptive.
4. **Lead-time profile (descriptive):** for high-risk clusters, the tau
   distribution of bad-band conversations' first entry, and the share of
   entries occurring before the conversation's final 2 user turns. This is the
   H3-shaped readout; no early-warning claim closes on it in this round.
5. **Bar 2 baseline head (predictive baseline, OOF):** logistic regression
   (L2, C=1.0, class_weight=balanced, max_iter=2000) on the concatenated
   running per-role mean embeddings (768-d) at each user-turn prefix; label =
   conversation band. GroupKFold k=5 grouped by conversation, seed-fixed fold
   assignment (groups hashed in corpus order). Report OOF ROC-AUC per tau
   quartile.
6. **Zone-dwell comparator (predictive, leakage-controlled):** within each CV
   fold, rank clusters by lift computed on the train fold only; the dwell
   feature at turn t = share of turns <= t in train-fold high-risk clusters.
   OOF ROC-AUC per tau quartile, same folds as (5). The spike's H1-relevant
   readout is dwell-AUC vs head-AUC; per the amended charter, zones "earn the
   name" only if dwell is at least competitive with the head - in this round
   this is recorded, not adjudicated.

## Spike 2 - endpoint descriptives (Q1)

Distance is cosine distance (1 - dot) on the L2-normalized embeddings.
Degree-1 fits are per-role least squares of embedding vs the role's own
normalized turn position tau in [0,1]; the gap curve is the difference of the
two fitted lines evaluated on a common 101-point tau grid.

**Functional family (5 tests, fixed):**

- F1 raw final-turn distance: d(last user turn, last system turn)
- F2 fitted endpoint gap ||gap(1)|| (labeled derived, per the degree-1 caveat
  in the amended charter)
- F3 closest-approach depth: min over the tau grid of ||gap(tau)||
- F4 closest-approach location: argmin tau of ||gap(tau)||
- F5 initial gap ||gap(0)|| (control row for reading F2)

**Analysis:** Spearman rho of each functional vs overall_mean, per corpus;
Benjamini-Hochberg FDR at q=0.05 within the 5-test family per corpus. SGD is
primary; MWOZ is the replication arm.

**Closing-ritual controls (pre-registered, decisive for interpretation):**

- Ritual detector: case-insensitive substring match of the final user turn OR
  final system turn against the fixed list: {"thank", "thanks", "goodbye",
  "bye", "welcome", "good day", "great day", "nice day", "that's all",
  "that is all", "have a good", "appreciate"}. Covariate = binary.
- Partial correlation: Spearman via rank-residualization (rank-transform F1 and
  outcome, regress both on the ritual covariate, Pearson on residuals).
- Strip-k: recompute F1 after removing the final k user turns and final k
  system turns, k in {1, 2}; conversations left with < 2 turns per role are
  dropped and the surviving n reported.
- Interpretation rule (fixed): an F1 effect that fails the partial correlation
  AND loses > 50% of |rho| (or FDR significance) at strip-1 is reported as the
  ritual confound, not as convergence.

**Deferred:** intersection candidates 2b(b) sign changes and 2b(c) lead/lag
asymmetry are NOT run in this round (2b(c) additionally requires the
permutation null pinned in the amended charter). Only closest-approach (2b(a))
runs, as F3/F4 above.

## Spike 3 - sub-turn PRISM (pinned now, runs after spikes 1-2)

- Unit: every PRISM model response (all candidates, chosen or not) from
  `data/raw/prism/conversations.jsonl`; target = the response's own 0-100
  score.
- Segmentation (fixed): split on `.`/`!`/`?` followed by whitespace; keep
  segments >= 10 characters; include responses with >= 3 surviving segments.
  Report surviving n and state the long-response selection bias as a scope
  condition.
- Encoder: sentence-transformers all-MiniLM-L6-v2 (matches the flagship),
  L2-normalized; heavy dependency lives in requirements.txt only.
- Ablation ladder (per the amended charter; untyped case, so the typed rung
  collapses away): rung 1 = mean sentence embedding + n_sentences; rung 3 =
  + degree-1 drift vector over the sentence sequence; rung 4 = + causal
  surprise profile (prefix-fit, vector of per-step surprises summarized as
  the coefficient-free vector [mean, max, last] - recorded as a vector
  channel, no single scalar claim).
- Model: ridge regression (alpha selected on train folds from {0.1, 1, 10}),
  GroupKFold k=5 grouped by conversation. Metric: OOF Spearman.
- Keep criterion per rung (fixed): +0.02 OOF Spearman over the previous rung,
  paired bootstrap (1000 draws, seed 42) 95% CI excluding zero. Lift stopping
  at rung 1 vs 3 vs 4 is attributed per the charter's ladder rule.

## Spike 4 - ToolBench pipeline port (wiring only this round)

Fetch is gated behind `--fetch-data`; nothing downloads implicitly. The port
pins the turn-encoding scripts' revision in KIT_PROVENANCE.json and adds
`goal_text` (= answer_generation.query) per conversation and `goal_embedding`
(MiniLM) to the encodings. No ToolBench analysis is pre-registered here; the
ToolBench arms of all charters get their own pre-registration once the data
exists locally.

## Hardware sizing (recorded)

Single laptop: 16-core Ultra 9 285H, 62 GB RAM, RTX PRO 2000 laptop GPU.
Spikes 1-2 are CPU-only, minutes. Spike 3's encoding pass is the only
GPU-relevant step; CPU fallback is acceptable. ToolBench raw (~multi-GB,
~126K files) fits disk but is deliberately deferred.

## Evidence rule

Results close only on registry artifacts: committed scripts, recorded seeds,
`git_dirty=false` provenance. Sampled/sandbox runs never enter the registry.
