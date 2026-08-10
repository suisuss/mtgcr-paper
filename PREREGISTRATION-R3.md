# Pre-registration: mtgcr-paper, Round 3 (full experiments on the survivors)

**Date closed:** 2026-07-23 (committed before any of these analyses runs; the
commit hash is the timestamp)
**Scope:** the debts and follow-ups registered by rounds 1-2, run as full
experiments on the existing ToolBench (and USS) artifacts. Five stages, one
new checkpoint producer (corrected error flags), everything else reading
existing checkpoints. Corpus, label, and distance conventions as
PREREGISTRATION-R2.md.

## Stage A - corrected error baseline (tb3_error; repairs the void Bar 1)

The round-2 keyword baseline saturated on ToolBench's literal JSON "error"
key. The corrected per-tool_result **true-error flag** (fixed; contents are
500-char-truncated so JSON parsing is unreliable - regex on the raw string):

    true_error = any of:
      r'"error"\s*:\s*"[^"]'        (non-empty error value)
      "status_code=4", "status_code=5"
      "HTTPSConnectionPool", "Function executing from"
      "Timeout", "timed out"        (case-insensitive for the last pair)

Running true-error share at each prefix sample (as R2). Readouts: AUC per
tau quartile; **Bar 1 test** = head AUC minus true-error-share AUC per
quartile, conversation-level bootstrap (1000 draws, seed 42) 95% CI.
Bar 1 passes for the head in a quartile iff the CI excludes zero in the
head's favor. Per-turn flags are checkpointed for stages B-D.

## Stage B - E1/E3 error-content confound (tb3_confound)

- Conversation-level flag: final tool_result has true_error.
- Pure error classifier: that flag alone -> AUC vs has_failure.
- Stratified endpoint AUCs: E1 and E3 recomputed within (i) conversations
  whose final tool_result has NO true error, (ii) those where it does.
- **Interpretation rule (fixed):** within the no-error stratum, E1/E3
  AUC < 0.55 -> the endpoint result is an error detector; AUC >= 0.60 with
  MWU p < 0.001 -> goal distance carries signal beyond error content;
  between -> inconclusive, reported as such.

## Stage C - goal-hugging mechanism tests (tb3_goalhug)

The three follow-ups registered in the round-2 report, plus the descriptive
curve:

- (a) **Quote-the-query artifact:** recompute the running assistant-mean
  goal distance at each prefix excluding the k most recent assistant turns
  (k in {1, 2}); AUC per quartile. Kill rule: if exclusion moves the Q2-Q4
  AUCs into [0.45, 0.55], the inversion is a recency-quotation artifact.
- (b) **Group stratification:** inversion AUC per quartile within G1, G2,
  G3 separately (compound-goal blend check).
- (c) **USS transfer proxy:** goal proxy = first user turn embedding;
  feature = running SYSTEM-role mean cosine distance to the proxy at each
  user-turn prefix (round-1 spike-1 prefix protocol); AUC per tau quartile
  against the bad/good bands on SGD and MWOZ. Two-sided question: does the
  inversion transfer?
- (d) **Descriptive curve:** mean assistant-turn-to-goal distance by tau
  decile, failures vs successes (the picture the claim lives or dies on).

## Stage D - the head as study object (tb3_head)

Recompute the OOF head (round-2 protocol: logistic C=1 balanced, GroupKFold
5 by conversation, 1536-d prefix features), this time persisting per-sample
OOF scores. Then:

- **Calibration:** reliability curve (10 equal-width bins) + ECE, overall
  and per tau quartile.
- **Alert protocol** (the zone charter's precision/lead-time readout, now
  head-based): for each threshold on the OOF-score grid (percentiles 50-99),
  alert fires at the first prefix with score >= threshold. Conversation-level
  metrics: precision P(failure | alerted), recall, false-alarm rate
  P(alert | success), median lead time among true positives in tool_result
  steps remaining and in (1 - tau) terms. Report the full curve; highlight
  the pre-registered operating points: the lowest thresholds achieving
  precision >= 0.5 and >= 0.7.

Alert metrics are conversation-level (did it ever fire, and when), which is
the censoring-honest framing from the amended zone charter.

## Stage E - canonical Gram port + equivalence (gram_equivalence)

Port `gram_fit` / `gram_basis_matrix_fast` verbatim from gp-paper
(`src/lib/encoders/gram.py`) into this kit's `lib/gram.py`. Equivalence
checks, pass criterion max |delta| < 1e-4:

- per-type g0 vs the type mean embedding (all ToolBench conversations);
- degree-1 fitted curve from Gram coefficients vs the lstsq line used by
  rounds 1-3, evaluated on the 101-point tau grid (all ToolBench per-type
  sequences with >= 2 turns, and the USS-SGD role gap curves of spike 2).

Pass closes the "canonical parameterization" debt: the stand-in and the
published instrument are the same object at degree <= 1. Fail (any delta
>= 1e-4) is investigated before any further analysis builds on the fits.

## Evidence rule

As rounds 1-2. Seeds 42 throughout; bootstraps 1000; registry artifacts
only; `./run.sh check` green before numbers are read.
