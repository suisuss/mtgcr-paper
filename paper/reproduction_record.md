<!-- src: tags name the producing script; regenerate with `python main.py`; artifact at data/results/experiments/<script-stem>.json, provenance in its _provenance block. Invisible when rendered. -->
# mtgcr-paper, Round 1: Shared Turn-Point Clustering and Endpoint Descriptives

```{=latex}
\noindent{\small\textbf{Jacob Sussmilch}}\\[1pt]{\footnotesize Heya}\vspace{0.4em}
```

Spike round for the SayHeya research program: the shared turn-point clustering spike (zone-risk H1 descriptive arm, endpoint 2a geography, the Bar 2 prefix-embedding baseline, and a leakage-controlled zone-dwell comparator) and the endpoint descriptive functionals (Q1) with the closing-ritual controls, on USS-SGD (primary) and USS-MWOZ (replication). All researcher degrees of freedom were fixed in PREREGISTRATION.md before any outcome-correlated result was computed. Everything here is the descriptive lane; no confirmatory claim closes on this round. Its output is a decision: which project charter graduates to a full pre-registered study.

---

## 1. Setup

Data are the actr-paper kit's committed artifacts at the pinned commit in KIT_PROVENANCE.json: MiniLM 384-d L2-normalized turn embeddings for USS-SGD and USS-MWOZ (1,000 conversations each), 3-annotator satisfaction labels, and the raw USS turn text (alignment verified against the meta files at load time). Bands: bad = overall_median <= 2.5, good = >= 3.5, middle excluded. The known band imbalance (tens of bad conversations per corpus) is why this round is descriptive with exact intervals, and why the balanced ToolBench arm remains the charters' primary H1 corpus once the data pipeline port lands.

## 2. Spike 1: shared clustering

k-means (k=64, seed 42) fit unsupervised on all turn embeddings. Zone lift is the per-cluster share of band turns from bad-band conversations against the corpus base rate, with Jeffreys intervals on turn counts and a conversation-level bootstrap. High-risk = lift >= 2x with the bootstrap CI excluding 1x. Lead-time and endpoint-geography readouts are descriptive. The Bar 2 baseline is an out-of-fold logistic head on running per-role mean embeddings at every user-turn prefix; the zone-dwell comparator scores test folds using high-risk clusters ranked on the train fold only.

<!-- table: 01_zone_lift -->
| Corpus | Bad/good convs | Base rate | Max lift [boot 95% CI] | High-risk clusters | First-entry tau (median) | Early share |
|---|---|---|---|---|---|---|
| SGD | 39/270 | 0.106 | 2.37 [0.92, 4.12] | 0 | - | - |
| MWOZ | 48/174 | 0.202 | 2.74 [1.27, 3.73] | 2 | 0.24 | 1.00 |
<!-- src: paper_tables.py -->

<!-- table: 02_zone_auc -->
| Corpus | Signal | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| SGD | prefix-embedding head | 0.521 | 0.541 | 0.570 | 0.630 |
| SGD | zone dwell | 0.522 | 0.510 | 0.517 | 0.521 |
| MWOZ | prefix-embedding head | 0.596 | 0.709 | 0.723 | 0.720 |
| MWOZ | zone dwell | 0.487 | 0.552 | 0.572 | 0.587 |
<!-- src: paper_tables.py -->

## 3. Spike 2: endpoint descriptives

Five pre-registered functionals of the role-split degree-1 gap curve and the raw final turns, Spearman against overall_mean with BH-FDR (q=0.05) within the family, per corpus. The closing-ritual controls (fixed keyword detector, rank-residualized partial correlation, strip-k recomputation) decide whether any F1 effect is convergence or a goodbye detector.

<!-- table: 03_endpoint -->
| Corpus | Functional | rho | p | FDR pass |
|---|---|---|---|---|
| SGD | F1_raw_final | -0.067 | 0.0341 | yes |
| SGD | F2_fitted_end | -0.167 | 0.0000 | yes |
| SGD | F3_closest_depth | -0.160 | 0.0000 | yes |
| SGD | F4_closest_tau | -0.009 | 0.7877 | no |
| SGD | F5_initial_gap | -0.178 | 0.0000 | yes |
| MWOZ | F1_raw_final | -0.039 | 0.2237 | no |
| MWOZ | F2_fitted_end | -0.082 | 0.0098 | yes |
| MWOZ | F3_closest_depth | -0.105 | 0.0009 | yes |
| MWOZ | F4_closest_tau | -0.012 | 0.7097 | no |
| MWOZ | F5_initial_gap | -0.088 | 0.0056 | yes |
<!-- src: paper_tables.py -->

<!-- table: 04_ritual -->
| Corpus | Ritual share | F1 rho | F1 partial rho | F1 strip-1 rho (n) | F1 strip-2 rho (n) |
|---|---|---|---|---|---|
| SGD | 0.91 | -0.067 | -0.054 | -0.005 (1000) | -0.002 (1000) |
| MWOZ | 0.94 | -0.039 | -0.061 | +0.063 (1000) | +0.053 (997) |
<!-- src: paper_tables.py -->

## 4. Spike 3: sub-turn ladder (PRISM, untyped)

Every PRISM model response with >= 3 surviving sentence segments (56,032 responses across 7,984 conversations; the >= 3-segment inclusion and its long-response selection bias are the pre-registered scope condition), sentences encoded with all-MiniLM-L6-v2, target = the response's own 0-100 score. Ridge with alpha selected on train folds, GroupKFold by conversation, OOF Spearman; rung deltas judged by conversation-level paired bootstrap against the pre-registered keep bar (+0.02, CI > 0).

<!-- table: 05_subturn -->
| Rung | OOF Spearman | Delta vs previous [95% CI] | Keep |
|---|---|---|---|
| R1_content_length | 0.1905 | - | - |
| R3_plus_drift | 0.1875 | -0.0029 [-0.0066, +0.0008] | no |
| R4_plus_surprise | 0.1883 | +0.0008 [+0.0002, +0.0015] | no |
| n | 56032 responses / 7984 conversations (median 5 sentences) | | |
<!-- src: paper_tables.py -->

## 5. Round 2: ToolBench arms

The three surviving threads, run sequentially off one label-blind checkpointed feature stage per PREREGISTRATION-R2.md, on the ported ToolBench data (12,598 conversations, 20.2% failure rate, goal embeddings first-class).

### 5.1 Goal-distance endpoints

<!-- table: 06_tb_endpoint -->
| Functional | AUC | p | FDR pass | n |
|---|---|---|---|---|
| E1_final_turn | 0.730 | 9.97e-283 | yes | 12598 |
| E2_final_thought | 0.579 | 8.27e-28 | yes | 7651 |
| E3_final_tool_result | 0.722 | 1.64e-263 | yes | 12598 |
| E4_min_goal_dist | 0.458 | 3.63e-11 | yes | 12598 |
| E5_min_tau | 0.500 | 1.00e+00 | no | 12598 |
<!-- src: paper_tables.py -->

E1/E3 are large and in the registered direction: failed conversations end far from their stated goal (AUC 0.73). Two functionals are **degenerate by construction** and are reported as such: the goal text is itself turn zero of every conversation, so the minimum goal distance (E4) is pinned near zero for everyone and its tau location (E5) is constant at the start - E4/E5 should have excluded user turns, an under-specification of the pre-registration, not a finding. E1/E3 also carry a content confound to respect: a failing conversation's final turns are often error blobs or give-up prompts, which are far from any goal text; "ends far from goal" and "ends in error content" are not yet separated.

### 5.2 Typed sub-turn ladder

<!-- table: 07_tb_ladder -->
| Rung | OOF AUC | Delta vs previous [95% CI] | Keep |
|---|---|---|---|
| R1_content_length | 0.8629 | - | - |
| R2_typed_content | 0.8403 | -0.0226 [-0.0291, -0.0166] | no |
| R3_typed_structure | 0.8388 | -0.0015 [-0.0082, +0.0057] | no |
| R4_dynamics_goal | 0.8433 | +0.0045 [+0.0031, +0.0059] | no |
<!-- src: paper_tables.py -->

The typed arm completes the sub-turn charter's kill branch. Content plus length alone reaches OOF AUC 0.863; adding typed content *hurts* (-0.023, CI entirely negative - a capacity cost at fixed C=1, itself a boundary statement); typed structure adds nothing; dynamics + goal coupling adds +0.0045, statistically nonzero but half the +0.01 bar. Combined with round 1's powered PRISM null, the pre-registered decision rule fires: **structure-beyond-content is null at every granularity tested, typed and untyped.** The reasoning-monitoring line does not open on trajectory geometry over these corpora; the scope boundary the flagship started is now complete at the sub-turn level.

**Scope correction (round 8).** This null is *against a degree-0 / binary target* - the mean-pooled response score or the failure label. Round 7's channel diagnostic (11.1) shows the same low-degree coefficients carry real signal against degree-*matched* shape targets (satisfaction trend 0.42, recovery 0.47, both-roles). So the honest boundary is narrower than "structure adds nothing": structure adds nothing *to a target that has already discarded shape by construction*. The degree-matched version of this test is answered positively at conversation level (round 7) and is not runnable at sub-turn granularity for lack of any public per-sub-turn human ratings - a data limit, not a null.

### 5.3 Zones vs head, balanced labels

<!-- table: 08_tb_zone -->
| Base rate | Max lift [boot 95% CI] | High-risk clusters | Entering convs | First-entry tau (median) | Early share | Geography TV (perm p) |
|---|---|---|---|---|---|---|
| 0.224 | 2.75 [2.49, 3.00] | 5 | 702 | 0.00 | 0.86 | 0.546 (0.000) |
<!-- src: paper_tables.py -->

<!-- table: 09_tb_auc -->
| Signal | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| head | 0.705 | 0.799 | 0.813 | 0.834 |
| dwell | 0.579 | 0.581 | 0.592 | 0.598 |
| err_share | 0.500 | 0.500 | 0.500 | 0.500 |
| goal_dist | 0.498 | 0.299 | 0.319 | 0.364 |
| step_count | 0.500 | 0.492 | 0.536 | 0.496 |
<!-- src: paper_tables.py -->

With balanced labels the zones finally exist: 5 high-risk clusters with bootstrap CIs clear of 1x, and a huge endpoint-geography separation (TV 0.546 against a permutation p95 of 0.07). But the median first entry is tau = 0: much of the "zone" signal is *query-type risk* - hard, compound requests plot in risky regions from turn zero - not a conversation degrading. And Bar 2 fails for the third time in a row: the prefix-embedding head beats dwell in every quartile (0.71 rising to 0.83 vs ~0.58-0.60), now by a wide margin. The early-warning signal is real, early, and semantic - and it lives in the discriminative head.

Two baseline notes, reported plainly. The **error-keyword baseline is degenerate as implemented**: every ToolBench tool_result is a JSON blob containing the literal key "error", so the share saturates at 1 and the Bar 1 comparison is void this round - a corrected detector (non-empty error values / status codes) is a protocol amendment for the next pre-registration, and until it runs, "beats the keyword baseline" cannot be claimed. **step_count is flat** (~0.5 per quartile): at fixed tau, length carries nothing - the head's signal is not length in disguise. The third baseline inverted, and earns its own section.

### 5.4 The goal-hugging inversion

**What was measured.** The pre-registered running-goal-distance baseline: at every tool_result prefix, the cosine distance between the running mean of the conversation's *assistant-type* turns (thoughts and plain assistant messages only - user turns, tool calls, and tool results are excluded by the type partition) and the conversation's goal embedding, scored raw against has_failure. It was registered as a baseline the head should beat, with the implicit expectation that failures drift *away* from the goal.

**What came back.** The opposite, and not subtly (table 09): AUC 0.498 in Q1, then 0.299 / 0.319 / 0.364 across Q2-Q4. Scoring failure-as-positive with higher distance, an AUC that far *below* 0.5 means failing conversations sit systematically **closer** to their stated goal than succeeding ones from mid-conversation on - effective discrimination up to
|0.5 - 0.30| = 0.20, appearing by the second quartile and persisting to
the end.

**What it is not.** Not length: step_count is flat at every quartile. Not the error-keyword artifact: that baseline is constant; this one varies per conversation and position. Not the E4/E5 turn-zero degeneracy: the assistant-type running mean never contains the user's query turn or the retry prompts that echo it - those are user-type turns, outside this feature by construction.

**Reading (interpretation, not a closed claim).** Successful trajectories leave request space: their assistant content moves toward answer content
- specific entities, retrieved data, results - which embeds away from the
phrasing of the ask. Failing trajectories circle the request: thoughts restate the goal, plan again, re-describe what they are about to try; the semantic center of the agent's own output never escapes the query's neighborhood. Staying near the goal text is grinding in place. This is consistent with the flagship's position-dominates result (where you sit in embedding space carries the outcome signal) and gives it a direction: *distance from the request is progress; proximity to the request is struggle.*

**Alternative mechanisms to kill before the claim hardens** (registered here as the follow-up's opening tests): (a) give-up trajectories may quote the query in their final thoughts, mechanically pulling the mean goalward - test by excluding the last k assistant turns; (b) compound G3 queries embed as topic blends, possibly compressing all distances - test by group-stratified replication; (c) the effect may be ToolBench-specific - the USS corpora have no goal field, so the transfer test needs a goal proxy (first user turn) or Heya call data with stated intents.

**Why it matters for the program.** It is the sharpest concrete instance yet of the essay's warning about naive reward shaping: a "goal-proximity = progress" shaping term would *reward the failure signature*. It also hands the Goodhart arm its first named exploit candidate in the opposite direction ("flee the query without answering it"), and it gives the early-warning head its most interpretable input channel. This finding is a post-hoc reading of a pre-registered measurement: the measurement stands at descriptive-lane strength; the mechanism claims above are hypotheses until the follow-up runs.

**Correction (round 3) — the interpretation above did not survive its own follow-up.** The registered mechanism tests found the inversion to be an imputation artifact riding on a thought-presence confound; see section 6.3. The measurement was real; the "proximity is struggle" reading is retracted for ToolBench.

## 6. Round 3: full experiments on the survivors

Five pre-registered stages (PREREGISTRATION-R3.md) closing the round-2 debts: the repaired Bar 1, the endpoint confound, the goal-hugging mechanism tests, the head as study object, and the canonical Gram port.

### 6.1 The repaired Bar 1, and the head's operating numbers

The corrected true-error flag (non-empty error values, HTTP status patterns, connection failures) fires on 6.6% of tool_results - a real baseline, unlike the saturated keyword share.

<!-- table: 10_bar1 -->
| Quartile | Head AUC | True-error AUC | Diff [95% CI] | Bar 1 |
|---|---|---|---|---|
| Q1 | 0.705 | 0.556 | +0.149 [+0.128, +0.169] | pass |
| Q2 | 0.799 | 0.592 | +0.206 [+0.193, +0.221] | pass |
| Q3 | 0.813 | 0.625 | +0.187 [+0.172, +0.200] | pass |
| Q4 | 0.834 | 0.686 | +0.147 [+0.134, +0.160] | pass |
<!-- src: paper_tables.py -->

Bar 1 passes in every quartile with the bootstrap CI clear of zero: the head is not an error detector in disguise. Its deployment shape:

<!-- table: 13_head_ops -->
| Operating point | Threshold | Precision | Recall | False-alarm rate | Median lead (steps) | Median lead (tau) |
|---|---|---|---|---|---|---|
| precision >= 0.5 | 0.745 | 0.510 | 0.573 | 0.139 | 2 | 0.58 |
| precision >= 0.7 | not reachable | | | | | |
| overall | AUC 0.802 | ECE 0.173 | | | | |
<!-- src: paper_tables.py -->

At the reachable operating point the head catches 57% of failures with half its alerts correct, a 14% false-alarm rate, and a median lead of 2 tool steps (58% of the conversation remaining). Precision 0.7 is not reachable on raw scores, and ECE 0.173 says the balanced-class scores are far from probabilities - calibration (Platt/isotonic on a held-out fold) is mandatory before any threshold is deployment-meaningful.

### 6.2 The endpoint result survives its confound

<!-- table: 11_confound -->
| Functional | All AUC | No-error-final AUC (n) | Error-final AUC (n) | Verdict |
|---|---|---|---|---|
| E1 | 0.730 | 0.746 (10638) | 0.598 (1960) | signal-beyond-error |
| E3 | 0.722 | 0.748 (10638) | 0.522 (1960) | signal-beyond-error |
| pure error classifier | 0.671 | | | |
<!-- src: paper_tables.py -->

The registered rule returns **signal-beyond-error** for both functionals: among the 10,638 conversations whose final tool_result carries no error, the goal-distance AUC *rises* to ~0.75. "Ends far from goal" is not an error-content detector; it is the strongest surviving goal-relative construct in the program.

### 6.3 The goal-hugging inversion was an artifact - retraction

<!-- table: 12_goalhug -->
| Variant | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| base | 0.467 | 0.542 | 0.548 | 0.562 |
| excl-1 | - | 0.767 | 0.546 | 0.540 |
| excl-2 | - | - | 0.740 | 0.546 |
| G1 | - | - | - | - |
| G2 | - | 0.519 | 0.519 | 0.529 |
| G3 | 0.200 | 0.535 | 0.530 | 0.539 |
| USS-SGD proxy | 0.484 | 0.501 | 0.472 | 0.470 |
| USS-MWOZ proxy | 0.529 | 0.528 | 0.462 | 0.433 |
<!-- src: paper_tables.py -->

The mechanism tests killed the finding, and the autopsy is exact. The round-2 feature imputed distance 1.0 ("maximally far") for prefixes with no assistant-type turns yet - 54.3% of all samples, but 61.0% of success-conversation samples versus 27.9% of failure-conversation samples, because **92.7% of failing conversations contain explicit thought turns against 71.5% of succeeding ones**: ToolBench's DFS agent emits reflection text when things go wrong. Reproducing the feature with imputation restores the round-2 curve exactly (0.498/0.299/0.319/0.364); excluding the imputed samples leaves a mild, *non-inverted* signal (0.54-0.56), and the tau-decile curves show failures slightly farther from goal at every decile. The USS transfer proxy is null. What round 2 read as "failures hug the query" was "failures deliberate in text, and absence-of-deliberation was encoded as distance." The residual true observation - thought-presence itself predicts failure (AUC 0.61) - is a generation-process property of ToolBench (reflection is *caused by* failure there), not a geometric finding. Section 5.4's interpretation is retracted; its follow-up protocol is what caught it, one round later.

### 6.4 The canonical parameterization debt is closed

<!-- table: 14_gram -->
| Check | Max abs delta |
|---|---|
| g0 vs type mean | 0.00e+00 |
| degree-1 curve vs lstsq line | 1.94e-16 |
| USS-SGD gap-norm curves | 7.77e-16 |
| verdict (tol 1e-4, 4866 sequences) | PASS |
<!-- src: paper_tables.py -->

The ported gp-paper Gram projection and the rounds' lstsq stand-in agree to machine precision at degree <= 1, across 4,866 typed ToolBench sequences and the USS-SGD gap curves. Every fit in rounds 1-3 was the published instrument's object all along.

## 7. Reading

**Spike 1 (zones).** SGD produces no high-risk cluster that survives the conversation-level bootstrap (max lift 2.37, CI includes 1x); MWOZ produces two (max lift 2.74, CI [1.27, 3.73]), and the 20 bad-band conversations that enter them do so early (median first-entry tau 0.24, every entry before the final two user turns). But the pre-registered Bar 2 comparison is one-sided everywhere measured: the prefix-embedding head beats zone dwell in every tau quartile on both corpora (MWOZ Q2: 0.709 vs 0.552). On these corpora the zone framing does not earn its name - whatever early signal exists is better read directly by a discriminative head on the same running positions. The endpoint-geography readout is the one genuinely zone-flavored survivor: final-turn cluster distributions differ by band beyond the permutation null on both corpora (SGD TV 0.325 vs perm p95 0.163, p ~ 0.001; MWOZ 0.301 vs 0.274, p ~ 0.02). Caveats stand: tens of bad conversations per corpus; the balanced ToolBench arm remains the real H1 test.

**Spike 2 (endpoints).** The functional family splits exactly along the lines the amended charter predicted. The fitted functionals (F2, F3, F5) all pass FDR on both corpora with small negative rho - but F5, the *initial* gap, is as strong as F2, the endpoint gap (SGD: -0.178 vs -0.167). The gap's overall level carries the correlation, not anything endpoint-specific - this is "stays close" (gap_mean) re-read, consistent with the degree-1 containment caveat. The one genuinely new quantity, raw final-turn distance (F1), is weak (SGD -0.067, MWOZ n.s.) and the pre-registered ritual controls kill it: strip one closing exchange and it vanishes on SGD (-0.005) and flips sign on MWOZ (+0.063). Per the fixed interpretation rule, F1 is reported as the closing-ritual confound, not as convergence. "Ends close" on USS is a goodbye detector.

**Spike 3 (sub-turn geometry).** The untyped branch of the additivity question dies cleanly. Content plus length reaches OOF Spearman 0.19 against the response's own score; the drift vector *subtracts* (-0.003, CI spanning zero) and the surprise profile adds +0.0008 - statistically nonzero and forty times short of the pre-registered +0.02 keep bar. With 56K responses the delta CI is ~±0.004 wide, so this is a powered null, not a shrug: within-response structure beyond content carries nothing on PRISM, exactly matching the flagship's conversation-level result one granularity down. Per the charter's decision rule the reasoning-monitoring line does not open on untyped data; the typed arm (ToolBench thought/action/observation) is the hypothesis's last stand.

**Round-1 decision input.** Endpoint consensus (Q1) has no USS-side construct left to pursue: its unique content died under its own pre-registered controls, and what survives is already the flagship's "stays close" vocabulary. The sub-turn project's untyped branch is a powered null. The zone project survives in modified form: the early-warning signal exists but lives in the discriminative head, not in density zones - the honest continuation is the early-warning-calibration framing (head-based, per the amended charter's Bar 2 clause) tested where the labels are balanced. Every surviving thread now runs through the same gate: the ToolBench pipeline port (with the goal field), which carries the zone/head early-warning test on balanced labels, the typed sub-turn arm, and the goal-distance baselines. That port is the round's single remaining enabler.

**Round-2 decision (the convergence).** Across both rounds, three charters resolve: the sub-turn project closes on its own kill branch (null at every granularity, typed and untyped); the endpoint project's surviving content is goal-relative, not consensus-geometric - "ends far from goal" (E1/E3, AUC 0.73) folds into the early-warning story rather than standing alone; and the zone project's framing dies (Bar 2 failed three of three) while its *question* - dense semantic early warning - is answered emphatically by the prefix-embedding head (0.71 to 0.83 across the call, not length, not the degenerate keyword artifact). The program's one live instrument after two rounds is: **a prefix-embedding early-warning head with goal-relative features, calibrated per position** - plus the inverted goal-distance finding (failures hug the query; AUC 0.30-0.36) as its most interpretable channel *(superseded: retracted by the round-3 mechanism tests, section 6.3)*. The natural next investigations, in order: (1) a corrected error-value baseline (the standing Bar 1 debt) plus a same-prefix content-confound test of E1/E3; (2) the head as the object of study - calibration, lead-time precision curves, and the canonical Gram parameterization ported from the published instrument; (3) the Goodhart charter, which now has a concrete candidate signal worth attacking, on the harness this kit already provides.

**Round-3 decision.** Items (1) and (2) ran as round 3 (section 6). State of the program after three rounds: the early-warning head is the live instrument - Bar 1 passed legitimately in every quartile, operating numbers known (precision 0.51 / recall 0.57 / 2-step median lead at the reachable point), calibration the mandatory next fix (ECE 0.173). "Ends far from goal" (E1/E3, AUC ~0.75 in the no-error stratum) is the strongest surviving goal-relative construct and belongs in the head's feature set. The goal-hugging inversion is retracted (6.3) - the follow-up protocol registered at the moment of the finding is what caught it. The Gram debt is closed at machine precision. What remains: corpus expansion (the dataset survey accompanying this round) and the Goodhart arm, now aimed at the head itself.

## 8. Round 4: new corpora (ABCD, SWE-agent-trajectories)

The dataset survey's top two acquisitions, transformed to the normalized schema and run through the pre-registered battery (PREREGISTRATION-R4.md). ABCD: 6,399 labeled human-to-human customer-service dialogues (TRACER-style kb-query action-completion label, 13.4% failure). SWE: 11,983 code-agent trajectories across sampled instances (PR-test-verified label, 84.6% failure - a heavily failure-skewed training dump).

### 8.1 The escalation gate closes

<!-- table: 15_escalation -->
| Measure | Count |
|---|---|
| Conversations | 10042 |
| Manager notifications (canonical escalation) | 83 |
| Other-team notifications | 486 |
| Conversations with utterance-pattern hits | 174 |
| Viability bar / verdict | >= 300 / NOT VIABLE |
<!-- src: paper_tables.py -->

ABCD's canonical escalation action (notify-team with value "manager") appears in 83 conversations against the pre-registered bar of 300. Per the registered rule, the ABCD escalation early-warning study does not proceed; escalation work moves to Heya call data, where the events are real and plentiful.

### 8.2 The head replicates on dialogue, and fails on code agents

<!-- table: 16_ds_heads -->
| Corpus | Signal | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| ABCD | head | 0.699 | 0.742 | 0.786 | 0.812 |
| ABCD | goal_dist | 0.499 | 0.512 | 0.539 | 0.567 |
| ABCD | step_count | 0.495 | 0.491 | 0.495 | 0.493 |
| SWE | head | 0.505 | 0.508 | 0.519 | 0.522 |
| SWE | goal_dist | 0.565 | 0.622 | 0.664 | 0.684 |
| SWE | step_count | 0.655 | 0.709 | 0.711 | 0.712 |
| SWE | err_share | 0.568 | 0.534 | 0.525 | 0.525 |
<!-- src: paper_tables.py -->

The round's headline is this split. On ABCD - human-to-human dialogue with procedurally verified outcomes - the head reproduces the ToolBench curve almost exactly (0.699 rising to 0.812; length flat). Two corpora, two task ecologies, one instrument. On SWE code-agent trajectories the head collapses to chance (0.51-0.52) while the *baselines* carry the signal: step count reaches 0.71 (grinding long predicts failure) and running goal distance 0.68 (drifting far from the issue text predicts failure). The honest boundary statement: prefix-embedding early warning works where failures share a semantic signature across conversations (customer service, tool dialogues); on repository-specific engineering tasks the cross-instance transfer that the head depends on is absent, and process signals dominate. That is not a small caveat - it is the first corpus where the program's live instrument loses to a step counter, and it maps directly onto which deployment settings the instrument should be trusted in.

### 8.3 Endpoints do not travel

<!-- table: 17_ds_endpoints -->
| Corpus | Functional | AUC | FDR | No-error AUC (n) | Error AUC (n) |
|---|---|---|---|---|---|
| ABCD | E1_final_turn | 0.450 | yes | - | - |
| ABCD | E3_final_typed | 0.566 | yes | - | - |
| SWE | E1_final_turn | 0.422 | yes | 0.426 (8985) | 0.533 (2998) |
| SWE | E3_final_typed | 0.545 | yes | 0.555 (8985) | 0.585 (2998) |
<!-- src: paper_tables.py -->

"Ends far from goal" was ToolBench's strongest surviving construct (0.73, rising to 0.75 in the no-error stratum). It does not transfer: on ABCD E1 is weakly *reversed* (0.450) and on SWE clearly reversed (0.422, stable at 0.426 in the no-error stratum). Two readings, both recorded: ABCD's goal is a categorical template and every dialogue ends in the same closing ritual, so its endpoint semantics are weak by construction; SWE's reversal is more interesting - successful trajectories end in submission/patch content that embeds *away* from the issue text, while failures end still in the issue's neighborhood. Either way, the endpoint functional is corpus-specific vocabulary, not a portable definition of resolution. The charter's construct survives only as a per-corpus, per-goal-semantics measurement.

### 8.4 The ladder is null a third time

<!-- table: 18_swe_ladder -->
| Rung | OOF AUC | Delta vs previous [95% CI] | Keep |
|---|---|---|---|
| R1_content_length | 0.5944 | - | - |
| R2_typed_content | 0.5815 | -0.0129 [-0.0830, +0.0532] | no |
| R3_typed_structure | 0.5721 | -0.0094 [-0.0577, +0.0432] | no |
| R4_dynamics_goal | 0.5797 | +0.0076 [-0.0023, +0.0176] | no |
| assistant-share AUC | 0.500 | | |
<!-- src: paper_tables.py -->

Typed content subtracts, typed structure subtracts, dynamics adds under the bar - now on a third ecology (and note even content-plus-length only reaches 0.59 here; the instance-bootstrap CIs are wide because a few hundred instances carry the grouping). The assistant-share check is exactly 0.5 - SWE trajectories alternate strictly, so the ToolBench thought-presence confound structurally cannot exist here, which is what makes the ladder null clean. Structure-beyond-content is now null on PRISM (untyped), ToolBench (typed), and SWE (typed): the scope boundary is three-for-three - *against degree-0/binary targets* (see the round-8 scope correction in 4). All three ladders predict a mean or a failure label; none had a degree-matched shape target available, so this three-way null bounds structure's value *for content-degree targets*, not for shape.

### 8.5 Round-4 decision

The program's map after four rounds: **the early-warning head is a dialogue-domain instrument** - replicated on two corpora with verified outcome labels, bounded away from repository-style agent tasks where process signals (step count, goal drift) rule; **endpoint goal distance is per-corpus vocabulary**, not a portable resolution definition; **structure-beyond-content is closed** at every granularity and ecology tested; **ABCD escalation is not viable** (83 < 300) - the escalation question belongs to Heya data. What the map points at next: the voice-transfer arm (SpokenWOZ is fetched and pinned, awaiting its own pre-registration), the calibration fix the head still owes, and the Goodhart arm - which now knows exactly which instrument to attack and on which terrain it is strong.

## 9. Round 5: voice transfer (SpokenWOZ) and the Goodhart attack

### 9.1 The early-warning signal survives ASR transcripts

The program's standing highest-leverage question - does the dialogue-domain head survive *genuine spoken* transcripts (ASR noise, disfluencies)? SpokenWOZ: 5,697 labeled spoken dialogues, derived MultiWOZ-Success label (33.6% failure), the round-4 battery.

<!-- table: 19_spokenwoz -->
| Signal | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| head | 0.575 | 0.632 | 0.665 | 0.690 |
| goal_dist | 0.495 | 0.507 | 0.520 | 0.515 |
| step_count | 0.526 | 0.545 | 0.546 | 0.548 |
| E1/E3 endpoint AUC | 0.480 | 0.480 | | |
<!-- src: paper_tables.py -->

The head transfers: 0.575 rising to 0.690 across the call, the same monotone-in-tau shape seen on ToolBench and ABCD, cleanly above both baselines (goal distance ~0.51, step count ~0.55). It is attenuated - 0.69 at the end vs ~0.81-0.83 on the written corpora - which is the expected cost of ASR noise and spoken disfluency, not a collapse. This is the program's first evidence on real spoken transcripts, and it is positive: semantic early warning is a property of task-oriented dialogue, not an artifact of clean written text. (Endpoints are degenerate here - every SpokenWOZ dialogue ends on a system turn, so E1 and E3 are the same quantity; both land at 0.48, weakly reversed, consistent with the round-4 finding that endpoint goal distance does not travel.) The attenuation is the number to beat when the signal moves to Heya's own call transcripts - the real target, of which this is the closest public proxy.

### 9.2 The head is materially gameable by selection alone

The Goodhart arm - the anti-Goodhart charter's sanctioned sandbox violation. The head and its alert threshold are trained on fold-0 train; the attack substitutes each agentlike turn of a held-out *failing* conversation with the best of n real turns drawn from successful training conversations (n=1 = random, the drift control), and measures how much alert recall the attacker suppresses. The caller's failure content is never touched.

<!-- table: 20_goodhart -->
| Corpus | Base recall | n=1 (drift) | n=4 | n=16 | Degradation | Verdict |
|---|---|---|---|---|---|---|
| TOOLBENCH | 0.735 | 0.639 | 0.496 | 0.367 | 0.50 | graded |
| ABCD | 0.355 | 0.384 | 0.116 | 0.058 | 0.84 | gameable |
<!-- src: paper_tables.py -->

The head breaks under selection pressure, and the drift control is what makes the reading honest. On ABCD the n=1 random substitution leaves recall flat (0.355 -> 0.384, no drift), yet selecting the lowest-risk of 16 real agent turns collapses recall to 0.058 - an 84% suppression that is *entirely* selection, not distribution shift. On ToolBench the effect is graded: random substitution already costs ~0.10 (0.735 -> 0.639, some drift), and selection takes it to 0.367 - half the alerts suppressed, about three-quarters of that attributable to genuine selection. The attack never edits the failure; it only swaps in agent phrasings that have historically accompanied success. The white-box rung agrees: the minimal shift of the agent-mean block needed to cross the threshold at a median failing prefix is tiny (L2 ~0.012 on ToolBench, 0 on ABCD where most failures already sit below the precision-0.5 threshold), and the nearest real turns to the exploit direction are ordinary success-associated agent utterances, not adversarial noise.

The consequence for the program is direct and was the whole point of running it: **the early-warning head is an offline triage and monitoring instrument, not a training target.** Optimizing a policy against it would teach the model to emit success-shaped agent turns while the call still fails - exactly the Goodhart failure the essay names. TRACE claimed "a reward signal for LLMs" without running this experiment; we ran it on our own candidate and it is gameable by selection alone. That measured result is the firewall the anti-Goodhart stance is built on, now with a number on it.

### 9.3 Round-5 decision

Five rounds in, the program's instrument and its limits are mapped. The early-warning head is real across four corpora (ToolBench, ABCD, SpokenWOZ, and by extension the written-dialogue domain), it survives transfer to genuine spoken transcripts at attenuated strength, and it is materially gameable under optimization pressure - so it is a monitoring and triage signal, never a reward target, precisely as the containment prior required. What remains for the program is what only Heya can supply: the same head on Heya's own call transcripts (the attenuation measured here is the yardstick), real escalation events (ABCD's density failed the bar), and downstream CRM outcome labels (the resolution truth no public corpus carries). The public-corpus phase has done its job - built the instrument, found its shape, and proved where it breaks.

## 10. Round 6: the predecessor design's reward-arm matrix (verified-label judge)

The predecessor design proposes reward *governance via channel-factorizable features*: the thesis that per-role (per-position-basis) features admit a governance layer - score one channel, monitor another, tripwire a third - that relational suites (TRACE) structurally cannot support. Its rung-1 comparator matrix (floor = dense mean-pool, foil = TRACE-relational, method = Gram role-split, governed = Gram + channel tripwire) was never run; its own rung-1 spike found the LLM judge is *more* arc-biased than the reward is hackable, compromising the metric. This round runs the minimum discriminating matrix with the one instrument the predecessor design could not use: the **verified outcome label as the cross-modal judge** (ToolBench `target`, ABCD TRACER-rule - non-embedding-spoofable, so selection over turns cannot move it). Selection-pressure manipulability (round-5 harness: best-of-16 substitution of agentlike turns from the successful-conversation pool, n=1 random drift control), reported per the predecessor design's matched-predictor rule as an (accuracy, manipulability) pair.

<!-- table: 21_goodhart_matrix -->
| Corpus | Arm | Accuracy AUC | Base recall | n=1 drift | n=16 recall | Manipulability | Tripwire caught |
|---|---|---|---|---|---|---|---|
| TOOLBENCH | floor (dense g0) | 0.785 | 0.643 | 0.564 | 0.289 | 0.55 | - |
| TOOLBENCH | foil (TRACE) | 0.674 | 0.034 | 0.000 | 0.000 | 1.00 | - |
| TOOLBENCH | method (Gram) | 0.788 | 0.735 | 0.639 | 0.367 | 0.50 | - |
| TOOLBENCH | governed (Gram+trip) | 0.788 | 0.735 | 0.639 | 0.500 | 0.32 | 66 |
| ABCD | floor (dense g0) | 0.735 | 0.145 | 0.256 | 0.047 | 0.68 | - |
| ABCD | foil (TRACE) | 0.522 | 0.070 | 0.180 | 0.035 | 0.50 | - |
| ABCD | method (Gram) | 0.760 | 0.355 | 0.384 | 0.047 | 0.87 | - |
| ABCD | governed (Gram+trip) | 0.760 | 0.355 | 0.384 | 0.174 | 0.51 | 22 |
<!-- src: paper_tables.py -->

Two results, both in the predecessor design's own terms.

**P1 (naked parity) holds, with the confound the predecessor design flagged made visible.** The naked dense-g0 and naked-Gram heads are comparably strong predictors (AUC 0.74-0.79) and comparably hackable by selection (manipulability 0.50-0.87) - all dense-ish embedding proxies fall to best-of-16 over real turns. The TRACE foil cannot be placed cleanly: it is a much weaker predictor here (AUC 0.52-0.67, base recall 0.03-0.07), so its manipulability (0.50-1.00) is uninterpretable - exactly the matched-predictor confound the predecessor design warned about, now empirical. TRACE's relational features, forced through the same head class and label, simply do not predict these verified failures well enough to be a reward at all.

**P2 (governance separates) is supported - the round's finding.** The channel tripwire, computable *only* from the factorizable Gram blocks (system-vs-user divergence and system-channel energy, monitored not optimized), catches a material share of the selection hacks the naked head misses: on ToolBench it cuts manipulability 0.50 -> 0.32 (66 suppressions caught, held-out recall recovered 0.37 -> 0.50); on ABCD 0.87 -> 0.51 (22 caught, recall 0.05 -> 0.17), at identical head accuracy. This is the first *run* evidence that the predecessor design's expressibility thesis has teeth: the same predictor, governed through a channel a relational suite cannot express, is materially harder to game.

**But governance does not close the gap, exactly as the predecessor design's ceiling predicted.** Residual manipulability is still 0.32-0.51: roughly half the selection hack is *on-manifold* - substituted real success-turns whose channel divergence and energy sit inside the training envelope - and the envelope tripwire is blind to it (consistent with round 5, where the nearest exploit turns were ordinary success-shaped utterances). So the governed geometry raises hack cost by ~40% and makes a chunk of hacks visible; it does not make them impossible. The residual is precisely what the predecessor design reserves the outcome anchor for.

**Net for both programs.** This confirms the convergence: geometry is a governance and triage layer on an outcome-anchored reward, never the backbone. The predecessor design reasoned to it; rounds 5-6 measured it - naked heads (geometric or relational) are gameable by selection alone, channel governance recovers ~40% of the loss but leaves an on-manifold residual, and the verified outcome label is the only arm with zero manipulability (it is the judge). The scope is the predecessor design's: selection is a weak adversary, so every manipulability number is a lower bound; the governed separation is a lower bound on the thesis, and it already clears zero.

## 11. Round 7: the shape-target reward, judged by per-turn humans

The one arm the predecessor design said it could not measure. It argues the dialogue-mean satisfaction label is the wrong reward target (degree-0, dynamics-blind) and the right one is *shape* - reward the satisfaction TREND (via g1_sys) and RECOVERY (via g2_sys). Its own rung-1 sandbagging spike then found its LLM judge is *more* arc-biased than the trend reward is sandbaggable, so it could not adjudicate shape-reward manipulability. USS ships the instrument the predecessor design named as the fix: `turn_ratings`, three human annotators per user turn. This round rewards the system-channel shape coefficients and judges with those per-turn human ratings, position-neutral (equal-weight mean) and peak-end (final-turn). Heads are Paper 3's degree-matched setup (level g0_sys->y0, trend g1_sys->y1, recovery g2_sys->y2, combo); selection is best-of-32 over OOF reward scores, 2000 trials. USS-SGD primary, USS-MWOZ replication.

### 11.1 The safety mandate costs most of the shape signal

<!-- table: 22_shape_channel -->
| Corpus | Target | System-only (safe) | User-only | Both roles (Paper 3) |
|---|---|---|---|---|
| SGD | trend (y1) | 0.216 | 0.423 | 0.416 |
| SGD | recovery (y2) | 0.139 | - | 0.474 |
| MWOZ | trend (y1) | 0.149 | 0.312 | 0.304 |
| MWOZ | recovery (y2) | 0.183 | - | 0.320 |
<!-- src: paper_tables.py -->

This is the round's cleanest result and it is a tension the predecessor design never quantified. The pipeline reproduces Paper 3 exactly with both roles (trend 0.416/0.304, recovery 0.474/0.320) - but the predecessor design mandates the SYSTEM channel only, to deny the reward any gradient on the user turns it could manipulate. System-only roughly halves the trend signal (0.216/0.149) and cuts recovery to a third (0.139/0.183), and on both corpora the trend signal the safe channel throws away is carried by the USER channel alone (0.423/0.312 - at or above the both-roles number). The shape the predecessor design wants to reward lives disproportionately in the channel it forbids rewarding. The manipulation-safe restriction is not free: it discards half to two-thirds of the very signal that motivated rewarding shape.

### 11.2 The safe shape reward does not track the human judge

<!-- table: 23_shape_probe -->
| Corpus | Reward | d judge_mean | d judge_final | d reward_own |
|---|---|---|---|---|
| SGD | level | -0.028 | -0.008 | +0.062 |
| SGD | trend | -0.052 | -0.143 | +0.091 |
| SGD | recovery | -0.004 | +0.028 | +0.079 |
| SGD | combo | -0.024 | -0.048 | +3.704 |
| MWOZ | level | +0.033 | +0.030 | +0.047 |
| MWOZ | trend | -0.005 | -0.114 | +0.074 |
| MWOZ | recovery | -0.008 | +0.102 | +0.069 |
| MWOZ | combo | +0.004 | +0.025 | +3.238 |
<!-- src: paper_tables.py -->

Each column is the change from random (N=1) to best-of-32 selection: the reward's own score always rises (selection works on its own scale); the question is whether the independent per-turn human judge follows. It does not, and the pattern is consistent across both corpora:

- **trend** is Goodhart-divergent: its own score climbs (+0.09/+0.07) while the human judge's *final-turn* rating drops sharply (-0.143/-0.114) and the position-neutral mean stays flat-to-down. Maximizing the safe trend reward selects conversations humans rated *worse* at the end. This refines the predecessor design's hypothesis: the failure is not the predicted sandbagging (a manufactured arc a peak-end judge would *prefer*) - the human judge_final falls, it does not rise. At the strength the safe channel achieves, the trend reward is simply decoupled from, and mildly anti-correlated with, true per-turn satisfaction.
- **recovery** is the one arm that shows the arc-preference the predecessor design feared, localized to the curvature target: judge_final rises (+0.028/+0.102) while judge_mean stays flat - it selects end-high, average-unchanged conversations, the dip-recover shape, now visible against a human judge.
- **combo** damage-controls: requiring good level halts the trend reward's end-state degradation (judge_final -0.048/+0.025 vs trend -0.143/-0.114), partial support for the predecessor design's "must satisfy all three at once" mitigation - but it stops the bleeding rather than producing genuine improvement (judge_mean still flat).

The matched-predictor caveat is the finding, not a confound to excuse: the safe shape heads are weak (0.14-0.22) *because* of 11.1's channel restriction, and a weak reward selects noisily against the judge. Safe and weak are the same property here.

### 11.3 Round-7 decision

The shape-target arm, measured for the first time against a per-turn human judge, does not rescue geometry as a reward backbone - it sharpens why geometry belongs in the governance tier. Rewarding the safe (system-only) shape is weak by construction (11.1) and, at that strength, does not track human satisfaction under selection; the trend reward actively degrades end-state satisfaction while inflating itself (11.2). The degree-structured combo is the least-bad shape reward - it resists the divergence - but it buys resistance, not alignment. Across seven rounds the conclusion is one object: a verified outcome anchor carries the reward; trajectory geometry - role-split blocks, endpoint distance, shape coefficients - is the legible, channel-factorizable *monitoring and governance* layer on top of it, valuable precisely where it is not the optimization target. That is where the essay, the containment prior, the predecessor design, and now seven rounds of measurement all meet.

## 12. Round 8: convergence-phase follow-ups

Two arms the convergence phase (rounds 6-7) motivated, plus the null-scope correction already applied to sections 4 and 8.4. Neither reopens a closed result; both test a target the earlier rounds did not.

### 12.1 The voice head is gameable too - governance weaker on noisy channels

The round-6 reward-arm matrix applied to SpokenWOZ - the deployment-relevant gap, since rounds 5-6 attacked only ToolBench and ABCD.

<!-- table: 24_spokenwoz_manip -->
| Arm | Accuracy AUC | Base recall | n=16 recall | Manipulability | Tripwire caught |
|---|---|---|---|---|---|
| floor (dense g0) | 0.615 | 0.538 | 0.159 | 0.70 | - |
| foil (TRACE) | 0.535 | 0.044 | 0.000 | 1.00 | - |
| method (Gram) | 0.638 | 0.636 | 0.174 | 0.73 | - |
| governed (Gram+trip) | 0.638 | 0.636 | 0.231 | 0.64 | 22 |
<!-- src: paper_tables.py -->

The rounds 5-6 pattern holds on genuine spoken transcripts: the naked head is gameable by selection (Gram manip 0.73), the channel tripwire recovers a share (0.73 -> 0.64, 22 caught), an on-manifold residual survives, and TRACE is again too weak a predictor to place (AUC 0.54, base recall 0.04). Two voice-specific notes. The head is a weaker predictor here (AUC 0.62-0.64 vs 0.74-0.79 on text) - the round-5 ASR attenuation, now seen from the attack side. And the tripwire recovers *less* than on text (12% relative, vs 36% ToolBench / 41% ABCD): noisier spoken channels make the divergence/energy envelope less discriminating, so channel governance is weaker exactly where the transcripts are messier. The governance layer transfers to voice, but attenuated - which sharpens, not softens, the standing conclusion that the outcome anchor must carry the reward.

### 12.2 Endpoints against shape targets: signal where the mean was null, but single-corpus

The endpoint charter's "ends close" died against the satisfaction mean (degree-0, ritual confound) and against binary outcomes. Re-asked against the degree-matched shape targets (trend, recovery), FDR-corrected across the 4x3 family (* = survives):

<!-- table: 25_endpoint_shape -->
| Corpus | Functional | vs y0 mean | vs y1 trend | vs y2 recovery |
|---|---|---|---|---|
| SGD | E1_raw_final | +0.005 | +0.061 | -0.127* |
| SGD | E2_fitted_end | -0.068* | +0.081* | -0.097* |
| SGD | E3_closest | -0.053 | +0.105* | -0.088* |
| SGD | E4_initial | -0.093* | +0.078* | -0.061 |
| MWOZ | E1_raw_final | -0.028 | +0.030 | +0.007 |
| MWOZ | E2_fitted_end | +0.001 | +0.027 | -0.026 |
| MWOZ | E3_closest | -0.001 | +0.029 | -0.061 |
| MWOZ | E4_initial | +0.015 | +0.006 | -0.048 |
<!-- src: paper_tables.py -->

The methodological point is vindicated and the empirical claim is not overstated. On SGD, endpoint functionals *do* predict satisfaction shape where they were null against the mean: the raw final-turn gap tracks recovery at -0.127 (FDR-clean) - conversations ending with the parties farther apart have more negative curvature, i.e. *ending closer tracks recovery*, the direction the pre-registration named. Trend and recovery each pick up several FDR-surviving cells. This is exactly the signal the degree-0 test structurally could not see, and it is why the null-scope correction was owed. **But it does not replicate on MWOZ** - every cell near zero, none surviving FDR - and the SGD magnitudes are tiny (|rho| 0.06-0.13). Per the program's own single-corpus caveat, this reopens the endpoint descriptive lane as a *question worth a powered study* (does endpoint proximity track satisfaction shape?), not a settled finding. The honest status: the degree-matched re-ask found what the degree-0 test missed, confirming the method; the effect is small and SGD-only, so the construct is a candidate, not a result.

### 12.3 Round-8 decision

Nothing here changes the program's shape; both arms sharpen it. Voice manipulability confirms the head is gameable and governance transfers but weakens on noisy channels - the anchor's load only grows for the voice deployment. The endpoint-shape re-ask vindicates the degree-matched methodology (signal appears where the mean-target test was blind) while being too small and single-corpus to bank - it is logged as an open descriptive question, and the paper's structure-beyond-content nulls are now correctly scoped to their degree-0/binary targets (sections 4, 8.4). The through-line is unmoved: verified outcome anchors the reward; geometry is the legible monitoring and governance layer, valuable off the optimization path.

## 13. Round 9: the tool-anchored representation

The gameability mechanism (rounds 5-8) is that the signal reads fluid NL position, which a policy moves without moving the outcome. Tool calls are the one transcript element a policy cannot spoof - the log records what was actually invoked. This round tests whether anchoring the representation to the non-spoofable tool IDENTITY reduces manipulability, on ToolBench (median 3 tool calls; tool *count* is null, so any skeleton signal is identity, not length). Three arms, round-6 attack (best-of-16 substitution of assistant thought turns; tool calls are non-substitutable):

<!-- table: 26_tool_anchored -->
| Arm | Accuracy AUC | Base recall | n=16 recall | Manipulability |
|---|---|---|---|---|
| A — flat NL (round-6 head) | 0.788 | 0.735 | 0.367 | 0.50 |
| B — skeleton only (invariant) | 0.706 | 0.357 | 0.357 | 0.00 |
| C — tool-anchored (tethered) | 0.788 | 0.759 | 0.382 | 0.50 |
<!-- src: paper_tables.py -->

The result is sharp, and it falsified the registered prediction (P1) in a way that is more useful than a confirmation would have been.

**The tool skeleton alone is a good, un-gameable detector (B).** Which tools were called - nothing else, no NL - predicts failure at AUC 0.706 (against the flat head's 0.788), and its manipulability is zero: the substitution attack cannot touch it, because it cannot change which tools fired. The intuition that motivated the round is correct: tool identity is a robust anchor carrying real predictive signal, `invalid_hallucination_ function_name` and its kin being non-spoofable failure markers.

**But tethering does NOT inherit that robustness (C vs A).** Bolting the skeleton onto the flat NL head left manipulability at 0.50 - unchanged from A. The registered prediction was that a head handed a predictive invariant channel would shift weight onto it and off the spoofable block; it did not. A logistic head maximizes accuracy, not robustness - so it keeps its weight on the highest-accuracy channel (the assistant NL block), and the attack still moves the score across the threshold through that block, with the un-gameable skeleton sitting right beside it unused for defense. Adding an invariant feature adds capacity; it does not remove the head's reliance on the gameable one.

**So robustness comes from RESTRICTING to the non-spoofable channel, not adding it.** The honest tradeoff is now quantified: the skeleton alone is completely un-gameable at AUC 0.706; the last 0.082 of accuracy (0.706 -> 0.788) is bought *entirely* with gameability - it is precisely the fluid NL channel, and taking it reopens the full 0.50 manipulability. There is no blend: you get the robustness of whichever channel the head is permitted to use.

**Scope, honestly.** B's manipulability is zero *against this adversary* - NL substitution, which cannot alter the tool log. A real policy chooses its own tool calls and could try to game the skeleton directly - but faking the skeleton means actually invoking tools, which then either succeed or fail in the real tool result. Gaming the non-spoofable channel requires real actions with real outcomes, which is exactly why the skeleton sits so close to the verified outcome anchor. The round therefore lands where the whole program does, now from the representation side: the robust signal is the one tied to committed actions, and the dense NL geometry that beats it does so only by being gameable. Anchor the reward to what was done; keep the geometry in the monitor.

### 13.1 Addendum: does the FULL role-split(1) change the tethering result?

The round-9 arms used the per-role running *means* - degree-0 Gram (g0). The flagship's recommended representation is role-split(1), which adds the per-role *drift* coefficient g1. Does anchoring the full role-split(1) with tool calls behave differently - does the drift channel let the head lean on the anchor? We recompute the flat and tethered arms with per-role [g0, g1] (closed-form degree-1 Gram, validated equal to the canonical fit), same fold, same attack.

<!-- table: 30_rolesplit1_anchor -->
| Representation | Accuracy AUC | Manipulability |
|---|---|---|
| degree-0 flat | 0.788 | 0.50 |
| degree-0 + skeleton | 0.788 | 0.50 |
| role-split(1) flat | 0.794 | 0.49 |
| role-split(1) + skeleton | 0.793 | 0.47 |
| skeleton only | 0.706 | 0.00 |
<!-- src: paper_tables.py -->

Both prior findings hold for the richer representation. The drift coefficient buys essentially no accuracy (0.794 vs 0.788) - consistent with the ladder nulls (structure-beyond-content is null against the failure label at every granularity) - and anchoring role-split(1) with the skeleton does NOT inherit the skeleton's robustness: manipulability stays at 0.47, a hair below the flat 0.49 and nowhere near the skeleton's 0. Tethering does not help *regardless of representation degree*: the "restriction, not addition" result is a property of how a discriminative head weights an added invariant channel (it keeps its mass on the higher-accuracy spoofable block), not an artifact of dropping the drift coefficient. Restriction to the skeleton remains the only route to manipulability 0.

## 14. Round 10: skeleton restriction on ABCD - the road's payoff

Round 9 found the skeleton robust but tethering ineffective; the honest consequence was "restrict to the non-spoofable channel, don't add it." Round 10 drives that down on ABCD - the *most*-gameable corpus (flat head manipulability 0.80-0.87) - restricting to the canonical agent-action channel (30 action slugs). The label caveat is load-bearing and was pre-registered: ABCD's TRACER label is *derived from the action sequence*, so skeleton accuracy risks label-definitional leakage; a count-only arm (D) was included to measure it.

<!-- table: 27_skeleton_abcd -->
| Arm | Accuracy AUC | Base recall | n=16 recall | Manipulability |
|---|---|---|---|---|
| A — flat NL (round-6 head) | 0.769 | 0.333 | 0.067 | 0.80 |
| B — action skeleton (invariant) | 0.643 | 0.313 | 0.313 | 0.00 |
| C — tethered | 0.771 | 0.347 | 0.060 | 0.83 |
| D — action count only (leakage yardstick) | 0.534 | 1.000 | 1.000 | 0.00 |
<!-- src: paper_tables.py -->

**The payoff, clean and structural.** Restricting to the action channel takes ABCD from 0.80 gameable (A) to **zero** (B), at accuracy 0.643 - un-gameable by construction, because the NL-substitution attack cannot change which actions were performed. On the corpus that fell hardest to selection, the action-restricted representation cannot be selected around at all. This is the constructive counterpart to every gameability result in the paper: the one channel a policy cannot fake without actually acting.

**The count baseline defused the circularity worry (P3), honestly.** The pre-registered fear was that ABCD's action-derived label would let the skeleton predict by reconstructing the label rule. It largely did not: action *count* alone (D) is near chance (0.534), so the skeleton's 0.643 is carried by *which* actions were performed, not how many - not the trivial count-leakage the label's coverage rule would produce. And the tell is that ABCD's skeleton accuracy (0.643) is *below* ToolBench's clean, independent-label skeleton (0.706, round 9): a tautological reconstruction would sit near the ceiling, not below the corpus whose label owes nothing to the actions. The residual identity-leakage is real but small; the clean accuracy anchor remains ToolBench, and ABCD corroborates rather than inflates.

**Tethering fails again (P2 replicated).** Adding the anchor to the NL head (C) leaves manipulability at 0.83, unchanged from A - a second corpus confirming round 9: a logistic head keeps its weight on the spoofable block; robustness needs restriction, not addition.

**The road, quantified across two corpora.** Restricting to the committed- action channel costs modest accuracy - ToolBench 0.788 -> 0.706 (~0.08), ABCD 0.77 -> 0.643 (~0.13) - and buys complete un-gameability against selection (manipulability -> 0) on both. The un-gameable detector retains 84-89% of the flat head's accuracy. Same scope as round 9: the zero is against the NL-substitution adversary; a policy choosing its own actions must *perform* them, with real downstream effects, to move the skeleton - which is exactly why the action channel sits flush against the verified outcome anchor. The road ends where the program began: the robust signal is the record of what was actually done, and the dense NL geometry that beats it does so only by being gameable.

## 15. Round 11: the policy-choice adversary

Rounds 9-10 showed the skeleton un-gameable by NL substitution. But a real policy chooses its own actions, and can move the skeleton. The claim under test: the skeleton is un-gameable *in substance* - the only way to make the actions look successful is to actually perform the successful actions - because the outcome is action-determined. Two arms.

### 15.1 How much do actions determine the outcome (both corpora)

<!-- table: 28_determinism -->
| Corpus | Base purity | Skeleton purity | Determinism gap | Coverage | Ambiguous |
|---|---|---|---|---|---|
| ABCD | 0.866 | 0.940 | +0.073 | 0.88 | 0.84 |
| TOOLBENCH | 0.798 | 0.831 | +0.033 | 0.37 | 0.63 |
<!-- src: paper_tables.py -->

Grouping by the *set* of distinct actions, the skeleton weakly-to-modestly determines the outcome, and the corpus contrast is as predicted: ABCD's gap (+0.073, coverage 0.88) exceeds ToolBench's (+0.033, coverage 0.37). ToolBench determines *less* and shares *less* because tool RESULTS, not tool identity, decide give-answer/give-up - the same tool set can succeed or error, real slack between calling the right tool and it working. The gaps are modest because the distinct-action *set* is a lossy abstraction (it discards order, count, and task); the sharper instrument is Arm 2, which uses the actual label rule.

### 15.2 The policy-choice attack, with the real label recomputed (ABCD)

Because ABCD's TRACER label is computable from the action sequence, a gamed sequence's true outcome is knowable without executing anything. The adversary replaces a failing conversation's actions with the best of 16 real successful-conversation action sequences (chosen to fool the round-10 skeleton detector); we recompute the real label on the result.

<!-- table: 29_policy_choice -->
| Substitution pool | Detector fooled | Fooled & succeeded | Slack (fooled, not succeeded) |
|---|---|---|---|
| same-subflow | 0.960 (144) | 144 | 0.000 (0) |
| any-subflow | 1.000 (150) | 27 | 0.820 (123) |
<!-- src: paper_tables.py -->

This is the decisive result of the whole skeleton road, and it is two-sided. **Same-subflow: fooling the detector requires actually succeeding.** When the policy must draw its success-shaped actions from the *same task*, 144 of 150 attacks fool the detector - and every one of them *also flips the real TRACER label to success* (slack exactly 0). At the task level the skeleton is un-gameable in substance: the only way to make the actions look successful at this task is to perform this task's required actions, which is success. You cannot fake the work without doing the work.

**Any-subflow: a task-BLIND detector is wide open.** When the policy may copy success-shaped actions from *any* task, it fools the detector every time (150/150) but produces real success only 27 times - **82% slack**, the detector suppressed while the task genuinely failed. A skeleton detector that does not know which task it is scoring is gamed by generic success-shaped action mimicry.

**The gap between the two rows (0 vs 0.82) is the value of task-conditioning, and it is the round's lesson.** The action channel sits flush against the verified outcome anchor *only when the detector is conditioned on the specific task*; a task-blind action detector reopens almost the entire exploit. This is the concrete design rule for a deployed action-anchored signal: it must be conditioned on what the call was supposed to accomplish (the goal / intent / ticket type), or a policy games it by performing generic resolved-looking actions that do not resolve *this* caller's issue - the "resolves what it should escalate" failure, arrived at from the reward side. The skeleton is the robust channel the whole program was looking for, and round 11 states its one condition: anchor to the actions, but score them against the task.

## 16. Round 12: relational binding fails — restriction is the only route

Rounds 9-10 showed additive tethering leaves the head gameable; the round-9 addendum showed this holds regardless of representation degree. The natural next idea (and the predecessor design's "binding by parameterization"): let the content channel enter *only* as a quantity measured against the non-spoofable tool grounding — never as free absolute position — so that gaming the content must fight the grounding. The mechanistic bet was that substituting success-shaped NL onto a real failing tool skeleton would *raise* the say-do discrepancy and thus the head's risk, making the exploit self-defeating. We tested two bindings: a relational head (content enters only as cos(assistant, skeleton) and companion cosines) and a gated head (content multiplied by a grounding- consistency gate). Same corpus, same selection attack.

<!-- table: 31_relational -->
| Representation | How NL enters | Accuracy AUC | Manipulability |
|---|---|---|---|
| flat g0 | free absolute | 0.788 | 0.50 |
| additive tether | free absolute + skeleton | 0.788 | 0.50 |
| relational | cosine-to-grounding only | 0.762 | 0.68 |
| gated | grounding-gated content | 0.757 | 0.67 |
| skeleton only | not at all | 0.706 | 0.00 |
<!-- src: paper_tables.py -->

**The hypothesis was wrong, cleanly.** Relational binding did not defend against the attack — it made the head *more* gameable (manipulability 0.68 and 0.67, above the additive tether's 0.50), while recovering only part of the accuracy (0.76 vs the skeleton's 0.71). The self-defeating-exploit intuition failed: the selection adversary can move the say-do cosine to a low-risk value directly, by picking assistant turns from the success pool whose alignment with the (fixed, failing) skeleton reads safe — and because the relational features are low-dimensional and content-derived, they are an *easier*, not harder, target than the 384-dimensional free g0. The gate is equally leaky: the adversary opens it (aligns the content with the skeleton) and moves the content through it. This confirms the predecessor design's own caveat — binding by parameterization binds the *head*, not the policy — and shows it fails even against a non-co-moving selection adversary that can only touch the NL.

**The boundary is now sharp.** Across every scheme that lets the content channel into the reward — free (0.50), additively anchored (0.50), relationally bound (0.68), gated (0.67) — the head is gameable; only *not admitting* the content at all (restriction to the skeleton) yields manipulability 0. The dense-NL accuracy above the skeleton's 0.71 is not obtainable by any combination scheme we tried without reopening gameability: the accuracy and the exploitability are the same channel. Restriction, not combination, is the route to a non-spoofable reward — and the remaining accuracy gap is closed not by cleverer feature engineering but by the one lever round 11 identified: conditioning the action reward on the task.

## 17. Round 13: split-roles restriction recovers the accuracy - the cake

Round 12 closed off transforming or combining the adversary-controlled channel. But the selection attack controls only ONE role - the assistant's thoughts; the user turns, the tool calls, and the tool *results* are fixed. The split-roles geometry lets us drop exactly the spoofable role and keep the rest. And the assistant content can be re-admitted safely if it enters only as its residual against a model of what the fixed roles say should have happened (conditional typicality - the principled version of round 12's failed cosine).

<!-- table: 32_skeleton_roles -->
| Representation | Accuracy AUC | Manipulability |
|---|---|---|
| flat (all roles, incl. assistant) | 0.788 | +0.50 |
| skeleton only (tool identity) | 0.706 | +0.00 |
| drop-assistant (fixed roles) | 0.769 | +0.00 |
| drop-assistant + skeleton | 0.769 | +0.00 |
| + conditional typicality | 0.782 | -0.10 |
<!-- src: paper_tables.py -->

This is the constructive result the road was circling. **Dropping only the adversary-written role recovers almost all the accuracy at zero manipulability**: the fixed roles (user reaction, tool calls, tool results) reach AUC 0.769 - 97% of the flat head's 0.788 - and are invariant under the attack, because none is assistant-controlled. The spoofable narration carried the gameability and almost no *unique* accuracy. And conditional typicality does better still: re-admitting the assistant content only as its residual against p(content | fixed roles) reaches 0.782 (essentially flat) and is not merely un-gameable but mildly **anti-gameable** (manipulability -0.10) - the substitution attack *raises* the alert rate, because success-pool content is atypical under a failing context, so trying to fool the residual makes it more atypical, not less. This is the self-defeating exploit round 12 reached for, obtained through a fitted conditional model rather than a raw cosine.

**Honest reading (outcome-adjacency).** The recovered accuracy leans on tool_result content - what the tools actually returned - which is outcome-adjacent on ToolBench. So the win is not that dialogue *shape* carries the signal; it is that the non-spoofable *evidence* (what the tools returned, how the user reacted) is nearly as predictive as the full head and un-gameable, and the agent's narration adds exploitability but little independent signal. For a reward, anchoring on that evidence is exactly right - it is what the reward should read. The scope is the substitution adversary: a policy that chooses its own tools influences the tool results, but only by really calling tools with real outcomes (round 11).

**The refined boundary.** Round 12's "accuracy and exploitability are the same channel" was too strong: the exploitability lives in the adversary- *written role*, not in content per se. Drop that role and ~97% of the accuracy remains, un-gameable; re-admit its content as conditional typicality and essentially all of it returns, still un-gameable. Combined with round 11 (condition on the task), this is a complete recipe for an action-anchored reward that recovers dense accuracy without gameability: score the fixed evidence, admit the agent's narration only as its consistency with that evidence, and condition the whole thing on the task.

### 17.1 Replication on ABCD

The construction replicates on the second corpus. The adversary writes the *agent* role; customer and action turns are fixed.

<!-- table: 33_skeleton_roles_abcd -->
| Representation | Accuracy AUC | Manipulability |
|---|---|---|
| flat (all roles, incl. agent) | 0.769 | +0.80 |
| skeleton only (action identity) | 0.643 | +0.00 |
| drop-agent (fixed roles) | 0.772 | +0.00 |
| drop-agent + skeleton | 0.774 | +0.00 |
| + conditional typicality | 0.773 | +0.00 |
<!-- src: paper_tables.py -->

The **manipulability replication is exact and clean**: flat is 0.80 gameable (ABCD's naked head is the most gameable in the paper); every restriction/typicality arm is invariant at 0. And dropping the agent role does not merely retain accuracy — it *exceeds* the flat head (0.772 vs 0.769) at zero manipulability, the same "narration carries the gameability, not the signal" pattern as ToolBench, here in its strongest form: the agent's narration was net-negative for accuracy and all of the exploitability.

Two honest differences from ToolBench. Conditional typicality is un-gameable here (manipulability 0) but *not* anti-gameable (the -0.10 was ToolBench- specific), and it adds essentially nothing over drop-agent (0.773 vs 0.774) — because the action channel already carries the label-adjacent signal, leaving no residual role for the agent-content typicality. And the accuracy magnitude is contaminated: ABCD's label is action-derived, so the fixed-role evidence includes the channel the label is computed from. ToolBench (independent label) therefore remains the clean accuracy anchor; ABCD confirms the structural result — restriction to the fixed roles is un-gameable and recovers the accuracy — on a second, differently-labeled corpus.

### 17.2 Encoder robustness (bge-large-en-v1.5)

Everything in the paper uses MiniLM (384-d). To confirm the key result is not an artifact of a weak encoder, we re-encode ToolBench with a strong one (BAAI/bge-large-en-v1.5, 1024-d) and re-run the round-13 arms.

<!-- table: 34_bge_robustness -->
| Representation | MiniLM AUC | MiniLM manip | BGE-large AUC | BGE-large manip |
|---|---|---|---|---|
| flat (all roles) | 0.788 | +0.50 | 0.802 | +0.45 |
| skeleton only | 0.706 | +0.00 | 0.712 | +0.00 |
| drop-assistant | 0.769 | +0.00 | 0.786 | +0.00 |
| + conditional typicality | 0.782 | -0.10 | 0.797 | -0.04 |

<!-- src: paper_tables.py -->

The result holds, and sharpens. The stronger encoder *raises* every accuracy (better embeddings, more signal) but leaves the manipulability structure intact: the flat head is still gameable (0.45), and drop-assistant (0.786) and conditional typicality (0.797) remain un-gameable - typicality still mildly anti-gameable. Under BGE-large, the un-gameable typicality head reaches 99% of the flat head's accuracy (0.797 vs 0.802) at negative manipulability. The construction is an encoder-independent property of *which role the adversary writes*, not a low-dimensional-embedding artifact.

## 18. Round 14: tool-result content, enriched skeleton, structured vs semantic

Round 14 stress-tests round 13's honest concession (the recovered accuracy leans on outcome-adjacent tool-result content) and asks whether the tool channel can be enriched the way a real deployment (MCP) would allow. Pre-registered in `PREREGISTRATION-R14.md`; two corpus facts were checked and recorded in its 2026-07-24 amendment *before* the run: (1) `metadata.success` is the conversation label copied onto every call (constant within-conversation, 1.000/0.000 by outcome) and is therefore banned as a feature — all signal here is parsed from the tool_result `content` text; (2) the corpus carries no free-text tool descriptions, so Direction B uses the enrichments that are deterministic functions of tool id and present in the data (tool name, argument-key set), with literal descriptions left as a stated gap.

### 18.1 Is drop-assistant just counting errors? (Direction A)

The missing ToolBench process baseline. Two running error-rate scalars scored by the same head at every decision point: `err_naive` (fraction of tool results with a non-empty `error` field) and `err_strong` (+ empty payload + a frozen failure-string set). Conversation-level bootstrap 95% CIs (B = 1000).

<!-- table: 35_r14_error_baseline -->
| Representation | Accuracy AUC | 95% CI |
|---|---|---|
| drop-assistant (fixed roles) | 0.769 | [0.746, 0.791] |
| err_naive (error field only) | 0.601 | [0.576, 0.624] |
| err_strong (+ empty + failure strings) | 0.649 | [0.624, 0.674] |

<!-- src: paper_tables.py -->

**KEEP.** drop-assistant (0.769, reproducing §17's number exactly on the same fold) beats the stronger error baseline by 0.120 AUC with disjoint CIs — far past the pre-registered 0.03 keep margin. Step 5's "97% of the head" is signal beyond error-counting, not a computable error rate wearing geometry. The error-rate scalar is weak at the decision-point level because early prefixes (one or two results in) carry little; the geometry does not have that problem. `err_strong` is retained as the ToolBench process baseline the paper had been missing (Table 2's dash).

### 18.2 Does enriching the tool-identity encoding help? (Direction B)

The skeleton embeds the service name once per tool. Two enrichments, each a deterministic function of tool id: `+ tool name` (the fuller `x_for_service` identifier) and `+ argument keys` (the aggregated parameter-key set). Evaluated in-domain (shared tools) and on a frozen held-out-tool split (tool ids hashed, ~20% held out; 7,423 pure-seen vs 459 pure-unseen conversations).

<!-- table: 36_r14_skeleton_enrich -->
| Skeleton variant | In-domain AUC | Held-out-tool AUC |
|---|---|---|
| service name (baseline) | 0.706 | 0.671 |
| + tool name | 0.705 | 0.654 |
| + argument keys | 0.654 | 0.600 |

<!-- src: paper_tables.py -->

**Null-to-negative.** Tool name is null in-domain (-0.002) and *worse* out of domain (-0.017); argument keys are worse in both regimes. The clean service name is already the best available tool-identity encoding — the fuller identifier adds noise and the argument-key set is too generic. The pre-registered "gain on held-out tools" (P2b) fails for both available enrichments. This does not test literal free-text descriptions (absent from the corpus); the MCP case — real descriptions and JSON-schemas — remains an untested Direction-B gap requiring the external ToolLLM registry.

### 18.3 Structured flag vs semantic payload (Direction C)

The tool result encoded as parsed flags (`struct`: errored / empty / failure- class / length, over the running results) vs its embedding mean (`semantic`) vs both, each on the tool-identity skeleton.

<!-- table: 37_r14_struct_semantic -->
| Tool-result encoding | Accuracy AUC | 95% CI |
|---|---|---|
| structured flags | 0.741 | [0.718, 0.764] |
| semantic payload | 0.753 | [0.729, 0.778] |
| both | 0.753 | [0.730, 0.778] |

<!-- src: paper_tables.py -->

**Nuanced null.** The semantic payload beats the structured flags by only 0.012 AUC (overlapping CIs), and `both` adds nothing over `semantic` — below the pre-registered 0.03 keep margin, so P3 is not met. But this is *conditional on a strong structured vector*: 3,385 tool results are real failures with an empty `error` field (the conv-12543 pattern in Figures 1-2), which a naive `isError` flag misses entirely — `err_naive` alone scores 0.601. The lesson is not "flags suffice" but "the naive protocol flag lies, and once you build the flag properly (empty-body + failure-string detection) it captures most of what the embedding does." A naive MCP `isError` reward would be gameable by tools that return 200-with-empty-body.

### 18.4 Firewall (Direction D)

The success-channel arms (Direction A error-rate, Direction C struct) are invariant under the round-9 narration-only selection attack by construction — that attack rewrites only assistant turns, which none of these features touch. That 0 is **not** action-robustness and is not quoted as reward robustness: these arms are predictor-lane. Rewarding tool success moves the attack surface to the action channel, where the ABCD action-substitution result is the evidence — same-task slack 0.00, any-task slack 0.82 (§16.3, `policy_choice_abcd.json`): a policy games a task-blind success signal with generic success-shaped actions 82% of the time. Any success-channel reward must face that adversary and task-conditioning, not the narration-only attack.

## 19. Round 15: structure beyond the mean (order, goal, cross-channel)

Round 14 showed the per-role mean has nearly exhausted the fixed channels. Round 15 tests whether the structure the mean discards — order, the goal channel, and cross-channel alignment — recovers more accuracy, all on the fixed (un-gameable) channels, added to the drop-assistant base (0.769). Predictive lane, pre-registered in `PREREGISTRATION-R15.md`: keep margin 0.02 AUC with disjoint conversation-bootstrap CIs (B = 1000), negative controls frozen. Every arm is narration-invariant by construction, so robustness is inherited from round 13 and manipulability is not reported here.

### 19.1 Order and recency over the tool-result channel (Direction A)

The last result, an exponentially recency-weighted result mean, and the consecutive-result delta, versus an order-shuffle control that permutes only the already-seen prefix (leak-free — a full-sequence permutation would leak future results into early decision points).

<!-- table: 38_r15_order -->
| Arm | Accuracy AUC | Delta vs base |
|---|---|---|
| base (fixed roles) | 0.769 | - |
| + last result | 0.777 | +0.009 |
| + recency-weighted mean | 0.775 | +0.007 |
| + consecutive delta | 0.774 | +0.005 |
| + all three (ordered) | 0.777 | +0.008 |
| + all three (order shuffled) | 0.764 | -0.005 |

<!-- src: paper_tables.py -->

**KILL, with a signed nuance.** Order is not noise — the ordered features beat their leak-free shuffle by 0.013, and destroying order drops the arm *below* base (-0.005). But the total gain from order and recency is only +0.009 over base, well under the 0.02 keep margin, with overlapping CIs. The actr shape hypothesis, retried on the tool-result sub-trajectory specifically, carries a real but sub-threshold signal: degree-0 on the results is near-optimal.

### 19.2 Goal-conditioned alignment (Direction B)

The unused goal channel: the cosine of the embedded `goal_text` to the running result mean, and its trajectory, versus a mismatched-goal control (a fixed derangement) that isolates goal-specificity from generic topic similarity.

<!-- table: 39_r15_goal -->
| Arm | Accuracy AUC | Delta vs base |
|---|---|---|
| base (fixed roles) | 0.769 | - |
| + goal alignment | 0.773 | +0.004 |
| + goal progress | 0.770 | +0.002 |
| + goal (both) | 0.774 | +0.005 |
| + mismatched goal (control) | 0.768 | -0.000 |

<!-- src: paper_tables.py -->

**KILL.** There is a real goal-specific effect — own-goal alignment beats the mismatched-goal control by 0.006 (the control adds nothing, -0.000, confirming it is not mere topicality) — but at +0.005 over base it is far under the margin. The binary same-vs-any task-conditioning that bought "un-gameable in substance" (slack 0.00 vs 0.82, §16.3) does not translate into a dense accuracy gain: a continuous goal cosine is too coarse to sharpen the within-domain predictor, even though the discrete task identity governs substance.

### 19.3 Cross-channel consistency among fixed roles (Direction C)

Request-result and call-result cosines, tool-call repetition, and result novelty — interactions a linear head over concatenated means cannot form.

<!-- table: 40_r15_crosschannel -->
| Arm | Accuracy AUC | Delta vs base |
|---|---|---|
| base (fixed roles) | 0.769 | - |
| + cross-channel consistency | 0.774 | +0.005 |
| + tool-call repetition | 0.771 | +0.002 |

<!-- src: paper_tables.py -->

**KILL.** Cross-channel consistency adds +0.005, under the margin. Tool-call repetition alone reaches 0.527 AUC — weakly above chance (CI [0.509, 0.541] excludes 0.5), so stuck-loop repetition is a real give-up tell, but a faint one. The linear head over role means already captures nearly all of the cross-channel information these cosines make explicit.

### 19.4 What the round establishes

Three orthogonal attempts to exceed the fixed-role head — order, goal, and cross-channel structure — each add a real, correctly-signed, but sub-threshold signal (+0.005 to +0.009). None clears 0.02. The fixed-channel degree-0 representation is not merely near-optimal on the tool-result channel (round 14); it is near a ceiling across the whole fixed representation. Pushing ToolBench text accuracy past ~0.77 with un-gameable geometric features does not work — the remaining signal is in the verified outcome label itself (which is what a reward should anchor to), or in channels no text corpus carries: voice process signals (prosody, silence, hang-up) and real MCP tool descriptions.

## 20. Round 16: degree-1 role-split, and does velocity predict satisfaction?

*The satisfaction half of this round (the level-versus-velocity dissociation, 20.2-20.3) is being developed as a separate paper - see `PAPER2-level-vs-velocity.md`. It is retained here as the reproduction record of the run; only 20.1 (degree-1 is null for outcome) feeds the manipulability paper's ceiling claim.*

The paper uses degree-0 role-split (the per-role mean). Round 16 adds the degree-1 coefficient — the per-role *slope*, the velocity of each channel — and asks where it helps. Pre-registered in `PREREGISTRATION-R16.md`; the hypothesis is a dissociation: level geometry predicts whether a call resolves, velocity geometry predicts how it is going. Degree-1 on the fixed roles stays un-gameable; robustness is inherited. USS carries no tool structure and no raw text (VADER unavailable), so the same-turn-content baseline is the last USER-turn embedding — a stronger content baseline than a lexicon score (amendment).

### 20.1 Outcome (ToolBench, 4-role fixed)

<!-- table: 41_r16_outcome -->
| Representation | AUC | Delta vs base |
|---|---|---|
| base (degree-0 fixed roles) | 0.769 | - |
| + degree-1 slope | 0.774 | +0.006 |
| + slope (shuffled) | 0.765 | -0.003 |

<!-- src: paper_tables.py -->

**KILL.** Velocity adds +0.006 AUC over the degree-0 fixed-role head, under the 0.02 margin, and destroying order (shuffle) drops below base. Degree-1 confirms what round 15 predicted with a principled representation: for a binary outcome, degree-0 mean-pooling is sufficient.

### 20.2 Satisfaction (USS-SGD / USS-MWOZ, 2-role), per-turn Spearman

<!-- table: 42_r16_satisfaction -->
| Corpus / lead | base deg-0 | + degree-1 | slope-shuffled | same-turn content |
|---|---|---|---|---|
| SGD, lead 0 | 0.294 | 0.374 (+0.080) | 0.263 | 0.506 |
| SGD, lead 1 | 0.083 | 0.106 (+0.023) | 0.089 | 0.153 |
| MWOZ, lead 0 | 0.159 | 0.273 (+0.115) | 0.128 | 0.356 |
| MWOZ, lead 1 | 0.044 | 0.095 (+0.051) | 0.039 | 0.109 |

<!-- src: paper_tables.py -->

Two findings, and the discipline separates them.

**The asymmetry is real and large.** Degree-1 adds +0.080 (SGD) and +0.115 (MWOZ) Spearman over degree-0 role-split for satisfaction, versus +0.006 AUC for outcome — velocity carries an order of magnitude more signal for *how a call feels* than for *whether it resolves* (§20.3). Order is not noise: destroying it (shuffle) drops the correlation below the degree-0 base in every cell. The hypothesis that positional/velocity geometry matters is confirmed — and it matters for satisfaction, not outcome.

**But it does not clear the same-turn-content bar.** The last USER-turn embedding alone predicts satisfaction better than the whole role-split trajectory at both leads — 0.506 vs 0.374 (SGD lead 0), and still 0.153 vs 0.106 at lead 1, with MWOZ the same shape. So while velocity improves the *trajectory* representation substantially, the trajectory does not beat simply reading the current turn's content. Predicting the *next* rating (lead 1) is hard for everything (base rho ~0.04-0.08), degree-1's edge over degree-0 shrinks toward the margin, and the content baseline still leads. The pre-registered keep (degree-1 must beat base by 0.03 *and* survive the content bar at lead 1) is therefore not met: **KILL on the keep, KEEP as a finding** — per-turn satisfaction lives more in the current turn's content than in the shape of the trajectory, even though velocity is what the trajectory has to offer.

### 20.3 The dissociation, and the act channel

<!-- table: 43_r16_asymmetry -->
| Target | degree-1 gain over degree-0 |
|---|---|
| ToolBench outcome (AUC) | +0.006 |
| USS-SGD satisfaction (rho, lead 0) | +0.080 |
| USS-MWOZ satisfaction (rho, lead 0) | +0.115 |

<!-- src: paper_tables.py -->

The clean result of the round: **velocity geometry is a satisfaction signal, not an outcome signal** — a 13-to-20-fold larger degree-1 gain for satisfaction than for the binary label. Direction C (the dialogue-act channel, USS's coarse tool-outcome proxy: running NOTIFY_SUCCESS/FAILURE rate + act distribution) adds +0.046 on SGD and +0.001 on MWOZ at lead 0 — modest and corpus-dependent, as predicted for a proxy that is a per-turn tag, not a tool execution with a payload. What none of this could test, because no public corpus carries it, is velocity geometry over *real* tool-call dynamics against a satisfaction label — the intersection that only production voice data supplies.

## 21. Round 17: the two-mode deployment (running early-warning + terminal read)

Round 16 suggested a two-mode deployment rather than one representation: a running early-warning during the call, and a degree-0 level read at the end. Round 17 tests it on the leading method (the fixed-role degree-0 head), all on the un-gameable channels. Pre-registered in `PREREGISTRATION-R17.md`; the honest prior (from R15-16) was that the running mode is also degree-0.

### 21.1 Is there actionable lead? (Direction A)

At a conversation-level FPR-0.10 threshold (test FPR 0.12-0.15, a small generalization gap), the fixed-role head fires at each decision point; lead is measured on conversations with >= 3 decision points. Compared to a same-turn- content baseline (the current decision-turn embedding) at the identical operating procedure.

<!-- table: 44_r17_leadtime -->
| Corpus | recall | terminal-only | median lead | early-recall (>=2) | content early-recall |
|---|---|---|---|---|---|
| ToolBench | 0.493 | 0.448 | 2.0 | 0.370 | 0.287 |
| ABCD | 0.453 | 0.189 | 6.0 | 0.399 | 0.345 |

<!-- src: paper_tables.py -->

**KEEP on both corpora.** The running mode catches 37% (ToolBench) and 40% (ABCD) of *all* failures at least two decision points early — above the 25% bar and, crucially, above the same-turn-content baseline (0.287, 0.345), so the lead is genuine trajectory signal, not content arriving on schedule. Firing at any point rather than only at the terminal read adds +0.045 recall on ToolBench and +0.264 on ABCD, at a median lead of 2 and 6 decision points respectively — ABCD's longer dialogues give the running mode far more room, exactly as anticipated. The running mode is justified.

### 21.2 Does the running mode need degree-1? Is the terminal read degree-0? (B, C)

Degree-0 vs degree-1 fixed-role AUC, stratified by within-conversation position (early = first half of a conversation's decision points), with the terminal point isolated (ToolBench).

<!-- table: 45_r17_stratified -->
| Stratum | degree-0 AUC | degree-1 AUC | Delta |
|---|---|---|---|
| early | 0.747 | 0.748 | +0.001 |
| late | 0.803 | 0.813 | +0.010 |
| terminal | 0.837 | 0.848 | +0.011 |

<!-- src: paper_tables.py -->

**B KILL, and the disconfirmation is clean.** The sharp hypothesis — velocity helps *early*, before levels diverge — is not merely null, it is backwards: degree-1's (already sub-threshold) effect is *smallest* early (+0.001) and grows toward the terminal point (+0.011). Whatever tiny outcome signal velocity carries is a *late* effect, not an early one. The running mode is degree-0. **C confirm.** At the terminal point degree-1 adds +0.011, under the 0.02 bar, so the final judgment is a level read — "cut it down to degree-0 for the last utterance" is correct.

### 21.3 Verdict

**Two-mode, but two read-positions of ONE degree-0 head** — an early flag and a terminal level read, the same representation, not two. The deployment is validated in the sense that matters (there is real, content-beating early-warning value, especially on longer dialogues) while the "degree-1 running representation" idea is cleanly killed. This closes the arc of rounds 14-17: the fixed-role degree-0 head is the leading method, it is at its accuracy ceiling, and it is deployed as one representation read twice — flagged early during the call, read as a level at the end.

## 22. Round 18: the adversary-strength sweep to the selection ceiling

PREREGISTRATION-R18.md; `src/adversary_sweep.py`. The published manipulability numbers are best-of-16; R18 sweeps the selection adversary's budget `n` up to the class ceiling (exhaustive argmin over the entire fold-train success-turn pool). The typicality arm of this sweep was subsumed by R19's frontier (`res_2s`, the manuscript's Table 8); the **naive-head (`flat`) budget curve below is this round's alone** — no frontier arm carries it. The built-in nesting guard reproduced the published best-of-16 values exactly (flat 0.50, typicality -0.097).

ToolBench/MiniLM (clean-provenance rerun 20260725T114923; base recall 0.735, AUC 0.788):

| budget n | 1 (random) | 16 | 64 | 256 | exhaustive |
|---|---|---|---|---|---|
| naive head, recall after attack | 0.639 | 0.367 | 0.303 | 0.259 | 0.239 |
| naive head, manipulability | 0.131 | 0.500 | 0.587 | 0.648 | 0.675 |
| typicality residual, manipulability | -0.144 | -0.097 | -0.061 | -0.030 | +0.030 |

<!-- src: adversary_sweep.py -->

**The naive head degrades monotonically to +0.67 at the ceiling** — the published best-of-16 figure (+0.50) understates its exploitability, exactly as the lower-bound framing predicts. The fixed-role arms are 0 at every budget by construction (the writable block is absent from the feature; asserted, not swept). The typicality erosion (-0.14 -> +0.03) is the R19 frontier's subject.

Descriptive only (evidence-rule caveat): the ABCD and BGE-large replicas of this sweep ship as flat copies from a dirty-tree run (their provenance twins were pruned; only the ToolBench rerun is clean-stamped). Their pattern matches — naive-head ceiling 0.94 (ABCD, from 0.80 at best-of-16) and 0.661 (BGE, from 0.433); BGE typicality erodes -0.090 -> +0.005, ABCD stays 0.000 at every budget — but these cells do not close under the evidence rule and are quoted nowhere in the manuscript.

## References

USS: Sun et al., Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems. ABCD: Chen et al., NAACL 2021 (TRACER label rule: arXiv:2607.03974). SWE trajectories: nebius/SWE-agent-trajectories. SpokenWOZ: Si et al., 2023. Reward governance design and degree-matched shape targets: the predecessor design and Paper 3, both internal and not public. actr-paper: github.com/suisuss/actr-paper; the project charters it cites are internal.
