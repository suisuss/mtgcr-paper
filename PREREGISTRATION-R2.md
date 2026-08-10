# Pre-registration: mtgcr-paper, Round 2 (ToolBench arms)

**Date closed:** 2026-07-23 (committed before any label-touching analysis of
the ToolBench encodings; the commit hash is the timestamp)
**Data:** the round-1 ToolBench port at the pins in KIT_PROVENANCE.json:
12,598 conversations, 115,945 MiniLM turn embeddings, one goal embedding per
conversation. **Observed failure rate 20.2%** (recorded here against the
charter's ~34% figure, which came from the old Drive-zip subset).
**Label:** `has_failure` (success = valid_data AND finish_type=give_answer,
inverted). Distance is cosine distance (1 - dot) on L2-normalized embeddings.

**Lane:** descriptive + pre-registered predictive baselines, as round 1. The
three arms run sequentially off one shared checkpointed feature stage
(`tb_features`), ordered cheapest-first: endpoint -> typed ladder -> early
warning.

## Shared feature stage (tb_features, label-blind)

Per conversation, from turn embeddings + turn/step types + goal embedding
(+ raw tool_result text from conversations.json for the keyword baseline):

- Type partition: user / assistant / tool_call / tool_result; thought =
  assistant turns tagged step_type=thought.
- Conversation-level: overall mean embedding; n_turns; per-type mean
  embedding and per-type degree-1 slope vector (least squares over the
  type's own normalized positions; zeros + presence flag when the type has
  < 2 turns); thought-recurrence vector [mean pairwise cos, max pairwise
  cos, n_thoughts] (zeros when < 2 thoughts); per-type mean cosine distance
  to goal (4 values) + overall mean distance to goal.
- Prefix-level (early-warning samples): one sample at every **tool_result**
  turn (the natural step-completion points). Features: concat of the 4
  per-type running means over turns <= t (1536-d). Baseline columns:
  running error-keyword share over tool_result texts <= t (fixed list:
  error, invalid, not found, fail, timeout, exceeded, unauthorized,
  forbidden, rate limit, 400, 401, 403, 404, 429, 500, 502, 503;
  case-insensitive substring); running cosine distance of the assistant-type
  running mean to goal; step count t; tau = t/(n_turns-1).

This stage never reads `has_failure`.

## Arm 1 (runs first): goal-distance endpoints (endpoint charter, ToolBench)

Functional family (5 tests, fixed):

- E1 d(final turn, goal)
- E2 d(final thought turn, goal) - conversations with >= 1 thought
- E3 d(final tool_result turn, goal)
- E4 min over turns of d(turn, goal) (closest approach to goal)
- E5 tau location of that minimum

Analysis: per functional, ROC-AUC against has_failure with Mann-Whitney p;
BH-FDR q=0.05 within the 5-test family. Registered direction: failures end
and stay *farther* from goal (AUC > 0.5 scoring failure as positive with
higher distance). This is the endpoint project's last arm; a null here
parks the charter.

## Arm 2 (runs second): typed sub-turn ladder (sub-turn charter, ToolBench)

Classifier: logistic regression (L2, C=1.0, class_weight=balanced,
max_iter=2000), stratified 5-fold CV, seed 42, OOF ROC-AUC. Rungs
(cumulative):

- R1 content + length: overall mean embedding + n_turns
- R2 + per-type mean embeddings + presence flags (typed content)
- R3 + per-type degree-1 slope vectors (typed structure)
- R4 + thought-recurrence vector + per-type goal-distance vector (dynamics
  + goal coupling; vector channels only, per the scalarization ban)

Keep bar per rung (fixed): +0.01 OOF AUC over the previous rung with a
conversation-level bootstrap (1000 draws, seed 42) 95% CI excluding zero.
Attribution per the amended charter: lift stopping at R2 is typed content,
not deliberation structure; the reasoning-monitoring line opens only on
R3/R4 lift. Tool-result content encodes error text; that is *content* and
belongs to R2 by construction.

## Arm 3 (runs last): zones vs head early warning (zone charter, ToolBench)

Identical protocol shape to round-1 spike 1, on balanced-ish labels:

- k-means k=64, seed 42, n_init=10, fit unsupervised on all turns
  (checkpointed). Lift per cluster over **all** conversations (no band
  construction; the label is binary), Jeffreys CIs + conversation-level
  bootstrap (1000, seed 42); high-risk = lift >= 2x with bootstrap CI
  excluding 1x. Sensitivities: k=16 and seed 123, max-lift/count only.
- Endpoint geography TV + permutation null (1000), as round 1.
- Lead-time profile for high-risk clusters; "early" = first entry strictly
  before the final 2 tool_result turns.
- OOF comparison, GroupKFold k=5 by conversation, AUC per tau quartile at
  the prefix samples: **head** (logistic on the 1536-d prefix features) vs
  **zone dwell** (train-fold-ranked high-risk clusters, as round 1) vs the
  pre-registered baselines scored raw: error-keyword share (Bar 1), running
  goal distance, step count. Bar 1 = head and dwell must beat the keyword
  share at lead time (Q1-Q3); Bar 2 = dwell vs head decides whether the
  zone framing survives, per the amended charter.

## Evidence rule

As round 1: registry artifacts only, fixed seeds, git_dirty=false;
`./run.sh check` green before any arm's numbers are read as results.
