# Pre-registration: Round 15 (structure beyond the mean: order, goal, cross-channel)

**Date closed:** 2026-07-24 (committed before any label-touching run).

Round 14 located the ceiling. On the tool-result channel the raw payload
embedding beat a handful of parsed flags by only +0.012 AUC (Direction C): when
384 dimensions barely outperform nine booleans, the per-role *mean* has
extracted nearly everything it can from that channel. Enriching the tool
identity (Direction B) added nothing. So the remaining accuracy is not in richer
per-turn encodings - it is in the structure the running mean discards (**order,
recency, cross-channel alignment**) and in one fixed channel we have never used
as a feature at all: **the goal**.

This is the predictive lane: each arm ADDS fixed-channel features to the
drop-assistant base (fixed roles, AUC 0.769) and must show added signal against
a pre-registered kill/keep threshold. Two standing constraints frame it:

- **Un-gameability is inherited, not re-litigated.** Every R15 feature is a
  function of the fixed channels only (user, tool-call skeleton, tool-result,
  goal) - never the narration. So every arm is invariant under the round-9
  narration selection attack by construction (manipulability 0). Per R14's
  firewall that 0 is NOT quoted as robustness; R15 asks only whether the fixed
  evidence, read more cleverly, carries MORE accuracy. Robustness is inherited
  from round 13.
- **These beat the containment null only if they are not linear functionals of
  the degree-0 blocks.** actr B.5 predicts linear functionals of the role means
  are null, and every scalar graft so far is 0-for-4. R15's features are
  nonlinear (cosines), different statistics (last / recency-weighted / deltas),
  or a new channel (goal) - outside the null's scope. Running them is the
  sanctioned graft that extends the ledger either way.

Base for every comparison: **drop_assistant** (the round-13 fixed-role head,
0.769). Fold, encoder, and regularization identical to rounds 13-14 (GroupKFold
by conversation, fold 0; MiniLM; LogisticRegression C=1.0, class-weight
balanced). Bootstrap 95% CIs are conversation-level, B = 1000. **Keep margin is
0.02 AUC with disjoint CIs, frozen here for all three directions.**

## Direction A (primary): order and recency over the tool-result channel

The running mean of results `[fail, fail, recover]` approximately equals the
mean of `[recover, fail, fail]`, but their outcomes are opposite. Three added
features, all functions of the fixed result sequence:

- **last**: the most recent tool-result embedding (not the average).
- **recency**: an exponentially recency-weighted result mean, decay lambda =
  0.5 (frozen).
- **delta**: the consecutive-result difference at the current step
  (trajectory improving vs degrading).

Arms: `base` (drop_assistant), `+last`, `+recency`, `+delta`, `+all`.

**Negative control (the confound bar):** an `+all_shuffled` arm that recomputes
the same order/recency features after permuting the tool-result order within
each conversation (seed 42). Order features must beat their own shuffled version
- otherwise they encode magnitude, not order.

**Prediction P1 (keep):** at least one order/recency arm beats `base` by >= 0.02
AUC with disjoint CIs, AND beats `+all_shuffled` by >= 0.02. Then order in the
result channel carries signal the mean discards - the actr shape hypothesis,
null on the interleaved all-role trajectory, is alive on the tool-result
sub-trajectory specifically.

**Kill P1:** if no arm clears both bars, degree-0 on the results is confirmed
near-optimal; we stop dressing the tool-result channel and pivot effort to
Direction B. Recorded as a registered null.

## Direction B (primary): goal-conditioned alignment

`goal_text` is present in ToolBench (and ABCD carries flow/subflow) and has
never been a feature. Embed the goal once per conversation; add, at each
decision point:

- **goal_result**: cosine(goal, tool-result running mean) - does the returned
  evidence address what was asked.
- **goal_progress**: the change in goal_result alignment over the trajectory -
  is the conversation moving toward the goal region.

Arms: `base`, `+goal_result`, `+goal_progress`, `+goal_all`.

**Negative control (topicality confound):** a `+goal_mismatch` arm using a
*different* conversation's goal (a fixed derangement, seed 42). Own-goal
alignment must beat mismatched-goal alignment by >= 0.02 - otherwise the feature
is generic topic-similarity / verbosity, not goal-specific.

**Prediction P2 (keep):** `+goal_result` (or `+goal_all`) beats `base` by >=
0.02 with disjoint CIs AND beats `+goal_mismatch` by >= 0.02. This is the dense,
continuous form of the task-conditioning lever that bought "un-gameable in
substance" (same-task slack 0.00 vs any-task 0.82, §16.3).

**Substance cross-check (secondary):** re-run the ABCD action-substitution
adversary (`policy_choice`-style: substitute actions, recompute the true label)
against the goal-conditioned head. Prediction: the any-task slack falls below
its current 0.82, because the goal-conditioned score penalizes actions that
succeed at *a* task but not *this* goal. If the slack does not move, dense
goal-conditioning does not add substance over the binary switch.

**Kill P2:** if own-goal alignment does not beat mismatched-goal, or adds
nothing over `base`, dense goal-conditioning fails and the binary
task-conditioning captured something the continuous cosine cannot.

## Direction C (secondary): cross-channel consistency among fixed roles

A linear head over concatenated role means *cannot* form cross-channel
interactions; cosines between channels are therefore genuinely new features, all
fixed (no narration). Round 12's "relational binding is gameable" does not apply
- that argument was about binding the *adversary-written* channel; these bind
only fixed channels.

- **request_result**: cosine(user mean, tool-result mean) - does the evidence
  match the ask (real customer channel on ABCD; synthetic on ToolBench - noted).
- **call_result**: cosine(tool-call skeleton mean, tool-result mean) - did the
  invoked capability return something coherent.
- **repetition**: fraction of tool calls repeating a previously-called tool id
  (a stuck-loop / give-up tell, from the skeleton sequence alone).
- **novelty**: running mean of (1 - cosine(current result, previous result)) -
  are results changing or thrashing.

Arms: `base`, `+consistency` (all four scalars), and `+repetition` alone
(near-free, tested standalone as a classic give-up signal).

**Prediction P3 (keep):** `+consistency` beats `base` by >= 0.02 with disjoint
CIs; and `+repetition` alone carries standalone signal above chance. Because the
linear head cannot synthesize these interactions from the means, a lift here is
information genuinely added, not reparametrized.

**Kill P3:** if `+consistency` adds nothing over `base`, the linear combination
of role means already suffices and cross-channel structure is redundant - a
registered null that sharpens what the degree-0 head is implicitly capturing.

## Scope

ToolBench is the primary corpus (clean independent label, goal_text present).
ABCD carries the real customer channel (Direction C) and the action-computable
label for the Direction B substance cross-check; its label is action-derived, so
its fixed-role accuracy is contaminated (noted since round 13) - it is the
robustness/substance corpus, not the clean accuracy anchor. All arms are
narration-invariant, so manipulability is 0 by construction and not reported as
robustness. Everything is still text corpora; voice process channels (prosody,
silence, hang-up) and MCP tool descriptions remain the standing data-gated gaps.

## Evidence rule

As rounds 1-14: seed 42, GroupKFold by conversation, out-of-fold, registry
artifacts with `git_dirty=false`, `./run.sh check` green. Conversation-level
bootstrap CIs (B = 1000). Decay lambda, the 0.02 keep margin, and the two
negative controls (order-shuffle, goal-mismatch derangement) are frozen here and
not revised after seeing outcomes. No result counts from an unclean tree or an
unregistered run.
