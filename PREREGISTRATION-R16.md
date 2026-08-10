# Pre-registration: Round 16 (degree-1 role-split, and does velocity predict satisfaction?)

**Date closed:** 2026-07-24 (committed before any label-touching run).

The entire paper uses **degree-0** role-split geometry - the per-role running
*mean*, the zeroth coefficient of the role-split polynomial fit. Degree-1 keeps
the mean AND the per-role **slope**: the direction each channel's sub-trajectory
moves through semantic space (velocity), which the mean discards. This is the
next Gram coefficient, already in the framework's vocabulary, and - being a slope
rather than a linear functional of the means - it is outside the containment
null (actr B.5), the same reason R15's features were legitimate to test.

Two priors frame it. (1) Round 15 probed a crude proxy for degree-1 dynamics -
last / recency / consecutive-delta over the tool-result channel - and it added
only +0.009 on ToolBench *outcome*, sub-threshold. A principled degree-1 role
fit is cleaner, but R15 is a real prior that velocity buys little for a binary
end-state. (2) The cell where velocity SHOULD pay off is a per-turn *dynamic*
target - **satisfaction** - not a conversation-final label. The core hypothesis:
**level geometry (degree-0) predicts whether a call resolves; velocity geometry
(degree-1) predicts how it is going - so degree-1 should help satisfaction more
than outcome.**

Degree-1 on the *fixed* roles stays un-gameable (the adversary cannot write
them); degree-1 on the narration would return the lever, so any assistant slope
enters only via the typicality residual, as in the current construction.
Robustness is inherited; this round is about accuracy on the un-gameable base.

## Data reality (checked before the run, frozen here)

- **ToolBench** carries the 4-role fixed structure (user / tool-call skeleton /
  tool-result) and a verified give-answer/give-up label. Degree-0 vs degree-1 is
  tested on the fixed roles. Base = drop_assistant (degree-0 fixed roles, 0.769).
- **USS-SGD / USS-MWOZ** carry per-turn human satisfaction ratings and two roles
  (USER, SYSTEM) plus a per-turn dialogue-act tag (INFORM, REQUEST, OFFER,
  NOTIFY_SUCCESS, ...). They have **no tool_call/tool_result turns and no service
  identity** - the tool-anchored construction does not transfer. The
  degree-0/degree-1 test therefore uses the two dialogue roles; the "tool"
  aspect is proxied only by the coarse dialogue-act channel (Direction C).
- **The clean intersection - rich tool-calling WITH satisfaction ratings - does
  not exist in any public corpus.** That intersection is exactly what production
  voice logs supply; its absence here is a stated gap, not a design choice.

**Degree-1 role-split, defined.** For role r with prefix embeddings e_1..e_k at
normalized positions x_j in [-1, 1], fit e = a0 + a1*x per dimension by least
squares; a0 is the degree-0 mean, a1 the degree-1 slope. Degree-1 feature =
concat(a0, a1) per role. **Short-role fallback (frozen):** if a role has < 2
turns in the prefix, its slope a1 = 0 (degree-0 only). Everything is causal -
computed on the prefix up to the decision/rated turn.

## Direction A (primary): degree-1 vs degree-0 for OUTCOME (ToolBench)

Arms on the fixed roles: `base` (degree-0, = drop_assistant), `+slope`
(degree-0 + per-role degree-1 slopes for user / tool_call / tool_result). Same
fold, encoder (MiniLM), regularization, and conversation-bootstrap CIs (B =
1000) as rounds 13-15. **Keep margin 0.02 AUC with disjoint CIs (frozen).**

**Negative control (frozen):** `+slope_shuffled` - refit the slope after
permuting each role's *already-seen* prefix order (leak-free; a full-sequence
permutation would leak future turns). Degree-1 must beat its own shuffle.

**Prediction P1 (expected KILL):** `+slope` adds < 0.02 over base, consistent
with R15. If so, degree-1 role-split is confirmed near-null for outcome and
degree-0 mean-pooling is settled as sufficient - now with the principled
representation, not R15's ad-hoc deltas. **Keep** only if `+slope` beats base by
>= 0.02 with disjoint CIs AND beats `+slope_shuffled` by >= 0.02.

## Direction B (primary): degree-1 vs degree-0 for SATISFACTION (USS)

Predict the per-turn USER satisfaction rating from the prefix, on USS-SGD and
USS-MWOZ (MiniLM, the actr-paper embeddings). Base = degree-0 role-split (USER
mean + SYSTEM mean); `+slope` adds the two per-role degree-1 slopes. Head =
Ridge; metric = **Spearman rho** between predicted and true rating on held-out
turns, GroupKFold by conversation, conversation-bootstrap CIs. **Keep margin
0.03 Spearman with disjoint CIs (frozen).**

**Negative controls (frozen):**
- `+slope_shuffled` as in Direction A.
- **Sentiment-confound bar.** A `sentiment` baseline = VADER compound score of
  the last USER turn + the last USER-turn embedding (same-turn content). Degree-1
  geometry must beat this baseline, and must retain its advantage at **lead 1**
  (predicting the *next* user rating from the current prefix) - otherwise it is a
  same-turn sentiment reader, not a trajectory signal.

**Prediction P2 (the asymmetry - the round's real test):** degree-1 improves
Spearman over degree-0 by more on USS satisfaction than it does AUC on ToolBench
outcome, and clears the 0.03 margin on at least one USS corpus with disjoint CIs,
survives the sentiment bar, and holds at lead 1. That would establish: **velocity
geometry predicts how a call feels; level geometry predicts whether it resolves**
- a clean, publishable dissociation. **Kill P2:** if degree-1 adds nothing over
degree-0 for satisfaction either, velocity geometry is null across both targets
and degree-0 is the whole story.

## Direction C (secondary): the dialogue-act channel on USS

The closest USS analog to a tool-outcome channel: a fixed per-turn dialogue-act
feature - the running rate of outcome-bearing acts (NOTIFY_SUCCESS /
NOTIFY_FAILURE) and the running act-type distribution. Arms: `base` (degree-0
role-split), `+acts`.

**Prediction P3:** `+acts` gives a modest satisfaction lift (the acts are
outcome-adjacent, like tool results) but is a weak proxy - it is a coarse tag,
not a tool execution with a payload. This is the honest, bounded gesture at "does
the action/outcome channel help satisfaction"; the un-proxied version needs true
tool calls, which USS lacks. **Kill:** if `+acts` adds nothing, even the coarse
outcome tag is redundant with the role-split geometry.

## Encoder (cited, not re-run)

The weak-vs-strong-encoder question is already answered (Round 13: MiniLM vs
bge-large; stronger encoder raises every accuracy and leaves the manipulability
structure intact, the un-gameable typicality head reaching 99% of the flat head).
It is the robustness backdrop for R16, not a fourth axis; no new encoder ladder
is run.

## Scope

Degree-1 on the fixed roles is un-gameable by construction (fixed channels);
manipulability is inherited from round 13 and not re-measured. ToolBench outcome
is the clean accuracy anchor; USS satisfaction is WoZ text with human ratings but
no tool structure, so its "action" channel is a coarse dialogue-act proxy.
Everything is text; voice process channels and true tool-call-plus-satisfaction
data remain the standing production-only gaps. Role structures differ across
corpora (4-role ToolBench vs 2-role USS) - the comparison being tested is the
*degree* effect within each target, and its asymmetry across targets, not a
cross-corpus accuracy transfer.

## Amendment (2026-07-24, before any label-touching run)

Facts about the USS artifacts, checked before running, that fix two frozen
choices. Kill/keep thresholds unchanged.

1. **No raw per-turn text; VADER unavailable.** USS ships MiniLM embeddings +
   meta (roles, per-turn `turn_ratings`, per-turn `actions`) with no raw text,
   and neither `nltk` nor `vaderSentiment` is installed. The
   sentiment-confound baseline therefore uses the **last USER-turn embedding**
   (same-turn content) in place of the VADER compound. This is a *stronger*
   content baseline, not a weaker one: the MiniLM embedding encodes the turn's
   sentiment and more, so degree-1 geometry beating it at lead 1 is a stronger
   confound defeat than beating a lexicon score would be.
2. **Ratings and acts.** `turn_ratings[i][t]` is a list of annotator scores
   (1-5) on USER turns (SYSTEM turns empty); the target is the annotator mean.
   `actions` carries NOTIFY_SUCCESS (674) and NOTIFY_FAILURE (481) among a
   frozen 17-act vocabulary, so Direction C's channel is the running
   success/failure rate plus the running act-type distribution over that
   vocabulary.

## Evidence rule

As rounds 1-15: seed 42, GroupKFold by conversation, out-of-fold, registry
artifacts with `git_dirty=false`, `./run.sh check` green. Conversation-level
bootstrap CIs (B = 1000). The degree-1 definition, short-role fallback, the 0.02
(AUC) and 0.03 (Spearman) margins, the slope-shuffle control, the VADER
sentiment baseline, and the lead-1 requirement are all frozen here and not
revised after seeing outcomes. No result counts from an unclean tree or an
unregistered run.
