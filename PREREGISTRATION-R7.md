# Pre-registration: Round 7 (shape-target reward, per-turn human judge)

**Date closed:** 2026-07-23 (committed before any selection-probe result).
Completes the one arm turn-encoding#92 said it could not measure: the
manipulability of a *shape* reward (trend/recovery targets), judged by a
**position-neutral per-turn human judge** instead of the arc-biased
whole-transcript LLM judge that compromised #92's own rung-1.

## The instrument #92 needed, and USS already has

#92's sandbagging spike found its LLM judge is MORE arc-biased (+0.275)
than the trend reward is sandbaggable (+0.027) - an LLM rating whole-
transcript satisfaction has a peak-end/recency bias, so it cannot
adjudicate shape-reward manipulability. The fix #92 specified verbatim:
"held-out per-turn human ratings (USS has them; score each turn, not the
whole transcript)." USS-SGD/MWOZ ship exactly that: `turn_ratings`, three
annotators per user turn. This round uses them.

## Targets and heads (Paper 3 degree-matched setup, taken as given)

Per conversation the satisfaction series = mean-annotator rating per rated
(user) turn. Its degree-0/1/2 Gram coefficients are the targets:
y0 (level/mean), y1 (trend), y2 (curvature/recovery). Reward heads predict
these from the SYSTEM channel geometry only (#92: score the system
channel, never the user channel) - g0_sys, g1_sys, g2_sys are the degree-
0/1/2 Gram coefficients of the system turn-embedding subsequence (canonical
lib/gram). Ridge (RidgeCV over {0.1,1,10}), GroupKFold-5 OOF, seed 42:

- **level**: g0_sys -> y0 (the honest baseline; should track a mean judge)
- **trend**: g1_sys -> y1
- **recovery**: g2_sys -> y2
- **combo**: z(level_pred) + z(trend_pred) + z(recovery_pred), z over OOF

Report each head's OOF Spearman against its own target (Paper 3 replication
check; expected trend ~0.4, recovery ~0.45 on SGD).

## The two judges (both human, both per-turn, differing only in weighting)

- **judge_mean** (position-neutral): equal-weight mean of the per-turn
  human ratings = y0. No recency bias - the anti-arc judge #92 asked for.
- **judge_final** (peak-end proxy): the last user turn's human rating.
  Reported to expose whether a shape reward's selections are preferred by
  a recency-weighted rater even when the position-neutral rater disagrees -
  the "is the arc a hack or a real preference" question, measured not
  assumed.

Both come from the same human annotations; neither is an LLM.

## Selection probe (#92 best-of-N manipulability, generation-free)

OOF reward scores over all conversations (each scored by a head trained
without it). For N in {1,2,4,8,16,32}, T=2000 trials (seed 42): draw N
conversations, select the one maximizing the reward, record the selected
conversation's judge_mean, judge_final, and the reward's own value.
N=1 = random baseline. Manipulability signature (#92): a reward aligned
with true satisfaction raises the judge score as N grows; a gameable one
raises its OWN score while the independent judge flatlines or falls.

**Reported per reward:** delta(judge_mean) and delta(judge_final) from N=1
to N=32, with the reward's own delta alongside (the divergence is the
manipulability). USS-SGD primary, USS-MWOZ replication.

## Registered predictions (#92's shape-target hypotheses)

- **level** raises judge_mean with N (it predicts y0; near-tautological,
  the robust anchor).
- **trend** raises judge_final but NOT judge_mean - selects improving-but-
  not-high-mean conversations (the sandbagging signature against a
  position-neutral judge). If judge_mean falls while trend's own score
  rises, the trend target is gameable as a standalone reward.
- **recovery** similar - selects dip-recover shapes; judge_mean flat/down.
- **combo** raises judge_mean (requiring good level kills the sandbag -
  "must satisfy all three at once", #92's mitigation). This is the
  falsifiable claim: combo's judge_mean-vs-N slope > trend's.

Either direction is a result. If trend also raises judge_mean, shape is
tracking real satisfaction, not gaming it (#92's live alternative). If it
does not, the standalone shape reward is confirmed sandbaggable and the
degree-structured combo is the fix - measured against a human judge for
the first time.

## Scope

Selection over real conversations is a weak, generation-free adversary -
lower bound, as every round. The judge is third-party annotator per-turn
satisfaction, not the caller's lived experience (#92's stated caveat
carries). USS-only by structure (PRISM has no turn-level ratings). No LLM
judge, no generation.

## Evidence rule

As rounds 1-6: seeds 42, registry artifacts, `./run.sh check` green
before numbers are read.
