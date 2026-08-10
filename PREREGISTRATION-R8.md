# Pre-registration: Round 8 (convergence-phase follow-ups)

**Date closed:** 2026-07-23 (committed before either arm's labels are
touched). Two new arms motivated by the convergence phase (rounds 6-7),
plus one interpretive correction to the paper's null claims. NO closed
result is reopened - these are new hypotheses against targets the earlier
rounds did not test.

## Interpretive correction (no experiment; documentation)

The paper's boundary claim "structure-beyond-content is null at every
granularity tested" (sub-turn PRISM, typed ToolBench/SWE ladders) was
established entirely against **degree-0 / binary** targets. Round 7's
channel diagnostic showed the same coefficients carry real signal against
**degree-matched shape** targets (trend 0.42, recovery 0.47 both-roles).
The null therefore holds *against a mean/degree-0 target*, not against
shape. Every "structure adds nothing" statement is scoped to target-degree,
with a pointer to round 7. This is honesty owed, not a rerun.

## Arm A - SpokenWOZ manipulability (the deployment-relevant gap)

Rounds 5-6 measured selection-gameability on ToolBench and ABCD but never
on the voice corpus - our only spoken-transcript evidence, and the most
deployment-relevant. This applies the round-6 reward-arm matrix
(goodhart_matrix) protocol verbatim to SpokenWOZ (already encoded, derived
Success label), no redesign:

- Corpus config: types [user, system], decision = user turns, agentlike =
  system, userlike = user.
- Four arms (floor dense-g0, foil TRACE-relational, method Gram role-split,
  governed Gram + channel tripwire); head + threshold on fold-0 train
  (precision >= 0.5); best-of-16 substitution of system turns from
  successful training conversations; n=1 drift control. Seeds 42.
- Reported per #92's matched-predictor rule as (accuracy AUC,
  manipulability) pairs, tripwire-caught for the governed arm.
- **Registered expectation:** the head is gameable by selection here too
  (naked manip clears zero); the tripwire recovers some share, leaves an
  on-manifold residual (rounds 5-6 pattern). A null (voice head
  ungameable) would be a surprise worth its own note.

## Arm B - endpoint functionals against the shape targets

The endpoint charter's "ends close" died against satisfaction-mean
(degree-0, killed by the ritual confound) and against binary verified
outcomes (rounds 4-5, reversed/weak). It was never asked whether endpoint
proximity predicts the satisfaction TREND / RECOVERY shape targets - the
degree-matched re-ask. USS-SGD/MWOZ (per-turn human ratings available).

- Satisfaction series = mean-annotator rating per user turn; targets
  y0 (mean), y1 (trend), y2 (recovery) = its degree-0/1/2 Gram
  coefficients (round-7 construction).
- Role-split degree-1 gap curve d(tau) = USER line - SYSTEM line on a
  101-tau grid (canonical; gram==lstsq at degree 1, closed round 3).
  Functionals (fixed family): E1 raw final-turn gap 1-cos(U_last, S_last);
  E2 fitted endpoint ||d(1)||; E3 closest approach min ||d(tau)||;
  E4 initial gap ||d(0)|| (control).
- Test: Spearman of each functional against each of y0, y1, y2; BH-FDR at
  q=0.05 across the full 4x3 family per corpus. Inclusion: >= 3 rated
  turns, >= 2 user and >= 2 system turns.
- **Registered prior:** endpoint proximity is a final-state quantity, weak
  prior it predicts trend/recovery; "ends close" *might* track recovery (a
  converging conversation may be one that recovered). Two-sided. A
  degree-matched hit where the degree-0 test was null would reopen the
  endpoint charter's descriptive lane; a null closes it at both degrees.

## Evidence rule

As rounds 1-7: seeds 42, registry artifacts, `./run.sh check` green before
numbers are read. Selection is a weak generation-free adversary (lower
bound); USS shape targets are third-party annotator dynamics.
