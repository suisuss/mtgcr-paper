# Pre-registration: Round 19 (the accuracy/gameability frontier of re-admitted narration)

**Date closed:** 2026-07-24 (committed before any label-touching run).

Round 18 showed the conditional-typicality residual's anti-gameability is not
structural: on ToolBench/MiniLM its manipulability erodes from -0.10 (best-of-16)
to +0.03 at the exhaustive selection ceiling, while on ABCD it is flat at 0 for
every budget. Working the mechanism through yields a near-conservation law that
this round tests directly and uses to map the design space.

## The claim under test (the conservation law)

The two-sided residual `res = ||assistant_mean - ridge.predict(context)||`
enters a head that can lower the score by driving `res` down; a strong enough
selection adversary finds the rare pool turns that are typical given the failing
context, removing the residual's contribution to early alerts. The residual's
*accuracy* gain over the fixed-role head (0.769 -> 0.782) is the same
independent early-warning signal the adversary removes. Formalized on the
one-sided (risk-only) construction:

    reward = H_fixed(context) + lambda * max(0, res - r0)

where `H_fixed` is the standalone fixed-role (drop-written-role) head, which has
manipulability 0 (structural). Under attack the adversary drives `res <= r0`,
the penalty vanishes, and the score floors at `H_fixed(context)`. Therefore
`attacked_recall = recall(H_fixed alone)` - a constant, independent of budget -
and `manip = (base_recall - attacked_recall)/base_recall >= 0`, strictly
positive exactly when the penalty creates alerts `H_fixed` alone misses. Hence:

**Conservation law:** a residual's ceiling manipulability approximately equals
the independent recall it contributes over the fixed-role head. One-sided
rectification does NOT reset manipulability to 0 (the earlier framing was
wrong); it makes manipulability *budget-independent and capped* at the
accuracy the residual adds. Zero manipulability requires giving up the
independent accuracy (that is the fixed-role head, Option C).

This round maps every construction into the plane (independent accuracy over
0.769) x (ceiling manipulability) and tests four predictions.

## Frozen setup

Identical to Rounds 13/18: MiniLM (primary), GroupKFold fold-0 TRAIN for head +
ridge + threshold, held-out FAILING conversations for attack metrics, pool =
agentlike turns of SUCCESSFUL fold-train conversations, seed 42, manipulability
`(base_recall - recall_after)/base_recall`. Reuses the R18 adapter machinery
(imported round-13 helpers). No existing script or artifact is modified.

**Budget grid:** `n in {1, 4, 16, 64, 256, EXHAUSTIVE}`. n=4 added so Direction
D can speak to realistic best-of-4 decoding; EXHAUSTIVE = argmin over the whole
pool (the class ceiling).

**Nesting guard:** the `res_2s` arm at n=16 must reproduce the published
typicality manipulability within |delta| <= 0.02 (-0.0969529 on ToolBench), or
the run is void.

## Arms (ToolBench/MiniLM primary)

- **fixed_role** (C): drop-written-role head. Reference: manipulability 0
  (structural), accuracy 0.769. The zero-gameability anchor.
- **res_2s**: two-sided residual (the current typicality arm). Reference:
  accuracy 0.782, the R18 erosion (-0.10 -> +0.03).
- **res_1s@p50 / p75 / p90** (A): one-sided residual `max(0, res - r0)`,
  `r0` = the p50 / p75 / p90 percentile of the TRAIN residual distribution.
  The r0 dial trades accuracy contribution against manipulability cap.
- **res_task_2s** (B): two-sided residual against a task-conditioned model
  `p(assistant | context, task)`, task descriptor = the opening user-turn
  embedding of the conversation (ToolBench task proxy; the residual model's
  inputs gain the fixed opening-query vector).
- **res_task_1s@p50** (B compose A): one-sided task-conditioned residual.

Per arm, report: accuracy AUC, base_recall, the full manipulability curve over
the budget grid, `independent_recall = base_recall - attacked_recall_at_ceiling`
(the conserved quantity), and curve flatness `ceiling_manip - n1_manip`.

## Predictions and kill conditions

**P1 (A flattens the curve).** Every `res_1s@*` arm is budget-flat:
`|ceiling_manip - n1_manip| <= 0.02`, versus `res_2s`'s erosion span (~0.17 on
ToolBench). **Kill P1:** any `res_1s` arm erodes with budget comparably to
`res_2s` (span > 0.10).

**P2 (conservation law - the headline).** Across all seven arms, `ceiling_manip`
rises monotonically with independent accuracy `(acc - 0.769)`: Spearman rank
correlation > 0. **Kill P2:** no positive monotone relation - which would mean
gameability and accuracy are separable and the law is false (a more interesting
result, reported as such).

**P3 (one-sided weakly dominates two-sided).** At matched or higher accuracy,
`res_1s` ceiling_manip <= `res_2s` ceiling_manip. **Kill P3:** one-sided is not
weakly below two-sided in the plane.

**P4 (B improves the exchange rate).** `res_task_*` arms sit below the
corresponding non-task arms in the (accuracy, ceiling_manip) plane - same
accuracy at lower ceiling_manip, or same manipulability at higher accuracy.
**Kill P4:** task-conditioning yields no better accuracy-per-unit-gameability
than the plain residual.

**P5 (C is the anchor).** `fixed_role` remains manipulability 0 at accuracy
0.769 for every budget (structural; asserted and reconfirmed).

## Direction D (framing, no new run)

Read `res_2s` manipulability at n in {4, 16, 64} from its own curve as the
realistic-budget scoping statement (best-of-4/16/64 decoding). Reported as a
scope note beside the ceiling, never as a substitute for it.

## Negative-result framing (frozen)

- Law holds (P2), A flattens (P1), one-sided dominates (P3): the paper gains a
  frontier figure; the recommended reward is `fixed_role` (C, manip 0) with the
  one-sided residual offered as a bounded, budget-independent density add-on
  whose cost is stated as its accuracy contribution.
- B beats the exchange rate (P4): task-conditioning becomes the recommended way
  to re-admit narration; ties the soft layer to the section-5 substance layer.
- Law fails (kill P2): report that accuracy and gameability are separable in
  this setting - the strongest possible result - and characterize which arm
  achieves the separation.
- The fixed-role structural 0 is unaffected under every outcome and remains the
  load-bearing un-gameability claim.

## Replication (pre-registered, run if primary completes)

ABCD/MiniLM and ToolBench/BGE-large, same arms and grid, same guard against each
cell's published n=16 typicality number. ABCD is expected flat near 0 across all
arms (its residual carries ~no independent signal); a nonzero exchange-rate
improvement from B there would be the cleaner cross-corpus evidence.
