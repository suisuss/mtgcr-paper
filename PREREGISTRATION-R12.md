# Pre-registration: Round 12 (relational binding of content against grounding)

**Date closed:** 2026-07-23 (committed before any label-touching run).
Rounds 9-10: ADDITIVE tethering (concat NL g0 + skeleton) does not inherit
the skeleton's robustness - a logistic head keeps its weight on the free NL
block. Round 9 addendum: this holds for role-split(1) too, so it is a
property of additive combination, not of representation degree. This round
tests whether letting the content channel enter ONLY as a quantity measured
against the non-spoofable tool grounding - never as free absolute position -
reduces manipulability while recovering accuracy above the bare skeleton.
This is #92's "binding by parameterization" (mechanism 3), instantiated
against the tool skeleton and, for the first time, tested under the
selection attack.

## The mechanistic hypothesis

In the additive tether, substituted success-shaped NL lowers risk directly,
independent of the skeleton (manipulability 0.50). In a relational frame,
the head sees only cos(assistant content, tool skeleton): substituting
success-shaped NL onto a real FAILING skeleton should keep or raise the
say-do discrepancy (success-NL is calibrated to success skeletons), which
the head reads as MORE risk - the gaming move becomes self-defeating. The
selection adversary can move the NL but not the tools, so it cannot co-move
the two channels to keep them consistent.

## Arms (ToolBench, round-9 attack protocol, seed 42)

- **skeleton** (anchor): skeleton mean only. Invariant -> manipulability 0;
  should reproduce round-9 B (accuracy ~0.706).
- **relational**: [skeleton, cos(assistant g0, skel), cos(tool_result, skel),
  cos(user, skel)]. No raw assistant g0 - the content enters only through
  its consistency with the grounding.
- **gated**: [skeleton, gate * assistant g0], gate = clip(cos(assistant g0,
  skel), 0, 1). Ungrounded NL (low say-do consistency) is zeroed, so the
  head cannot use ungrounded content to lower risk.

Compared against round-9 committed baselines (same fold/seed): flat g0
0.788 / 0.50; additive tethered 0.788 / 0.50.

Reported per arm as (accuracy AUC, manipulability), per the matched-
predictor rule.

## Registered predictions

- **P1 (the target result):** relational and/or gated show manipulability
  well below the additive tether's 0.50 - ideally approaching the skeleton's
  0 - while accuracy exceeds the bare skeleton's 0.706. If so, this is the
  first construct that recovers dense-NL accuracy WITHOUT reopening
  gameability: binding beats both restriction (more accuracy) and tethering
  (less manipulability).
- **P2 (kill condition):** if relational/gated manipulability is also ~0.50,
  binding-by-parameterization does not survive selection - the head still
  finds a spoofable direction (e.g. picks assistant turns whose cosine-to-
  skeleton reads safe). Reported cleanly; confirms #92's own caveat that
  head-binding does not stop a co-moving policy, now shown to fail even
  against a non-co-moving selection adversary.
- **P3 (accuracy floor):** if relational accuracy == skeleton (0.706), the
  cosines add nothing and the arm is just the skeleton; if gated accuracy
  < flat (0.788), the gate costs accuracy - the honest (accuracy,
  manipulability) tradeoff, reported as a pair so a low manipulability that
  is merely low accuracy is not mistaken for a win.

## Scope

ToolBench only (rich tool skeleton). Selection is a weak, generation-free
adversary that moves NL but not tools - the exact adversary relational
binding is meant to defeat; a policy-choice adversary that co-moves tools is
round 11's territory (defeated there by task-conditioning, not by binding).
So relational binding (if it works) defends the NL-only axis; task-
conditioning defends the action-mimicry axis; the two are complementary,
not substitutes. Manipulability numbers are lower bounds.

## Evidence rule

As rounds 1-11: seeds 42, registry artifacts, `./run.sh check` green before
numbers are read.
