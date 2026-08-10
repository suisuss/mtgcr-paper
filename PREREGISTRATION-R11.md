# Pre-registration: Round 11 (the policy-choice adversary)

**Date closed:** 2026-07-23 (committed before any label-touching run).
Rounds 9-10 showed the action skeleton un-gameable *by NL substitution*.
But a real policy chooses its own actions. This round asks: can a
policy-choice adversary move the skeleton, and at what real-outcome cost?
The claim under test is that the skeleton is un-gameable *in substance* -
the only way to make the actions look successful is to actually perform the
successful actions - because the outcome is (largely) action-determined.

## Arm 1 - skeleton-to-outcome determinism (both corpora)

If identical action skeletons yield identical outcomes, the skeleton
*determines* the outcome and a policy cannot move it toward "looks
successful" without moving it toward "is successful." Group conversations
by their set of distinct action slugs (ABCD) / tool services (ToolBench).
For groups shared by >= 5 conversations, report:

- **outcome purity**: conversation-weighted fraction agreeing with the
  group's majority label, vs the base-rate purity max(p, 1-p) a
  non-determining skeleton would give. Determinism = purity - base.
- **coverage**: fraction of conversations that fall in such shared groups
  (ToolBench services are diverse; coverage will be lower there - reported,
  not hidden).
- **ambiguous fraction**: conversations whose exact action-set appears in
  BOTH outcome classes - the slack a policy could exploit.

**Registered prediction:** ABCD high determinism (label is action-derived -
the round-10 circularity, now the mechanism of robustness); ToolBench lower
(tool *results*, not just tool identity, decide give-answer/give-up, so the
same tool set can succeed or error - genuine slack between calling the right
tool and it working).

## Arm 2 - policy-choice attack with real-label accounting (ABCD)

ABCD's TRACER label is a deterministic function of the action sequence, so
we can compute the real outcome of a *gamed* action sequence directly - the
one corpus where "did gaming the detector actually change the outcome" is
answerable without executing anything.

- Detector = the round-10 skeleton head (B), frozen on fold-0 train.
- Adversary: for each held-out FAILING conversation, replace its action
  sequence with the best of 16 real SUCCESSFUL-training-conversation action
  sequences, chosen to minimize the detector's risk score (the policy
  picking success-shaped actions). Seed 42.
- Two substitution pools, run separately:
  - **same-subflow**: successful conversations of the SAME subflow.
  - **any-subflow**: successful conversations of any subflow.
- Measure, over the gamed conversations:
  - **(a) detector fooled**: the skeleton head no longer alerts (this WILL
    be high - we are directly moving the skeleton).
  - **(b) real success**: recompute the TRACER label (>= 80% of the
    ORIGINAL conversation's subflow required kb-query actions, in order) on
    the gamed action sequence - did it actually become a success?
  - **slack = fooled-but-not-succeeded**: detector alert suppressed while
    the real label stays failure. This is the exploitable surface.

**Registered predictions:**
- same-subflow: fooling the detector ~= flipping the label (slack ~ 0) -
  the only way to look successful *at this task* is to perform this task's
  required actions, i.e. to succeed. Un-gameable in substance.
- any-subflow: nonzero slack - the subflow-BLIND detector can be fooled by
  generic success-shaped actions that do not satisfy the specific task.
  This is a real, measured exploit, and it names its own fix: condition the
  detector on the task/subflow so generic-success actions no longer fool it.

The gap between the two pools is the value of task-conditioning, quantified.

## Scope

Arm 2 exploits that ABCD's label is action-computable; it is not available
on ToolBench (label needs tool execution) - Arm 1's determinism is the
ToolBench evidence. Substituting a whole successful action sequence is a
strong-but-idealized policy (it can copy any real success wholesale);
real policies are more constrained, so slack here is an UPPER bound on
gameability, not a lower bound - the opposite direction from the NL
selection rounds, stated plainly.

## Evidence rule

As rounds 1-10: seeds 42, registry artifacts, `./run.sh check` green before
numbers are read.
