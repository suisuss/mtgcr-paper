# Pre-registration: Round 10 (skeleton restriction on ABCD)

**Date closed:** 2026-07-23 (committed before any label-touching run).
Drives down the round-9 road: restrict the representation to the
non-spoofable canonical-action channel, on ABCD - the most-gameable corpus
(round 6 flat head: manipulability 0.87).

## The circularity caveat (load-bearing, stated before the run)

ABCD's label is the TRACER rule: success iff >= 80% of the subflow's
kb-query required actions are performed in order. It is **derived from the
action sequence.** Action count already differs by label (fail 3.65 vs
success 4.01 actions/conversation), because completing required actions IS
the label. So the action skeleton predicts the label partly by
*reconstructing the label rule*, not by carrying independent generalizable
signal. Two consequences fixed here:

1. **ABCD skeleton ACCURACY is an upper bound**, inflated by
   label-definitional leakage. The clean, non-circular accuracy anchor for
   the skeleton road is ToolBench's round-9 B = 0.706 (independent
   give-answer/give-up label). ABCD accuracy is reported *with* this caveat,
   never as clean evidence.
2. **A count-only baseline (D)** is included to quantify the leakage: if the
   skeleton's accuracy ~= the n-actions-in-prefix accuracy, it is pure
   count-leakage; if skeleton > count, action IDENTITY adds signal beyond
   the count the label rule encodes.

The **manipulability** result is clean and label-independent - it is a
structural property (the NL-substitution attack cannot change which actions
were performed), true regardless of how the label was built. That is the
round's real deliverable: does restriction take ABCD from 0.87 gameable to 0.

## Arms (round-9 protocol, decision = customer turns, agentlike = agent)

Skeleton source = the canonical agent ACTION slugs (30 distinct;
pull-up-account, verify-identity, notify-team, ...) from meta.action_names;
embedded once with MiniLM.

- **A - flat NL**: four... three per-role running means [customer, agent,
  action]. The round-6 ABCD head (accuracy ~0.76, manipulability 0.87).
- **B - skeleton only**: mean of action-slug embeddings performed in the
  prefix. Invariant under agent-turn substitution -> manipulability 0.
- **C - tethered**: A + skeleton + discrepancy cos(agent mean, skeleton).
- **D - count only**: n actions performed in the prefix (1 feature).
  Invariant -> manipulability 0. The leakage yardstick.

Best-of-16 substitution of agent turns from successful-conversation pool,
fold-0 train head + threshold (precision >= 0.5), GroupKFold-5 by
conversation, seed 42. Reported as (accuracy AUC, manipulability) pairs.

## Registered predictions

- **P1 (the road's payoff):** B and D manipulability = 0 (invariant); A
  reproduces round 6 (accuracy ~0.76, manipulability ~0.87). Restriction to
  the action channel takes the most-gameable corpus to un-gameable.
- **P2 (round-9 mechanism replicates):** C manipulability ~= A (adding the
  anchor to the NL head does not remove reliance on the spoofable block -
  robustness needs restriction, not addition).
- **P3 (leakage decomposition):** B accuracy > D accuracy would mean action
  identity adds over the count the label encodes; B ~= D would mean the
  ABCD skeleton accuracy is essentially the label rule leaking through
  count. Either way the honest accuracy anchor stays ToolBench 0.706.

## Scope

ABCD label is action-derived (the whole caveat above); accuracy is not
clean evidence, manipulability is. B/D manipulability = 0 is against the
NL-substitution adversary; a policy choosing its own actions must perform
real actions (with real downstream effects) to move the skeleton - the same
"faking the non-spoofable channel requires real commitments" point as round
9. Selection is a weak adversary (lower bounds).

## Evidence rule

As rounds 1-9: seeds 42, registry artifacts, `./run.sh check` green before
numbers are read.
