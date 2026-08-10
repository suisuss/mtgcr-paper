# Pre-registration: Round 9 (tool-anchored representation)

**Date closed:** 2026-07-23 (committed before any label-touching run).
Tests one hypothesis: anchoring the trajectory representation to the
non-spoofable tool-call IDENTITY reduces selection-manipulability versus a
purely-NL representation, ideally without losing accuracy. Motivated by the
gameability mechanism (rounds 5-8): the signal is gameable because it reads
fluid NL position, which a policy moves without moving the outcome; tool
calls are the one transcript element a policy cannot spoof (the log records
what was actually invoked).

## Corpus and anchor

ToolBench (rich tool skeleton, median 3 tool calls/conversation, the corpus
where the flat head is gameable at manipulability 0.50 in round 6). The
tool anchor is the MiniLM embedding of each tool-call's SERVICE name
(fixed identity; the name stands in for the description - a stated
limitation). 3,181 unique services; tool COUNT is null (fail 3.15 vs
success 3.08), so any skeleton signal is about identity/sequence, not
length. `invalid_hallucination_function_name` is a canonical non-spoofable
failure marker in the skeleton.

## Three arms (isolating the anchor's effect)

Scored at decision points (tool_result turns), round-6 attack protocol:
head + threshold on fold-0 train (precision >= 0.5), best-of-16
substitution of assistant (thought) turns from the successful-conversation
pool, n=1 drift control, GroupKFold-5 by conversation, seed 42.

- **A - flat NL** (= round-6 method_gram head): the four per-role running
  means [user, assistant, tool_call, tool_result]. Fully NL; the assistant
  block is what the attack substitutes.
- **B - skeleton only**: the mean of service-name embeddings of tool calls
  seen in the prefix. INVARIANT under the attack (substituting assistant
  turns cannot change which tools were called), so its manipulability is 0
  by construction - the number of interest is its ACCURACY.
- **C - tool-anchored (tethered)**: A + skeleton mean + the discrepancy
  cos(assistant running mean, skeleton mean). The dense NL representation
  tethered to the invariant tool identity.

Reported per arm as (accuracy AUC, manipulability) pairs, per #92's
matched-predictor rule.

## Registered predictions

- **P1 (mechanism):** A reproduces round 6 (accuracy ~0.79, manipulability
  ~0.50). B's manipulability is ~0 (invariant). C's manipulability < A's -
  because a logistic head, given the predictive invariant channel, shifts
  weight onto it and off the spoofable assistant block.
- **P2 (anchor-is-outcome check, the honest guard):** if B's accuracy is
  already ~= C's, the skeleton is doing all the work and the tethered NL
  adds nothing - the anchor is just an outcome proxy. If C's accuracy > B's,
  the tethered NL adds dense signal the bare skeleton lacks.
- **Decider:** C shows manipulability below A at accuracy >= A AND above B
  -> tool-anchoring reduces gaming BY DESIGN while keeping the dense signal;
  the first construct that lowers manipulability by construction rather
  than by monitoring. Any other pattern is reported as measured:
  - C manip ~= A -> the tether does not constrain the fluid channel (the
    head keeps leaning on the spoofable block); anchoring re-frames but
    does not defend.
  - C accuracy ~= B (both low or both high) -> the skeleton is the whole
    story; NL geometry, anchored or not, adds nothing here.

## Scope

ToolBench-only (needs a rich tool log; the public voice corpus has none,
but Heya's product logs tool calls). Selection is a weak generation-free
adversary - manipulability numbers are lower bounds. The anchor uses the
service NAME as a description proxy. The attack substitutes assistant turns
only; that the tool calls are non-substitutable is the invariance the
experiment is built on, and it mirrors reality (a policy cannot select its
way to an invocation that did not happen).

## Evidence rule

As rounds 1-8: seeds 42, registry artifacts, `./run.sh check` green before
numbers are read.
