# Pre-registration: Round 6 (turn-encoding #92 rung-1, verified-label judge)

**Date closed:** 2026-07-23 (committed before any arm's manipulability is
computed). Instantiates the reward-arm comparator matrix of
turn-encoding#92's rung-1 as a selection-pressure probe, with one
deliberate change from #92's own rung-1: the judge is the **verified
outcome label**, not an LLM judge.

## Why the verified-label judge (the change from #92's rung-1)

#92's own rung-1 (its final comment) found its LLM judge is *more*
arc-biased than the shape reward is sandbaggable - so LLM-judged
manipulability of any embedding reward is confounded by the judge's
shared peak-end bias. On corpora with a **verified, non-embedding-spoofable
outcome label** (ToolBench `target`, ABCD TRACER-rule), that anchor IS the
ground truth: selection over NL turns cannot change tool success. So the
manipulability metric here needs no LLM judge - it is measured directly
against the verified label, which is the strongest possible cross-modal
judge and sidesteps #92's compromised instrument. This is the cleaner
rung-1 #92 could not run because USS satisfaction is soft.

## The probe (selection pressure, generation-free)

Extends mtgcr-paper round 5. Per corpus (ToolBench, ABCD - the two
verified-label corpora where the head works), per reward arm: train a
frozen head on GroupKFold fold-0 TRAIN; set the alert threshold to the
lowest reaching train precision >= 0.5; on held-out FAILING conversations,
substitute each agentlike turn with the best of n real agentlike turns
drawn from SUCCESSFUL training conversations (best = minimizes that arm's
risk at the current prefix); n in {1, 16}, n=1 = random drift control.
The caller/user content is never touched. Seeds 42 throughout.

**Reported per arm as a PAIR (the #92 matched-predictor rule):**
held-out failure-detection AUC (accuracy) AND manipulability = relative
alert-recall suppression at n=16 with the n=1 drift figure alongside.
Manipulability alone is uninterpretable; the accuracy confound must be
visible.

## The four arms (#92's minimum discriminating set)

1. **Floor - dense g0** (naive mean-pool): running overall mean embedding.
   The canonical warned-about embedding reward.
2. **Foil - TRACE-relational**: incrementally-maintainable relational
   suite - running roughness (mean consecutive-turn cosine distance),
   running drift (1 - cos(first, current)), running user-system gap
   (1 - cos(user running mean, system running mean)), running volatility
   (std of consecutive distances). Same head class, only the feature map
   differs.
3. **Method - Gram role-split (degree-0)**: per-role running means
   concatenated (the round-5 head). The channel-factorizable feature set.
4. **Governed - Gram + channel tripwire** (ONLY the factorizable feature
   admits this): the same Gram head, plus two monitored-not-optimized
   tripwires computed from the per-role blocks - system-vs-user divergence
   (1 - cos(system mean, user mean)) and system-channel energy
   (||system mean||). Envelope = [0.5, 99.5] percentile over training
   prefixes. A suppressed alert is CAUGHT if, at the decision point where
   the head's score is pushed below threshold, either tripwire feature is
   outside the envelope. Net governed manipulability counts only failures
   whose suppression is never caught.

## Registered predictions (#92's thesis, testable here)

- **P1 (naked parity):** floor, foil, and naked Gram show similar
  manipulability - all dense-ish proxies are hackable by selection. (Our
  round-5 result already put naked Gram at 84% ABCD / 50% ToolBench; the
  new content is whether TRACE and dense-g0 land near it.)
- **P2 (governance separates):** the tripwire, available only to the
  channel-factorizable Gram, catches a materially larger share of
  suppressions than zero - net governed manipulability < naked Gram
  manipulability, at equal head accuracy.
- **P2-null is publishable and expected-possible:** our round-5 attack was
  on-manifold (exploit turns were ordinary success-shaped utterances), and
  #92's own binding-ceiling paragraph says on-manifold hacks evade
  envelope tripwires. If the tripwire catches little, that is direct
  evidence the governance layer does not defend against the attack that
  actually works - a finding, not a failure.

## Scope

Selection over real turns is a weak adversary - stronger than
embedding-space existence, weaker than a trained policy - so every number
is a lower bound on manipulability, exactly as #92 scopes rung-1. No
policy claims; no generation; no LLM judge. This is the generation-free,
verified-label instantiation of #92's minimum discriminating set, run as
one campaign module per the contract.

## Evidence rule

As rounds 1-5: seeds 42, registry artifacts, `./run.sh check` green
before numbers are read.
