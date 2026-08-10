# Pre-registration: Round 13 (split-roles restriction + conditional typicality)

**Date closed:** 2026-07-23 (committed before any label-touching run).
Round 12 proved no transform of the adversary-controlled content channel is
robust (selection minimizes any function over the pool; a bilinear g0-S
coupling reduces to a linear functional of g0 for fixed S). The split-roles
geometry offers a different lever: the attack controls ONLY the assistant
role, so every OTHER role - user, tool_call, tool_result - is non-spoofable
under it. Restriction need not mean skeleton-only; it can mean *every role
except the one the adversary writes*.

## Primary (Direction A): drop-assistant restriction

The round-9 attack substitutes assistant (thought) turns; user, tool_call,
and tool_result turns are fixed. A head built on the non-assistant role
means is therefore invariant under the attack (manipulability 0) but richer
than the tool-identity skeleton, because tool_result content carries the
actual errors/answers and user content carries repetition/frustration.

Arms (ToolBench, round-9 attack, seed 42), features from per-role running
means:
- **flat**: [user, assistant, tool_call, tool_result] - reproduces round-9
  A (0.788 / 0.50).
- **skeleton**: tool-identity mean only - reproduces round-9 B (0.706 / 0).
- **drop_assistant**: [user, tool_call, tool_result] - no assistant block.
- **drop_assistant_skel**: [user, tool_call, tool_result, skeleton].

**Prediction P1 (the target):** drop_assistant (± skeleton) has
manipulability 0 (invariant) and accuracy well above 0.706, plausibly near
0.788. If so: the spoofable narration adds gameability but little unique
accuracy; restriction to the non-adversary roles is the near-"cake" -
near-full accuracy at zero manipulability. **Kill P2:** if drop_assistant
~= skeleton (0.706), the tool_result/user content adds nothing and the
tool-identity skeleton was the whole non-spoofable signal.

**Honest note (outcome-adjacency):** tool_result content is what the tools
actually returned, close to the verified outcome on ToolBench. So high
drop_assistant accuracy is partly "reading the real result evidence" rather
than dialogue geometry - which is a *feature* for a reward (that evidence is
exactly what it should anchor to) but must be stated: the win is that the
non-spoofable evidence is nearly as predictive as the full head and
un-gameable, not that dialogue shape is doing the work.

## Secondary (Direction B): conditional typicality of the masked content

The one mathematically honest way to use the assistant content without a
free absolute-position pathway: model p(assistant content | fixed context)
and score the residual - success-content pasted on a failing context reads
as atypical. Distinct from round 12's relational *cosine* (which was low-dim
and gamed): here the residual is against a fitted conditional model.

- Fit on train: multi-output Ridge predicting assistant running mean from
  [user, tool_call, tool_result, skeleton] running means. residual_norm =
  ||assistant_mean - prediction||.
- **typicality** arm: [user, tool_call, tool_result, skeleton,
  residual_norm]. The residual is the only assistant-derived (gameable)
  feature; the rest is the invariant drop_assistant base.

**Prediction P3:** if residual_norm adds accuracy over drop_assistant AND
the arm's manipulability stays near 0 (the adversary cannot drive the
residual to "typical" using success-pool content - because success-content
is atypical under failing contexts), conditional typicality is a usable
masked-content signal, unlike the failed relational cosine. If the residual
adds nothing, or the adversary games it to ~drop_assistant + gap, it fails -
either outcome cleanly separates conditional typicality (a data property)
from the impossible functional-mask route (round 12).

## Scope

Attack substitutes assistant turns only; drop_assistant's manipulability 0
is against THIS adversary. A policy choosing its own tools/behavior
influences tool_result and user reaction - but only through real actions
with real outcomes (round 11's point; tool_result is outcome-adjacent by
construction). ToolBench only. Selection is weak - lower bounds.

## Evidence rule

As rounds 1-12: seeds 42, registry artifacts, `./run.sh check` green.
