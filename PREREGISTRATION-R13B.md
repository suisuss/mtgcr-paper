# Pre-registration: Round 13B (split-roles restriction on ABCD - replication)

**Date closed:** 2026-07-23 (committed before the run). Direct replication of
round 13 (PREREGISTRATION-R13.md) on a second corpus, ABCD. Same design, no
new hypotheses.

## Design mapping (ToolBench -> ABCD)

- Roles: [customer, agent, action]; adversary-written role = **agent** (the
  attack substitutes agent turns, as rounds 6/10); decision points =
  customer turns; skeleton = action slugs (round-10 embedding).
- Non-adversary (fixed) roles = [customer, action]. Arms: flat
  [customer, agent, action]; skeleton; drop_agent [customer, action];
  drop_agent_skel [customer, action, skeleton]; typicality (drop_agent_skel +
  residual of agent content against a fitted p(agent | customer, action,
  skeleton)).

## Predictions (same as round 13)

- **P1:** drop_agent (+skeleton) manipulability 0 (invariant under agent
  substitution); accuracy above skeleton-only. **The manipulability
  replication is the clean, label-independent result** - it confirms the
  split-roles restriction is un-gameable on a second corpus.
- **P3:** conditional typicality stays un-gameable (manip <= ~0) and may be
  anti-gameable, as on ToolBench.

## Heightened circularity caveat (load-bearing)

ABCD's label is the TRACER rule, **derived from the action sequence** (round
10). So the fixed-role evidence here INCLUDES the action channel the label is
computed from - accuracy magnitudes are contaminated by label leakage and are
NOT clean evidence (round 10's count-baseline showed the leakage is
identity-driven, not trivially count, but it is present). The clean anchor
for the accuracy claim remains ToolBench (round 13, independent give-answer/
give-up label). ABCD contributes the **manipulability replication** (a
structural, label-independent property) and the relative pattern, not the
accuracy magnitude.

## Evidence rule

As rounds 1-13: seed 42, registry artifacts, `./run.sh check` green.
