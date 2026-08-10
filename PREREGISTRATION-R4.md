# Pre-registration: Round 4 (new corpora - ABCD and SWE-agent-trajectories)

**Date closed:** 2026-07-23 (committed before any label-touching analysis on
either new corpus). SpokenWOZ is fetch-wired in this round but its
experiments are deferred to their own pre-registration.

## Pins (also in KIT_PROVENANCE.json)

- ABCD: github.com/asappresearch/abcd @
  `6b8700ce67c6b37b062dd7a60abc76d7ef832a97` (MIT;
  data/abcd_v1.1.json.gz fetched raw at that commit; sha256 recorded at
  fetch time since upstream publishes no checksums).
- SWE-agent-trajectories: HF `nebius/SWE-agent-trajectories` @
  `68195a1450865274106246d0d0296a1d6807b88e` (CC-BY-4.0).
- SpokenWOZ text: HF `ssz1111/SpokenWOZ-Train-Text` @
  `d3aad10f2e5a37e7e1e84375f0db368b5872a044`, `ssz1111/SpokenWOZ-Test-Text-Fixed` @
  `d6c2d9e53b1005e327db582d327b69311092eb65` (CC BY-NC 4.0 per project page).

All fetches gated behind `--fetch-data`. Encoder: all-MiniLM-L6-v2,
L2-normalized, per-turn + one goal embedding per conversation (the
normalized schema of the ToolBench port).

## Transformations

**ABCD** (from abcd_v1.1.json.gz, all splits pooled): turns from the
utterance stream with roles {customer, agent, action}; goal_text templated
from the scenario (flow/subflow name + the scenario's personal-context
fields, fixed template committed with the preprocess script);
**outcome label** = TRACER-style rule: success iff >= 80% of the
subflow's guideline-required actions are performed in order (computed from
the repo's guidelines/kb files; the exact matching implementation is
committed before any geometry stage runs, and label construction is
reported with its match-rate diagnostics). Known limitation: the 80%
rule is a researcher heuristic, recorded as such.

**SWE-agent-trajectories**: roles mapped {system -> dropped, first user ->
user (carries the goal), later user -> tool_result (observations), ai ->
assistant}; goal_text = the first user message's ISSUE block (regex from
"ISSUE:" to the next all-caps section header, else the full message); step
text truncated at 500 chars (ToolBench convention); label = `target`
(PR-test-verified success). **Subsample (sizing, fixed):** sample
instance_ids uniformly (seed 42) until >= 12,000 trajectories are
included, taking every trajectory of a sampled instance. **All CV groups
by instance_id** (trajectories of one issue never cross folds).

## Arms

### Arm A - ABCD escalation density (the gate for the escalation study)

Counting rule (fixed): (1) subflow/action names matching
/escalat|manager|supervisor|transfer|handoff/i; (2) utterance-level
matches of the same pattern in customer or agent turns, spot-checked on a
random sample of 50 hits (seed 42) for genuineness. Report: events per
conversation, conversations with >= 1 genuine event, and the verdict
against the pre-registered viability bar: **>= 300 conversations with a
genuine escalation event** are required to power an early-warning arm;
below that, the escalation study moves to Heya data / ABCD is used for
outcome work only.

### Arm B - generic replication battery, per corpus (ABCD, SWE)

One parameterized stage battery, protocols identical to rounds 2-3:

1. **Early-warning head:** prefix samples at every {customer turn (ABCD),
   tool_result turn (SWE)}; features = concat of per-role running means;
   logistic (C=1, balanced), GroupKFold 5 by {conversation (ABCD),
   instance_id (SWE)}; OOF AUC per tau quartile.
2. **Baselines:** running length; running goal distance (assistant/agent
   running mean to goal); **content-error baseline (SWE only, fixed
   pattern list):** running share of observations matching
   /Traceback|error|FAILED|Exception|fatal/i - the Bar-1 analog. ABCD has
   no mechanical error channel; its head is compared against length +
   goal-distance only, stated as a limitation.
3. **Endpoint functionals:** E1 final-turn-to-goal, E3 final-{action
   (ABCD), tool_result (SWE)}-to-goal, AUC + MWU, BH-FDR within family;
   SWE also gets the round-3 error-content stratification (final
   observation matches the pattern list vs not, same verdict rule).
4. **Typed ladder (SWE only):** R1 mean+length, R2 +per-type means,
   R3 +per-type degree-1 (canonical Gram), R4 +recurrence-on-assistant +
   per-type goal distances; logistic OOF AUC, keep bar +0.01 with
   group-level bootstrap CI > 0 (1000, seed 42).
5. **Thought-presence transfer check (SWE):** the retracted inversion's
   residual - does presence/rate of assistant reflection predict failure
   outside ToolBench's DFS generator? Report assistant-turn share AUC,
   descriptive lane.

## Sizing (recorded)

SWE subsample ~12K trajectories x ~60 median steps ~ 0.7-1.1M step
encodings (~1.5 GB fp32) - GPU minutes, fits disk/RAM. ABCD ~10K x ~22
turns ~ 220K encodings - trivial.

## Evidence rule

As rounds 1-3: seeds 42, bootstraps 1000, registry artifacts only,
`./run.sh check` green before numbers are read.
