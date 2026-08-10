# Pre-registration: Round 14 (tool-result content: is fixed-roles just error-counting? does tool description help?)

**Date closed:** 2026-07-24 (committed before any label-touching run).

Round 13 recovered ~97% of the head's accuracy at manipulability 0 by dropping
the written role and scoring the fixed evidence (`drop_assistant` 0.769 vs flat
0.788, skeleton-only 0.706), and re-admitted the narration as conditional
typicality (0.782, manipulability -0.097). Round 13's own honest note says the
recovered accuracy leans on tool_result content, which is outcome-adjacent on
ToolBench. Round 14 stress-tests that concession directly, and asks whether the
tool channel can be enriched the way a real deployment (MCP) would allow.

Three questions, each with a kill condition fixed here:

1. **Is `drop_assistant` earning its accuracy, or is it counting errors?**
   Table 2 has no process baseline for ToolBench (the column reads "-"), so the
   corpus carrying the headline result has never been checked against the
   trivial control. This is the tool-result analogue of the standing sentiment-
   confound bar (a semantic signal must beat a keyword baseline).
2. **Does the tool *description* (and argument schema) add operational signal?**
   Under our adversary a description is as fixed as the tool name - the provider
   writes it, keyed by tool id - so it preserves manipulability 0. The question
   is whether it adds anything a name-embedding of the same discrete id cannot.
3. **Structured flag vs semantic payload for the tool result** - does the
   payload embedding beat a parsed success/failure flag, given that the flag
   demonstrably lies (see the exemplar below)?

## Direction A (primary): the missing ToolBench process baseline

Two error-rate baselines, each a scalar recomputed at every decision point
(`tool_result` turn), scored by the same logistic head and attack harness:

- **err_naive**: fraction of tool calls so far whose `error` field is non-empty.
  (The protocol flag - what MCP's `isError` would give you.)
- **err_strong**: fraction of tool calls so far that failed by a hand-built
  heuristic fixed *now*: non-empty `error` field, OR empty/`[]`/`""` payload,
  OR payload matching a frozen failure-string set (`disabled`, `rate limit`,
  `blocked`, `not found`, `no such function`, `does not exist`). The string set
  is frozen in this document and not tuned against outcomes.

`drop_assistant` (0.769) must beat the *stronger* of the two.

**Prediction P1 (keep):** `drop_assistant` beats `max(err_naive, err_strong)`
by >= 0.03 AUC with non-overlapping conversation-bootstrap 95% CIs. Then the
fixed-role geometry carries signal beyond error-counting and Step 5's claim is
earned.

**Kill P1 (requalify):** if `err_strong` reaches `drop_assistant` within 0.01
AUC or overlapping CIs, then "fixed roles recover 97% of the head" is largely
"count the failed calls" wearing geometry. Action if killed: the paper's Step 5
claim is requalified in §4 and §7 to "the fixed evidence is nearly as
predictive as the head, and most of that is a computable error rate" - and
err_strong is added to Table 2 as the ToolBench process baseline regardless of
outcome (the missing control ships either way).

## Direction B (secondary): description-enriched skeleton

The skeleton currently embeds `name.replace("_"," ")` once per tool id. Enriched
variants, each still a deterministic function of tool id (manipulability 0
preserved by construction - assert, do not re-measure as the headline):

- **skel_desc**: `embed(name + ": " + description)`.
- **skel_desc_resid**: `embed(description)` minus its projection on
  `embed(name)`, to strip provider boilerplate common-mode.
- **skel_schema**: `embed(name + " " + argument-schema-keys)`.

Descriptions/schemas are taken from ToolBench tool metadata (MCP is the
motivating target but there is no MCP corpus here - transfer is a stated gap,
not a claim). Two evaluations:

- **In-domain** (standard ToolBench split, tools shared train/test).
- **Held-out-tool** (tools in test disjoint from train - a fixed split by tool
  id, hashed, ~20% of ids held out, frozen now).

**Prediction P2a (null in-domain):** skel_desc gives <= 0.01 AUC over the plain
skeleton on the in-domain split. Predicted null by containment: description is a
deterministic function of tool id already learnable from data, so it adds metric
structure, not capacity. A *large* in-domain jump would be a red flag (leakage /
description text encoding the outcome) and triggers an audit, not a win.

**Prediction P2b (gain out-of-domain):** skel_desc (or skel_desc_resid) beats
the plain skeleton by >= 0.03 AUC on the held-out-tool split, where the name
alone is near-uninformative. This is the real target for MCP's changing tool
sets.

**Prediction P2c (task-conditioning gain):** in the action-substitution /
task-conditioning arm (Direction D), description-matching between the caller's
request and the invoked capability lowers the any-task slack below its current
0.82. This is where description is predicted to help most, not in-domain AUC.

**Kill B:** if skel_desc shows no gain out-of-domain (P2b fails) *and* no slack
reduction (P2c fails), tool descriptions add nothing operational here - drop the
idea and record the null.

## Direction C (secondary): structured flag vs semantic payload

Per `tool_result`, a small structured vector fixed now: `[errored, empty_payload,
status_class one-hot (ok / client_err / server_err / rate_limit / disabled /
other), payload_len_bucket]`. Latency is not in ToolBench, so omitted.

- **struct**: the structured vector's running mean (+ skeleton), no payload text.
- **semantic**: the current tool_result content mean (= what `drop_assistant`
  uses for that block).
- **both**: concat.

**Prediction P3 (keep semantic):** `semantic` beats `struct` by >= 0.03 AUC, and
`both` ~= `semantic` (the flag adds little the payload text does not). Reported
alongside: the count of decision points where err_naive mislabels a true failure
(empty `error` field on a failing call) - the mechanism.

**Kill P3:** if `struct` ~= `both` within 0.01 AUC, the payload embedding is
merely re-deriving the parsed flag; use the cheap structured vector and say so.

**Falsifying exemplar (already visible, conv 12543, Figs 1-2).** Every failing
tool result in it has an *empty* `error` field: turns 3 and 6 return
`{"error":"","response":"[]"}`, turn 13 returns
`{"error":"","response":"{'message':'This endpoint is disabled for your
subscription'}"}`. err_naive scores all three as success (0/3 failures caught);
the payload encoder sees "[]" and "disabled for your subscription" as failure-
shaped. This is why struct alone is predicted to underperform - and why a naive
MCP `isError` reward would be gameable by tools that return 200-with-empty-body.

## Direction D (gating, not optional): the adversary must move to the action channel

Any arm that scores tool *success* moves the attack surface from the narration
channel to the action channel, where our round-9 selection adversary (assistant-
substitution only) is blind. A policy rewarded for successful results learns to
call cheap always-succeeding tools (harvest 200s, resolve nothing) - exactly the
any-task failure Step 7 exists to catch (slack 0.82).

**Firewall (anti-Goodhart, fixed here):** no success-channel arm (Directions
A err-rate, C struct/both) is reported as a *reward* candidate without
evaluation under the *action-substitution* adversary and task-conditioning
(`policy_choice`-style: substitute tool calls, recompute the true label). Under
the narration-only attack these arms are trivially manipulability 0 and that
number is not to be quoted as robustness. The success-channel result is a
predictor result until the action adversary has run.

## Scope

ToolBench for A/B/C; ABCD (action-computable label) for the Direction D action-
substitution / task-conditioning evaluation. Descriptions and schemas are
ToolBench metadata; MCP is the motivating deployment but MCP transfer is unshown
- a stated gap alongside the standing voice-transcript gap. Selection remains a
weak adversary: all manipulability numbers are lower bounds. Held-out-tool and
failure-string sets are frozen in this document before any run.

## Evidence rule

As rounds 1-13: seed 42, GroupKFold by conversation, out-of-fold, registry
artifacts with `git_dirty=false`, `./run.sh check` green. Bootstrap CIs are
conversation-level. No result counts from an unclean tree or an unregistered
run; kill/keep thresholds above are fixed and not revised after seeing outcomes.

## Amendment (2026-07-24, before any label-touching run)

Two facts about the preprocessed corpus, checked before running, that constrain
the above. Recorded here so the change is closed with the pre-registration, not
made after seeing outcomes. Kill/keep thresholds are unchanged.

1. **`metadata.success` is a label leak - do not use it.** The `success`
   boolean on each `tool_call` is CONSTANT within every conversation
   (12,598/12,598) and equals the conversation outcome exactly: mean per-call
   success is 1.000 on give-answer conversations and 0.000 on give-up. It is the
   `has_failure` label copied onto each call, not a per-call HTTP status (e.g.
   in conv 12543 the `get_all_teams` call returns real data but is marked
   `success=False` like every other call in that failing trajectory). Direction
   A's `err_naive`/`err_strong` and Direction C's structured vector are therefore
   parsed **only from the tool_result `content` text** (the returned JSON's
   `error`/`response` fields); `metadata.success` is banned as a feature in every
   arm. Any arm that touched it would score ~1.0 AUC by construction.

2. **No free-text tool descriptions in the corpus.** Preprocessing kept only
   `tool_name`, `service`, and `parameters` per call; there is no API/tool
   description and no argument *schema* (only the concrete `parameters` an agent
   passed, which is agent-authored content, not a static schema). Direction B is
   therefore implemented with the enrichments that ARE deterministic functions of
   tool id and present in the data, tested against the current service-name
   skeleton:
   - **skel_name**: `embed(tool_name)` (e.g.
     `image_search_for_bing_image_search` -> `"image search for bing image
     search"`) - richer than the current `embed(service)` (`"bing image
     search"`).
   - **skel_argkeys**: `embed(sorted unique parameter-key set aggregated over all
     calls to that tool id)` - a corpus-derived argument-structure proxy, keyed by
     tool id, standing in for the schema.

   Literal free-text descriptions (and true JSON-schemas), i.e. the MCP case,
   require the external ToolLLM tool registry (`toolenv/tools/*/api.json`), which
   is not fetched by this kit. Sourcing it is a stated Direction-B gap, not run
   here; predictions P2a/P2b/P2c are evaluated on skel_name/skel_argkeys as the
   available deterministic-of-id enrichments, keeping the in-domain-null /
   held-out-tool-gain structure intact.
