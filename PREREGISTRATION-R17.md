# Pre-registration: Round 17 (the two-mode deployment - running early-warning + terminal read)

**Date closed:** 2026-07-24 (committed before any label-touching run).

Round 16 clarified that *lead* (the target horizon) and *degree* (the feature
order) are orthogonal, and that degree-1 nests degree-0 but does not dominate:
null for outcome, a satisfaction signal that loses to same-turn content. That
points not at one universal representation but at a **two-mode deployment**:

- **Running mode** - during the call, flag failure forming, from the prefix.
- **Terminal mode** - at the final turn, read the level (degree-0): did it
  resolve.

R17 tests whether that split is justified, on the leading method (the fixed-role
degree-0 head), all on the un-gameable channels - robustness inherited from
round 13, not re-measured. Three falsifiable questions; the honest prior (from
R15-16) is that the running mode is *also* degree-0, in which case "two-mode"
means two read-positions of one representation - still a real deployment result.

Base = the fixed-role head (drop_assistant). Fold, encoder (MiniLM),
regularization, and conversation-bootstrap CIs (B = 1000) as rounds 13-16.
Corpora: ToolBench (primary, verified label) and ABCD (replication, longer
customer-service dialogues with more lead headroom). Decision points are
tool_result turns (ToolBench) / action turns (ABCD). **Lead is measured in
decision points; lead statistics are computed on conversations with >= 3
decision points, where lead >= 2 is achievable** (ToolBench median is 3, a
compressed regime - hence the ABCD replication). Content baseline = the
last-turn embedding (as amended in R16; ToolBench has text but we keep the same
same-turn-content baseline for comparability). Everything frozen below.

## Direction A (primary): is there actionable lead? (running-mode justification)

Freeze two thresholds on the training fold: precision 0.5, and FPR 0.10. For each
held-out FAILING conversation, the head fires at each decision point; record the
first-alert decision-point index and **lead = (terminal index) - (first-alert
index)**. Report, at FPR <= 0.10: recall (any-point), median lead among alerted,
**the fraction of failures caught with lead >= 2**, and FPR (fraction of SUCCESS
conversations that ever alert). "Two-mode" = alert if ANY decision point fires;
"terminal-only" = alert on the terminal decision point alone.

**Confound (frozen):** the geometric head's early alerts (lead >= 1) must beat
the content baseline evaluated identically - the geometry must flag *earlier*
than reading the last turn does, or the "lead" is same-turn content arriving on
schedule.

**Prediction P1 (keep - the running mode is justified):** >= 25% of failures are
caught with lead >= 2 at FPR <= 0.10 (on conversations with >= 3 decision
points), AND the geometric head catches more failures early (lead >= 1), or at
longer lead, than the content baseline. Then acting mid-call adds value over
waiting for the terminal read.

**Kill P1:** early-lead recall < 25%, or the content baseline flags as early as
the geometry. Then failures are only detectable near the end, the running mode
adds nothing over the terminal read, and the deployment collapses to one mode.

## Direction B (primary): does the running mode need degree-1? (velocity, early)

R16 found degree-1 null for outcome in aggregate - but the aggregate averages
over positions. The sharp refinement: velocity (slope) may separate failing from
succeeding trajectories *early*, before their levels (means) have diverged.
Stratify decision points by within-conversation position at the per-conversation
**median index** (early = first half, late = second half; frozen). Arms per
stratum: degree-0 fixed roles vs `+degree-1 slope`, with the leak-free
slope-shuffle control (as R16). **Keep margin 0.02 AUC with disjoint CIs.**

**Prediction P2:** honestly expected KILL, given R15-16's consistent nulls for
structure-beyond-the-mean on outcome - but this is the test that would justify a
*distinct* running representation. **Keep** if `+degree-1` beats degree-0 by >=
0.02 in the EARLY stratum with disjoint CIs AND beats its shuffle: then the
running mode uses degree-1 and the two modes are two representations. **Kill** if
degree-1 is null in both strata: the running mode is degree-0, and "two-mode" is
two read-positions of one degree-0 head.

## Direction C (secondary): is the terminal read degree-0? (confirming the level read)

At the TERMINAL decision point only, degree-0 vs `+degree-1 slope`. This tests
the other half of the split: that the final outcome judgment is a *level* read,
so "cut it down to degree-0 for the final utterance" is correct.

**Prediction P3 (expected confirm):** degree-1 adds < 0.02 AUC at the terminal
point (consistent with R16's aggregate, which the later positions dominate), so
the terminal mode is degree-0. **Disconfirm** if degree-1 helps at the terminal
point by >= 0.02 with disjoint CIs - then the final judgment also needs velocity
and the "degree-0 terminal read" claim is wrong.

## Synthesis (the two-mode verdict, decided by the above)

- A keep + B keep -> full two-mode: **degree-1 running, degree-0 terminal**.
- A keep + B kill (+ C confirm) -> two-mode as **two read-positions of one
  degree-0 head** (early flag + terminal level read, same representation) - the
  honestly-expected outcome, and still a deployable finding.
- A kill -> **one mode**: terminal read only; no early-warning value at
  acceptable FPR.

## Scope

Lead is in decision points on short conversations (ToolBench median 3; ABCD
longer) - turns, not wall-clock; the achievable lead is inherently small and the
existing "two-step median lead" is the yardstick. All arms are fixed-role and
narration-invariant, so manipulability is 0 by construction and inherited, not
re-reported. The terminal read assumes the call's end is known (true post-hoc;
for live use "terminal" is the current last turn). Still text corpora (plus
ABCD's real customer channel); voice lead-time and true production intervention
value remain the standing gaps.

## Evidence rule

As rounds 1-16: seed 42, GroupKFold by conversation, out-of-fold, registry
artifacts with `git_dirty=false`, `./run.sh check` green. Conversation-level
bootstrap CIs (B = 1000). The FPR 0.10 and precision 0.5 operating points, the
lead >= 2 threshold, the >= 3-decision-point filter, the early/late median split,
the 0.02 margin, the slope-shuffle control, and the last-turn-embedding content
baseline are all frozen here and not revised after seeing outcomes. No result
counts from an unclean tree or an unregistered run.
