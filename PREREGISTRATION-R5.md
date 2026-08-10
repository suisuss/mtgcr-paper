# Pre-registration: Round 5 (SpokenWOZ voice transfer + Goodhart attack)

**Date closed:** 2026-07-23 (committed before any label-touching SpokenWOZ
analysis and before any attack result is computed).

## Arm S - SpokenWOZ: the voice-transcript transfer test

The program's standing highest-leverage question: does the early-warning
head - now replicated on two written-dialogue corpora - survive genuine
spoken transcripts (ASR noise, disfluencies, spoken phenomena)?

**Data:** pinned HF text splits (KIT_PROVENANCE.json), train (4,700) +
test (1,000) pooled = 5,700 dialogues, ~35 turns average, roles
user/system.

**Transformation:** turns from `log` (tag -> turn type user/system);
goal_text rendered from the MultiWOZ-style goal dict with a fixed template
(per domain: constraint slots, booking slots, requested slots - natural
phrasing, committed with the preprocess script).

**Label (derived Success-style rule):** a dialogue succeeds iff (a) for
every goal domain with requested slots (`reqt`), each requested slot name
appears among that domain's system dialog-act payload slots somewhere in
the dialogue; and (b) for every goal domain with a booking (`book`), a
same-domain system act carries a booking acknowledgment (a `ref` slot or a
booking/ack act - exact act-vocabulary mapping committed with the script
after inspection, with coverage diagnostics, before any geometry runs;
the ABCD precedent). has_failure = not success. Dialogues with empty
goals are excluded and counted.

**Battery:** the round-4 generic battery, config: decision points = user
turns; types = [user, system]; agentlike = system; E3 final-typed =
system. No typed ladder (two roles only); no mechanical error baseline
exists - stated as a limitation; baselines are step count and running
goal distance. GroupKFold by dialogue. The pre-registered readout is the
head's AUC-per-tau-quartile profile against those baselines - the
transfer verdict is descriptive (does the dialogue-domain curve shape
survive), not a keep/kill bar.

## Arm G - Goodhart: attacking the head (sandbox; deliberate charter violation, firewalled)

Target: the per-corpus early-warning head (ToolBench, ABCD - the two
corpora where it works). Train/attack split: GroupKFold fold 0 (seed
protocol as rounds 2-4): the head and the alert threshold (the lowest
threshold reaching precision >= 0.5 on TRAIN alert curves) come from the
training side only; all attack metrics are computed on held-out
conversations.

**Attack 1 - selection pressure with real candidates (best-of-n shape).**
For each held-out FAILING conversation, at each decision point the
adversary replaces the upcoming agentlike turn with the best of n
candidates (n in {1, 2, 4, 8, 16}), drawn uniformly (seed 42) from the
pool of agentlike turns of SUCCESSFUL training conversations; "best" =
minimizes the head's risk score at that prefix (running means updated
with the substituted turn; subsequent scoring uses the substituted
history). n=1 is the drift control (random substitution, no selection).
Metrics, conversation-level on held-out failures: alert recall at the
frozen threshold vs n; mean suppressed score vs n. The user-side turns -
the failure content - are never touched.

**Attack 2 - white-box direction (embedding-space rung).** The logistic
weight vector over the concatenated per-type running means: per-type
block norms (which channel carries the decision); the minimal L2 shift of
the agentlike-mean block needed to cross the threshold for the median
failing prefix; and the 5 nearest real agentlike turns (cosine, whole
corpus) to the negative exploit direction - the readability check on what
"sounding safe" looks like.

**Claim grading (per the amended Goodhart charter):** Attack 1 is
selection over real turns - stronger than embedding-space existence,
weaker than a policy; Attack 2 is embedding-space only. Both are upper
bounds on policy exploitability; no policy claims close on this round.
**Firewall:** this is the charter's sanctioned sandbox violation of the
anti-Goodhart rule; nothing here is optimized against in production.

**Interpretation guide (fixed):** recall degradation > 50% at n=16 with
the n=1 control roughly flat = the head is materially gameable by
selection alone (headline); degradation < 10% = selection-robust at this
pressure; between = graded, reported as measured. The n=1 control
separates gaming from substitution drift.

## Evidence rule

As rounds 1-4: seeds 42, registry artifacts, `./run.sh check` green
before numbers are read.
