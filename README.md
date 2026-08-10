# mtgcr-paper — reproduction kit

**Manipulability of Trajectory-Geometric Conversation Rewards and a Role-Restricted, Task-Conditioned Construction**

Jacob Sussmilch, Heya Enterprises Pty Ltd

Everything needed to read, check, and re-run the paper: the manuscript source (`paper/paper.md`, built to `paper/paper.pdf`), the full machine-checked reproduction record it curates from (`paper/reproduction_record.md`), the pre-registrations that closed before each round, every result artifact, the scripts that produced them, and a one-command pipeline that regenerates the lot from public data.

This is the **reward paper** of a three-paper line on reading conversations as trajectories through embedding space: a multi-round, pre-registered investigation into whether a trajectory-geometric representation of a conversation can serve as a **reward signal** for language models — and, when it cannot, what representation is robust instead.

*Status: preprint in preparation. The kit is complete and gated; the manuscript has not yet been posted or peer-reviewed.*

## Start here

| | |
|---|---|
| **1. Read the paper** | `paper/paper.pdf` (built from `paper/paper.md`, which is the source of truth). `paper/reproduction_record.md` is the full record behind it — 22 sections covering rounds 1-18, 45 generated tables — and the manuscript's 9 tables curate from it. |
| **2. Check it against its artifacts** | `./run.sh check` — seconds, needs no datasets, no torch, no GPU. Five gates: all 523 printed table cells regenerate from `data/results/` (242 in the manuscript, the rest in the record), every artifact is provenance-clean, all 46 `<!-- src: -->` tags resolve to real scripts, the committed PDF rebuilds with the same words. |
| **3. Spot-check a claim** | [Claim -> script -> artifact](#claim---script---artifact) maps every manuscript table and figure to the script and the JSON artifact behind it. Open the artifact; its `_provenance` block names the commit, script sha256, argv, and seed that produced it. |
| **4. Check what was fixed in advance** | `PREREGISTRATION*.md`, one per round, each committed before that round's labels were touched — the segmentation, the clustering, the manipulability metric, the selection attack, and each round's kill/keep criteria in writing. |
| **5. Re-run it** | `./run.sh` — the full reproduction. Resumable, and data-gated so a partial fetch still runs green. Then `git diff`: a clean diff *is* the reproduction, since regenerated artifacts overwrite the committed ones in place. |

Two scope facts bound every number below. **Manipulability is measured against a weak adversary** — selection over real turns, no policy trained — so each figure is a lower bound on what a generative adversary could find. And **no LLM judge is used anywhere**, because it shares the embedding proxies' blind spots; judging is against verified outcome labels and per-turn human ratings only.

Unexplained mismatches belong in an issue, with the failing gate's output and the `git diff` of the artifact in question — after checking the numerical-determinism caveat under [Requirements](#requirements), which accounts for last-decimal differences.

## Contents

- [The paper in brief](#the-paper-in-brief)
- [Where this sits: the three-paper line](#where-this-sits-the-three-paper-line)
- [The encoding: from conversation to reward input](#the-encoding-from-conversation-to-reward-input)
- [Data](#data)
- [Reproducing the paper](#reproducing-the-paper)
- [Claim -> script -> artifact](#claim---script---artifact)
- [The rounds](#the-rounds)
- [How the kit keeps the paper honest](#how-the-kit-keeps-the-paper-honest)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

## The paper in brief

### The question

A dense reward that scores a conversation on every turn is what reinforcement learning wants where the true outcome is sparse and late — and what Goodhart's law warns will be gamed. The question is whether a trajectory-geometric signal, which genuinely predicts failure *before* a task-oriented call ends, can be that reward.

### What it found

A trajectory-geometric early-warning predictor is **materially gameable by selection alone** — no training, just substituting in real agent turns that historically accompanied success suppresses 50-87% of its failure alerts — so it is a monitor, not a reward, and the RL optimizer is irrelevant to the vulnerability. No transform or combination of the adversary-controlled channel is robust.

But the exploitability is not a property of "content": it is a property of *which role the adversary can write*. A model optimizing the reward controls only its own turns; the user's turns, the tool calls, and the tool results are fixed evidence of what happened. Dropping only the writable role — scoring the fixed user/tool-call/tool-result channels — recovers ~97% of the accuracy at **zero** manipulability. Re-admitting the model's content only as its **conditional typicality** against that fixed evidence recovers essentially all of it while becoming mildly anti-gameable. And the whole thing is un-gameable *in substance* only when **conditioned on the task**, where the outcome is recomputable from the actions: fooling it requires actually completing the task.

### What it does not claim

Section 8 of the paper lists eight limitations, two of them declared scope decisions. The material ones:

- **The manipulability figures are lower bounds.** Selection over real turns cannot synthesize the novel turn a generative adversary could produce.
- **"Un-gameable" is scoped to the substitution adversary**, and the decisive substance result is scoped further to task-conditioning — the 82% task-blind slack is the standing reminder that a task-blind detector is fooled by generic success-shaped actions.
- **Substance needs a recomputable label.** The un-gameable-in-substance result exploits that one corpus's label (ABCD) is a deterministic function of the visible actions; the other corpus is a cleaner but weaker cross-check.
- **The recovered accuracy is outcome-adjacent** — it leans on tool-result content, what the tools actually returned. The paper argues that is a feature for a reward rather than a bug, and states it plainly either way.
- **Offline selection only.** No policy was trained and no RL loop was run.
- **Public role-play corpora**, predominantly Wizard-of-Oz or crowdworker role-play, not production traffic. The natural next step — the same construction on production calls with outcome-verified, task-labeled logs — is the one no public corpus can follow.
- **The domain boundary is real**: the head that replicates on dialogue collapses to chance on code-agent transcripts (SWE-agent).

## Where this sits: the three-paper line

Each repo is a self-contained, public reproduction kit built on the same harness.

| Repo | Role in the line | What it establishes |
|---|---|---|
| [`gp-paper`](https://github.com/suisuss/gp-paper) | Method | The Gram projection: what the encoding *is*, its exact identities, and that the readable basis costs nothing against the DCT. |
| [`actr-paper`](https://github.com/suisuss/actr-paper) | Representation | Which trajectory representation predicts satisfaction, the role-split recommendation, and the regime boundary. Closes by warning that the best predictor may be the most gameable reward. |
| **`mtgcr-paper`** (this repo) | **Reward** | Answers that warning — and builds the construction that survives it. |

Two concrete dependencies:

- **From `gp-paper`**: `src/lib/gram.py` is ported verbatim (minus unused decompositions) from that kit's `src/lib/encoders/gram.py`. The early-warning head here is the **degree-0 term** of that projection — which, by the method paper's first identity, is exactly the per-role mean. That the results below hold even at mean pooling is the point: the manipulability structure is a property of the *role split*, not of the geometric order.
- **From `actr-paper`**: the role-split representation, and the explicit hand-off — that paper recommends the best satisfaction predictor while flagging that it may be the most gameable reward. This paper is the answer to that flag. Rounds 1-3 also read `actr-paper`'s committed artifacts directly from a sibling checkout (see [Data](#data)).

## The encoding: from conversation to reward input

The construction is `gp-paper`'s; what this paper uses is its degree-0 term, applied per role and at every decision point.

**The early-warning head.** For each speaker role, mean-pool that role's turn embeddings into a per-role **running mean**, evaluated at every decision point of a conversation prefix; train an L2-regularized logistic head to predict eventual failure, out-of-fold and grouped by conversation.

```
G_0(role, prefix) = mean of that role's turn embeddings up to this point
head input        = concat over roles of G_0(role, prefix)
```

This is the simplest trajectory-geometric representation: the degree-0 (position) term, dropping the degree-1 *dynamics* — the direction the conversation is moving — that the predecessor emphasizes. Round 16 tests that dropped term directly and finds the dissociation: per-role velocity (degree-1 slope) is near-null for binary outcome (+0.006 AUC) but carries an order of magnitude more signal for per-turn satisfaction (+0.08 to +0.12 Spearman on USS). **Velocity geometry is a satisfaction signal, not an outcome signal.**

**The role split is the whole mechanism.** Roles divide into two classes:

| Class | Roles (ToolBench) | Who writes it |
|---|---|---|
| **Fixed evidence** | user, tool-call, tool-result | The world and the record — an optimizing policy cannot author these |
| **Adversary-controlled** | model / agent turns | The policy being optimized |

Every construction in the paper is a choice about which class enters the score, and how:

1. **Naive head** — all roles pooled. Strong predictor, manipulability 0.50-0.87. Gameable by selection alone.
2. **Fixed-role head (drop-assistant)** — score only the fixed evidence. ~97% of the accuracy, manipulability exactly 0 by construction.
3. **Plus conditional typicality** — re-admit the model's turns *only* as their consistency with the fixed evidence, never as free content. Recovers essentially all accuracy, mildly anti-gameable.
4. **Plus task-conditioning** — condition on the task, so the label is recomputable from the actions. Un-gameable in substance.

**The selection attack.** Optimization pressure is applied without training a policy: the reward is attacked by *selection over real turns* — substituting in agent turns drawn from the corpus that historically accompanied success. This is a weak adversary, so every manipulability number is a lower bound.

## Data

Six corpora across the rounds. Nothing is fetched implicitly — external data is gated behind `python main.py --fetch-data`, and every source is pinned to an exact revision in `KIT_PROVENANCE.json`.

| Corpus | Conversations | Label | Failure rate | Roles |
|---|---|---|---|---|
| ToolBench | 12,598 | verified give-answer / give-up | 20.2% | user, model, tool-call, tool-result |
| ABCD | 6,399 | action-completion (deterministic) | 13.4% | customer, agent, action |
| SpokenWOZ | 5,697 | derived goal-completion | 33.6% | user, system |
| SWE-agent-trajectories | 11,983 | PR-test-verified | 84.6% | user, agent, tool-result |
| USS-SGD / USS-MWOZ | 1,000 each | per-turn human ratings | — | user, system |

The main corpora carry a **verified or deterministically-derived** outcome label plus an explicit role structure — both are requirements, not conveniences: the role structure is what the construction turns on, and the verified label is what lets manipulability be judged without an LLM judge. USS-SGD/MWOZ carry per-turn human ratings and are used only for the judging methodology (§7) and the round-16 satisfaction contrast. SWE-agent is the domain boundary: the head that replicates on dialogue collapses to chance on code agents.

**Sources and pins** (all in `KIT_PROVENANCE.json`; each corpus remains under its own upstream terms, and none is redistributed here):

| Source | Pin |
|---|---|
| `Adorg/ToolBench` (HF mirror) | revision `0c25df5e6937aae854c2cc1bb203297600afce8b` |
| `asappresearch/abcd` | commit `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`, sha256 recorded |
| `nebius/SWE-agent-trajectories` (HF) | revision `68195a1450865274106246d0d0296a1d6807b88e` |
| `ssz1111/SpokenWOZ-Train-Text` (HF) | revision `d3aad10f2e5a37e7e1e84375f0db368b5872a044` |
| `ssz1111/SpokenWOZ-Test-Text-Fixed` (HF) | revision `d6c2d9e53b1005e327db582d327b69311092eb65` |
| Sibling `actr-paper` checkout | commit pin, verified at load time |

**The sibling checkout.** The round 1-3 stages read `actr-paper`'s committed artifacts (USS raw dumps, PRISM raw) from a sibling checkout at `../actr-paper`. `src/lib/actr_data.py` verifies that checkout's `HEAD` against `dataset_pins.actr_paper_commit` and **refuses to run on drift** — those numbers are only meaningful against the artifacts the predecessor actually published. The USS-SGD/MWOZ MiniLM embeddings are vendored into this kit (`data/embeddings/`, git-lfs) so the manuscript's Table 9 is self-contained; only the round 1-3 stages still need the sibling.

**Embedding models.**

| Model | Dim | Used for |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Primary encoder — every headline result; L2-normalized |
| `BAAI/bge-large-en-v1.5` | 1024 | §7 robustness: the main result repeated under a strong encoder |

**Vendored in this repo**: the canonical result artifacts the paper cites (`data/results/experiments/`, 57 canonical JSONs), the USS embeddings (git-lfs), the pre-registrations, and the reference figures and tables. Raw corpora and experiment caches are regenerated or downloaded, never tracked.

## Reproducing the paper

### Getting started

```bash
git clone https://github.com/suisuss/mtgcr-paper.git && cd mtgcr-paper   # a git clone, not a zip
./run.sh check      # seconds: validate the paper against its shipped artifacts
./run.sh            # full pipeline end to end (resumable; a killed run is safe to rerun)
python main.py status
```

`./run.sh check` regenerates every table, verifies it matches `paper/paper.md` and `paper/reproduction_record.md`, checks provenance is clean, confirms every `<!-- src: -->` tag resolves, and confirms the committed PDF reproduces. It needs only `requirements-core.txt` — no torch, no datasets, no GPU.

`./run.sh` in order: (1) creates `.venv` and installs `requirements.txt`; (2) installs the PDF toolchain via `install_pdf_tools.sh`; (3) runs `python main.py --fetch-data`, which acquires the pinned corpora and runs the whole pipeline, ending with the PDF.

**The pipeline is data-gated.** Stages join the DAG only once their data is present — a plain `python main.py` on a fresh clone without fetched corpora stays green rather than failing. `python main.py status` shows which stages are live, and lets you grow the reproduction corpus by corpus instead of all at once.

### What a full run costs

The pipeline (47 stages with every corpus fetched; fewer when the DAG is data-gated down) is dominated by the encode stages — embedding four corpora, tens of thousands of conversations — and by the attack sweeps; the analysis stages that read those caches are minutes each. Budget several hours on a GPU machine and considerably more on CPU, plus **~6GB of disk** for `data/` (raw corpora, embeddings, and ~1.6GB of checkpoint caches). `./run.sh check` needs none of it.

### Requirements

- Linux or macOS (POSIX file locking; on Windows use WSL).
- Python >= 3.10.
- A **git clone** — provenance stamping and the sibling-checkout pin use `.git`; a zip download will not work.
- **git-lfs**, for the vendored USS embeddings in `data/embeddings/`.
- A sibling `actr-paper` checkout at `../actr-paper`, at the pinned commit, if you intend to run the round 1-3 stages. Everything else is self-contained.
- Disk: several GB of raw corpora plus embeddings and caches for a full run (`data/` reaches ~6GB); `./run.sh check` needs none of it.
- GPU optional; embedding stages use it automatically when present.
- No API access at any point: judging is against verified labels and committed per-turn human ratings, never an LLM judge.
- PDF stage only: pandoc, XeLaTeX, rsvg-convert (`./install_pdf_tools.sh`).

**Numerical determinism.** Exact attack-manipulability numbers are bit-reproducible only under the pipeline's pinned single-thread BLAS (`OMP_NUM_THREADS=1`, set in `main.py`); a bare `python stage.py` can drift ~0.01-0.02 from multithreaded float-summation order. Trust `./run.sh check`, not an ad-hoc rerun. (Invariant arms report exactly 0 regardless.)

### Reading your reproduction

Regenerated artifacts overwrite the vendored ones in the working tree, so **`git diff` after a full run is a direct diff of your numbers against the paper's**. That is the intended way to read a reproduction — not a log to scan, but a diff that is empty when you have reproduced the paper.

## Claim -> script -> artifact

Every table and figure in the manuscript traces to one generator and to canonical artifacts under `data/results/experiments/`. Scripts live flat in `src/`; run them from `src/` (`cd src && python <script>`), or let `main.py` orchestrate everything. `manuscript_tables.py` renders the manuscript's tables from those artifacts — it computes nothing itself, so the "produced by" column is where the number is actually made.

| Manuscript element | Produced by | Canonical artifact(s) |
|---|---|---|
| Table 1. Corpora | `toolbench_preprocess.py`/`_encode.py`, `abcd_preprocess.py`, `spokenwoz_preprocess.py`, `swe_preprocess.py`, `ds_encode.py` | `data/embeddings/{toolbench,abcd,spokenwoz,swe}_meta.json` |
| Table 2. Early-warning accuracy by quartile | `tb_earlywarn.py`, `ds_experiments.py` | `tb_earlywarn.json`, `ds_experiments_{abcd,spokenwoz,swe}.json` |
| Table 3. The construction (ToolBench) | `skeleton_roles.py` | `skeleton_roles.json` |
| Table 4. Gaming the action grounding requires real success | `policy_choice_abcd.py` | `policy_choice_abcd.json` |
| Table 5. Naive-head manipulability + governance arms | `goodhart_matrix.py` | `goodhart_matrix_{toolbench,abcd,spokenwoz}.json` |
| Table 6. Every scheme that admits adversary content stays gameable | `tool_anchored.py`, `tool_relational.py`, `skeleton_roles.py` | `tool_anchored.json`, `tool_relational.json`, `skeleton_roles.json` |
| Table 7. Robustness — ABCD and BGE-large | `skeleton_roles_abcd.py`, `skeleton_roles_bge.py` | `skeleton_roles_abcd.json`, `skeleton_roles_bge.json` |
| Table 8. The accuracy/gameability frontier | `frontier_sweep.py` | `frontier_sweep_{toolbench,abcd,bge}.json` |
| Table 9. The satisfaction-shape alternative | `shape_reward.py` | `shape_reward_{sgd,mwoz}.json` |
| §4 early-warning head, its operating points | `tb3_head.py` | `tb3_head.json` |
| §4 lead time (two decision points on ToolBench, six on ABCD) | `round17.py` | `round17.json` |
| §6 adversary budget curve (+0.67 at the ceiling) | `adversary_sweep.py` | `adversary_sweep_{toolbench,abcd,bge}.json` |
| Figures 1-3 (role-split trajectory, the attack, the construction) | `paper_figures.py` | `paper_figures.json`, `paper/figures/` |
| The 45 tables of `paper/reproduction_record.md` | `paper_tables.py` | every artifact in `data/results/experiments/` |
| Both drafts checked cell by cell | `verify_draft_tables.py` | 523/523 cells |
| The PDF | `build_pdf.py` | `paper/paper.pdf` |

## The rounds

The investigation ran as pre-registered rounds; each has a `PREREGISTRATION-R*.md` closed before its run, and rounds 1-18 each have a section in `paper/reproduction_record.md` (round 19's frontier is reported in the manuscript, Table 8). The manuscript curates the surviving results; the record keeps the rest, including what died.

| Round | Corpus / question | Outcome |
|---|---|---|
| 1-3 | USS, PRISM: shared clustering, endpoint descriptives, sub-turn ladders | The zone framing loses to a plain head; "ends close" is a goodbye detector; structure-beyond-content is null. A goal-hugging inversion is found and then *retracted* by its own registered follow-up. |
| 4 | ABCD, SWE-agent | The head replicates on dialogue and collapses to chance on code agents — a domain boundary. |
| 5 | SpokenWOZ; the Goodhart attack | Survives voice transfer, attenuated. The head is gameable by selection: ABCD 84%, ToolBench 50%. |
| 6-7 | The reward-arm matrix; the shape-target arm judged by per-turn humans | Governance separates but leaves a residual; safety-strength tradeoff; the trend reward is Goodhart-divergent. |
| 8 | Convergence-phase follow-ups | Recorded in the reproduction record, §12. |
| 9-13 | The tool-anchoring road | Tethering fails; relational binding fails; restriction to the fixed roles + conditional typicality succeeds — un-gameable in substance iff task-conditioned; replicated on ABCD and under bge-large. |
| 14 | Tool-result controls | The fixed-role recovery is not error-counting: drop-assistant (0.769) beats a running error-rate baseline (0.649) by 0.12 AUC with disjoint bootstrap CIs. Enriching the tool-identity encoding and adding the raw payload embedding both add little — the signal is in the returned evidence, not in dressing it up. Literal tool descriptions (MCP) remain an untested gap. |
| 15 | Structure beyond the mean | Three orthogonal attempts to exceed the fixed-role head (0.769) with more fixed-channel structure — order/recency of results, continuous goal alignment, cross-channel consistency — each add a real but sub-threshold effect (+0.005 to +0.009; none clears 0.02). The degree-0 fixed-channel head sits near an accuracy ceiling. |
| 16 | Degree-1 role-split | Per-role velocity is near-null for binary outcome (+0.006 AUC) but carries an order of magnitude more signal for per-turn satisfaction (+0.08 to +0.12 Spearman on USS) — a clean level-vs-velocity dissociation, though even there it does not beat reading the current turn's content. |
| 17 | The two-mode deployment | The running early-warning is genuine: at matched false-alarm rate it catches more failures two-plus steps early than a same-turn-content baseline (37% vs 29% on ToolBench, 40% vs 35% on the longer ABCD dialogues, median lead six turns). Degree-1 velocity adds nothing early or late, so the deployment is one degree-0 head read at two positions. |
| 18-19 | Adversary strength to the selection ceiling; the accuracy/gameability frontier | The typicality residual's anti-gameability is not structural — it erodes at the exhaustive ceiling on ToolBench and is flat at zero on ABCD; the frontier maps the design space (Table 8), and the naive-head budget curve is round 18's alone. |

Rounds 6-7 and 12 run a design inherited from an earlier, non-public repository (issue shorthand `turn-encoding#92`, which is how the pre-registrations and two source files — `toolbench_preprocess.py`, `tool_relational.py` — refer to it): reward *governance via channel-factorizable features* — score one channel, monitor another, tripwire a third — with its comparator matrix, minimum discriminating arm set, and matched-predictor reporting rule. The one deliberate change is the instrument: that design's own spike found its LLM judge more arc-biased than the reward was hackable, so these rounds judge against verified labels and per-turn human ratings instead. Its ToolBench preprocess is vendored here in full (source commit pinned in `KIT_PROVENANCE.json` as `toolbench_turn_encoding_commit`), and the citing files are left byte-for-byte as they were when they produced their artifacts: a pre-registration edited after its round is no longer evidence of what was fixed in advance, and the two scripts still hash to the `script_sha256` recorded in their results.

## How the kit keeps the paper honest

The premise of the kit: a paper should be confident it reproduces itself *while being developed*, not just at the end. Four mechanisms in `src/lib/` carry that — **resumability, caching, provenance, and integrity checking** — each enforced by the pipeline rather than left to discipline. That matters here specifically: a manipulability number is a claim that someone *could not* game a reward, accumulated over many rounds of pre-registered work.

### The discipline beyond the harness

Two rules are this paper's own; the draft-vs-artifact gate and per-artifact provenance below are the harness's.

- **Pre-registration.** Researcher degrees of freedom are fixed in a committed `PREREGISTRATION-R*.md` before any outcome-correlated result is seen. Kill conditions and confound bars are written in advance.
- **Adversarial, verified-label judging.** Manipulability is measured with a weak selection adversary (lower bounds) and judged against *verified outcome labels and per-turn human ratings* — never an LLM judge, which shares the embedding proxies' blind spots (round 7 documents a case where the LLM judge is more biased toward the exploit than the reward is exploitable).

### The stage contract

Everything else hangs off this. Every script in `src/` is an importable module exporting up to three callables:

| Callable | Role |
|---|---|
| `run(**params) -> dict` | Computes and persists via `lib.campaign.save_result()` |
| `run_sample(max_samples, ...)` | Optional ephemeral sandbox alias — never writes results or caches |
| `main(argv)` | Thin CLI shim |

Importing a module never executes work. That is what lets `main.py` compile the stage DAG, and `src_check.py` resolve every claim in the paper to a script, without running anything. Stages are registered in `main.py`'s `STAGES` with their dependencies.

### Resumability — a killed run is always safe to rerun

`lib/pipeline.py` is a detached, restart-recoverable runner. Stage state lives in `.reproduce/main/`: `pipeline.json` is the compiled stage list, `pipeline.state.json` tracks per-stage status, and `logs/` holds per-stage output.

Re-run `./run.sh` after any interruption and completed stages are skipped; `python main.py status` shows the stage table. A file lock refuses a second concurrent supervisor on the same state, `lib/ioutils.py` writes every file atomically — an interrupted write cannot leave a partial file that `exists()` later reports as good — and `lib/runtime.py` caps CPU, memory, and process priority so a long run does not starve the host.

**The DAG is data-gated.** Stages join only once their fetched corpus is present on disk, so a fresh clone with no data runs green rather than failing — you can validate the paper, and grow the pipeline corpus by corpus, without ever hitting a wall of missing-data errors.

### Caching and invalidation — caches that expire themselves

Every expensive stage checkpoints under `data/experiment-caches/<script>/` through `lib.campaign.Checkpoint`, stamped in `producer.json` with the producing script's **logic digest**: a hash of the parsed AST, not of the file bytes. Comment- and formatting-only edits leave it unchanged and do not wipe cached checkpoints; only a change to the code itself re-invalidates. If the source does not currently parse (mid-edit), it falls back to a byte hash.

The logic digest is separate from the file sha256 recorded in each artifact's `_provenance`: one is for *cache correctness*, the other for *audit*.

**Never hand-delete a cache to force freshness**: cache validity derives from the code, not from memory of what you last edited. With 1.6GB of caches over encode-once corpora, that is what makes editing a late-stage script cheap and safe.

### Provenance — every number knows which code produced it

`lib/provenance.py` stamps every artifact with the git commit, the dirty flag, the script sha256, and the argv. `save_result()` writes both the canonical flat copy the tables read and an immutable provenance-stamped copy in the run history:

```
data/results/experiments/<name>.json                          canonical latest (tracked)
data/results/experiments/<name>/<dataset>/<config>/           immutable run history
```

Verify the whole set with `cd src && python -m lib.campaign verify`. The gate is that every artifact was built from a clean checkout: `git_dirty` must be `false`, and the commit and script sha256 must be present. **A number from a dirty tree does not count** — which is the same standing evidence rule the pre-registrations are written against.

These stamps come from the development history that produced the numbers; this repository ships as a single commit, so a shipped artifact's recorded commit identifies the code that generated it rather than resolving against the files beside it.

Provenance fixes *which code* ran; the pinned single-thread BLAS (`OMP_NUM_THREADS=1`, set in `main.py`) fixes *what it computes*. Multithreaded float-summation order is nondeterministic enough to drift manipulability figures ~0.01-0.02, so the pipeline pins it. This is why `./run.sh check` is authoritative and a bare `python stage.py` is not.

### Integrity checking — the five gates

`./run.sh check` runs all five in seconds, on core dependencies only — no torch, no datasets, no GPU.

| Gate | What it proves |
|---|---|
| 1. Environment | `.venv` builds from `requirements-core.txt` alone |
| 2. `verify_draft_tables.py` | Every table cell regenerated from `data/results/` equals the cell printed in `paper/paper.md` and `paper/reproduction_record.md` (523 cells: 242 manuscript, 281 record) |
| 3. `lib.campaign verify` | Every artifact is provenance-clean (clean checkout, commit and script sha256 present) |
| 4. `src_check.py` | Every `<!-- src: -->` tag resolves to a real script, every figure is tagged, every cited artifact is clean (46 tags) |
| 5. `verify_pdf_matches.py` | The committed `paper/paper.pdf` rebuilds from source with matching content |

Every generated figure and table carries a `<!-- src: script.py -->` tag, which is what makes gates 2 and 4 mechanical rather than manual. The PDF gate is content-fidelity, not a byte diff: XeLaTeX output is not byte-reproducible across TeX Live versions, so it compares the *multiset of words* in the rebuild against the committed PDF — metadata, whitespace, hyphenation, word order, and pagination all normalize away, so it fails exactly when a word changed, i.e. when the PDF is stale.

The gated record is unusually large: `paper/reproduction_record.md` carries 22 sections and 45 tables, every cell verified against its artifact. The manuscript curates from it, so the manuscript cannot print a number the record does not hold.

### External data is never fetched implicitly

Every corpus revision is pinned in `KIT_PROVENANCE.json` and every fetch is gated behind `--fetch-data`. The sibling `actr-paper` checkout is treated the same way: its `HEAD` is verified against the pin at load time and the loaders refuse to run on drift, so a round 1-3 result can never be computed against artifacts other than the ones the predecessor published.

Built from the same `paper-template` reproduction kit as the other two papers in the line, which carries the pipeline, checkpointing, provenance, and draft-gate machinery; `src/lib/campaign.py`'s docstring is the authoritative contract.

## Repository layout

```
run.sh                       one-command entry point (env, PDF tools, data, pipeline)
install_pdf_tools.sh         installs pandoc / xelatex / rsvg-convert
main.py                      the reproduction driver; STAGES is the data-gated DAG
KIT_PROVENANCE.json          dataset pins + the sibling actr-paper commit pin
requirements.txt             full deps (torch, sentence-transformers, pyarrow)
requirements-core.txt        light deps -- what `./run.sh check` installs

paper/
  paper.md                   the manuscript (source of truth for the PDF)
  paper.pdf                  built artifact, tracked and gated
  reproduction_record.md     the full machine-checked record the manuscript curates from
  figures/, tables/          generated, each tagged with its producing script

PREREGISTRATION*.md          one per round, closed before its run

src/                         stage scripts, flat; run as `cd src && python <script>`
  spike1_*, spike2_*, spike3*  rounds 1-3: zone clustering, endpoint descriptives,
                             sub-turn ladders (read the sibling actr-paper kit)
  toolbench_preprocess.py, toolbench_encode.py, tb_*.py, tb3_*.py
                             the ToolBench chain: preprocess -> encode -> features
                             -> endpoint/ladder/early-warning -> error/confound/head
  abcd_*.py, swe_preprocess.py, spokenwoz_preprocess.py, ds_encode.py, ds_experiments.py
                             round 4-5 corpora: preprocess, encode, run the head
  goodhart_attack.py, goodhart_matrix.py, adversary_sweep.py, frontier_sweep.py
                             the selection attack, the reward-arm matrix, budget curves
  tool_anchored*.py, tool_relational.py, skeleton_*.py
                             the tool-anchoring road: tethering, relational binding,
                             and the fixed-role skeleton that works (incl. bge + ABCD)
  round14.py .. round17.py   tool-result controls, structure beyond the mean,
                             degree-1 role-split, the two-mode deployment
  shape_reward.py, endpoint_shape.py, gram_equivalence.py, policy_choice_abcd.py
  manuscript_tables.py       renders the manuscript's 9 tables from the artifacts
  paper_tables.py            renders the reproduction record's 45 tables
  paper_figures.py           generates the three manuscript figures
  verify_draft_tables.py     the draft-vs-artifact gate (both documents)
  verify_pdf_matches.py      the committed-PDF content check
  src_check.py               resolves every <!-- src: --> tag to a real script
  build_pdf.py               pandoc/XeLaTeX build
  lib/                       shared infrastructure (below)

data/
  results/experiments/       canonical artifacts (tracked; what tables/figures read)
  embeddings/                vendored USS embeddings (git-lfs); other corpora regenerated
  raw/                       fetched corpora: toolbench, abcd, swe_trajectories, spokenwoz
  experiment-caches/         checkpointed intermediates, keyed by script logic digest
  tables/                    generated table sources
```

### `src/lib/` — the shared infrastructure

| Module | What it does |
|---|---|
| `gram.py` | **The Gram fit**, ported verbatim from `gp-paper`'s `encoders/gram.py`. The degree-0 term is this paper's head input. |
| `actr_data.py` | Loaders for the sibling `actr-paper` kit's committed artifacts. Verifies that checkout's `HEAD` against the pin and refuses to run on drift. |
| `campaign.py` | The stage contract: `run()` / `run_sample()` / `main()`, plus `save_result()` and `Checkpoint`. Read its docstring before writing a stage. |
| `pipeline.py` | Detached, restart-recoverable pipeline runner — what makes a killed run safe to rerun. |
| `provenance.py` | Stamps git commit, dirty flag, script sha256, and argv on every result. |
| `results.py` | Results registry: JSON index plus immutable per-run result files. |
| `runtime.py` | CPU/memory/priority limits so a run does not starve the host. |
| `ioutils.py` | Atomic file writes, so an interrupted run cannot leave a partial file that `exists()` reports as good. |
| `device.py` | GPU-if-present device selection for the embedding stages. |

## Troubleshooting

- **`actr-paper HEAD ... != pinned ...`**: the sibling checkout has drifted from `dataset_pins.actr_paper_commit`. Check out the pinned commit in `../actr-paper`, or update the pin deliberately if the drift is intended.
- **Stages missing from `status`**: their data has not been fetched. The DAG is data-gated; run `python main.py --fetch-data`.
- **Manipulability numbers differ in the last decimals**: you ran a stage directly rather than through the pipeline. See the numerical-determinism note under [Requirements](#requirements).
- **`data/embeddings/*.npz` are tiny text files**: git-lfs is not installed or not pulled — run `git lfs pull`.
- **PDF stage fails**: `./install_pdf_tools.sh`, then re-run — it resumes at that stage.
- **Provenance errors about git**: you are not in a git clone, or the repo has no commits — re-clone, don't unzip.
- **A gate fails after an honest edit**: expected — it means the committed artifact or PDF is now stale. Re-run the pipeline and commit what it regenerates.

## Citation and license

The preprint is in preparation; this entry will be replaced with the posted version when it is available.

```bibtex
@unpublished{sussmilch2026manipulability,
  title  = {Manipulability of Trajectory-Geometric Conversation Rewards
            and a Role-Restricted, Task-Conditioned Construction},
  author = {Sussmilch, Jacob},
  year   = {2026},
  note   = {Preprint in preparation. Reproduction kit:
            https://github.com/suisuss/mtgcr-paper}
}
```

When citing a reproduction, cite the commit you ran — the artifacts, both drafts, and the PDF are gated against each other at every commit.

Code and manuscript are MIT-licensed (see `LICENSE`), copyright Heya Enterprises Pty Ltd. The corpora are not redistributed here and remain under their own upstream terms (see [Data](#data)).
