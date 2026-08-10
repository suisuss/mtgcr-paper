# mtgcr-paper reproduction kit

This repository reproduces Heya's 'Manipulability of Trajectory-Geometric Conversation Rewards and a Role-Restricted, Task-Conditioned Construction' paper and its findings. It holds the manuscript source (`paper/paper.md`, built to `paper/paper.pdf`), the full machine-checked record the manuscript curates from (`paper/reproduction_record.md`), the pre-registrations that closed before each round, every result artifact, the scripts that produced them, and a pipeline that regenerates all of it from public data — plus gates that fail whenever a printed number and its artifact disagree.

## Run it

```bash
git clone https://github.com/suisuss/mtgcr-paper.git && cd mtgcr-paper   # a git clone, not a zip
./run.sh check             # seconds: validate both documents against their shipped artifacts
./run.sh                   # hours: regenerate every artifact from raw text, then the PDF
python main.py status      # which stages are live
```

`./run.sh` creates `.venv`, installs the PDF toolchain (`install_pdf_tools.sh`), fetches the pinned corpora, runs the 47 stages, and writes `paper/paper.pdf`. It is **resumable**: re-run after any interruption and completed stages are skipped. The DAG is also **data-gated** — stages join only once their corpus is present on disk, so a fresh clone with no data runs green rather than failing, and you can grow the reproduction corpus by corpus.

Regenerated artifacts overwrite the committed ones in place, so **`git diff` after a full run is your reproduction** — an empty diff means you reproduced the paper. Exact attack-manipulability numbers are bit-reproducible only under the pipeline's pinned single-thread BLAS (`OMP_NUM_THREADS=1`, set in `main.py`); a bare `python stage.py` can drift ~0.01-0.02 from multithreaded float-summation order, so trust `./run.sh check` over an ad-hoc rerun. Invariant arms report exactly 0 regardless.

`./run.sh check` needs no datasets, no torch, and no GPU — only `requirements-core.txt`. It runs five gates:

| Gate | What it proves |
|---|---|
| 1. Environment | `.venv` builds from `requirements-core.txt` alone |
| 2. `verify_draft_tables.py` | Every table cell regenerated from `data/results/` equals the cell printed in `paper/paper.md` and `paper/reproduction_record.md` (523 cells: 242 manuscript, 281 record) |
| 3. `lib.campaign verify` | Every artifact is provenance-clean (clean checkout, commit and script sha256 present) |
| 4. `src_check.py` | Every `<!-- src: -->` tag resolves to a real script, every figure is tagged, every cited artifact is clean (46 tags) |
| 5. `verify_pdf_matches.py` | The committed `paper/paper.pdf` rebuilds from source with matching content |

Unexplained mismatches belong in an issue, with the failing gate's output and the `git diff` of the artifact in question.

### Requirements

- Linux or macOS (POSIX file locking; on Windows use WSL); Python >= 3.10.
- A **git clone** — provenance stamping and the sibling-checkout pin use `.git`; a zip download will not work.
- **git-lfs**, for the vendored USS embeddings in `data/embeddings/`.
- Disk: `data/` reaches ~6GB for a full run (raw corpora, embeddings, ~1.6GB of caches); `./run.sh check` needs none of it.
- A sibling `actr-paper` checkout at `../actr-paper`, at the pinned commit, only if you intend to run the round 1-3 stages. Everything else is self-contained.
- GPU optional: embedding stages use it automatically when present.
- No API access at any point: judging is against verified outcome labels and committed per-turn human ratings, never an LLM judge.
- PDF stage only: pandoc, XeLaTeX, rsvg-convert — `./install_pdf_tools.sh` installs them (apt/brew).

## The paper

### What it argues

A dense reward that scores a conversation on every turn is what reinforcement learning wants where the true outcome is sparse and late — and what Goodhart's law warns will be gamed. A trajectory-geometric signal genuinely predicts failure *before* a task-oriented conversation ends, which makes it an obvious candidate. The paper argues that it cannot be used as a reward in that form, and that the reason is not the usual one: exploitability is not a property of scoring "content", it is a property of *which role the adversary can write*. A model optimizing the reward controls only its own turns; the user's turns, the tool calls, and the tool results are fixed evidence of what happened. Restricting the score to those fixed channels, re-admitting the model's turns only as their consistency with that evidence, and conditioning on the task is a construction that keeps the accuracy and removes the exploit.

The signal under test is the simplest trajectory-geometric one there is. For each speaker role, mean-pool that role's turn embeddings into a per-role running mean, evaluated at every decision point of a conversation prefix, and train an L2-regularized logistic head to predict eventual failure, out-of-fold and grouped by conversation:

```
G_0(role, prefix) = mean of that role's turn embeddings up to this point
head input        = concat over roles of G_0(role, prefix)
```

The roles divide into two classes, and every construction in the paper is a choice about which class enters the score:

| Class | Roles (ToolBench) | Who writes it |
|---|---|---|
| **Fixed evidence** | user, tool-call, tool-result | The world and the record — an optimizing policy cannot author these |
| **Adversary-controlled** | model / agent turns | The policy being optimized |

Optimization pressure is applied without training a policy: the reward is attacked by **selection over real turns**, substituting in agent turns drawn from the corpus that historically accompanied success. That is a weak adversary — it cannot synthesize the novel turn a generative one could — so every manipulability figure is a lower bound. Code: `src/lib/gram.py` (the degree-0 fit) and `src/goodhart_attack.py` (the attack). Full treatment: `paper/paper.md` §3-§5.

### What it finds

1. **The naive reward is materially gameable by selection alone.** No training — just substituting in real agent turns that historically accompanied success suppresses 50-87% of the head's failure alerts. It is a monitor, not a reward, and the RL optimizer is irrelevant to the vulnerability. No transform or combination of the adversary-controlled channel is robust.
2. **The fixed-role head recovers ~97% of the accuracy at zero manipulability.** Scoring only the fixed user/tool-call/tool-result channels is manipulability-0 *by construction*, and it is not error-counting: drop-assistant (0.769 AUC) beats a running error-rate baseline (0.649) with disjoint bootstrap CIs. It replicates on a second corpus and under a stronger encoder.
3. **Conditional typicality re-admits the model's content safely.** Scoring the model's turns only as their consistency with the fixed evidence — never as free content — recovers essentially all the accuracy while becoming mildly anti-gameable.
4. **Task-conditioning is what makes it un-gameable in substance.** Where the outcome is recomputable from the visible actions, fooling the reward requires actually completing the task; task-blind, a detector is still fooled by generic success-shaped actions.
5. **The early warning is genuine, and bounded by domain.** At matched false-alarm rate the head catches more failures two-plus decision points early than a same-turn-content baseline, and it survives transfer to spoken transcripts attenuated — but collapses to chance on code-agent trajectories.
6. **Level, not velocity.** Per-role velocity (the degree-1 slope) is near-null for binary outcome (+0.006 AUC) but carries an order of magnitude more signal for per-turn satisfaction (+0.08 to +0.12 Spearman) — a clean dissociation, and the reason the deployment is one degree-0 head read at two positions.

Scope and limitations are in Section 8 of the paper.

## Data

Six corpora across the rounds. Nothing is fetched implicitly; `python main.py --fetch-data` acquires whatever is missing, each pinned in `KIT_PROVENANCE.json` to the exact revision the paper used.

| Corpus | Conversations | Label | Failure rate | Roles |
|---|---|---|---|---|
| ToolBench | 12,598 | verified give-answer / give-up | 20.2% | user, model, tool-call, tool-result |
| ABCD | 6,399 | action-completion (deterministic) | 13.4% | customer, agent, action |
| SpokenWOZ | 5,697 | derived goal-completion | 33.6% | user, system |
| SWE-agent-trajectories | 11,983 | PR-test-verified | 84.6% | user, agent, tool-result |
| USS-SGD / USS-MWOZ | 1,000 each | per-turn human ratings | — | user, system |

The main corpora carry a **verified or deterministically-derived** outcome label plus an explicit role structure — both are requirements, not conveniences: the role structure is what the construction turns on, and the verified label is what lets manipulability be judged without an LLM judge. USS-SGD/MWOZ carry per-turn human ratings and are used only for the judging methodology and the level-vs-velocity contrast.

**Sources and pins**, all in `KIT_PROVENANCE.json`. None of the corpora is redistributed here; each remains under its own upstream terms and is fetched from the original source at the pinned revision.

| Source | Pin |
|---|---|
| `Adorg/ToolBench` (HF mirror) | revision `0c25df5e6937aae854c2cc1bb203297600afce8b` |
| `asappresearch/abcd` | commit `6b8700ce67c6b37b062dd7a60abc76d7ef832a97`, sha256 recorded |
| `nebius/SWE-agent-trajectories` (HF) | revision `68195a1450865274106246d0d0296a1d6807b88e` |
| `ssz1111/SpokenWOZ-Train-Text` (HF) | revision `d3aad10f2e5a37e7e1e84375f0db368b5872a044` |
| `ssz1111/SpokenWOZ-Test-Text-Fixed` (HF) | revision `d6c2d9e53b1005e327db582d327b69311092eb65` |
| Sibling `actr-paper` checkout | commit pin, verified at load time |

The round 1-3 stages read a sibling `actr-paper` checkout at `../actr-paper` for its committed artifacts (USS raw dumps, PRISM raw). `src/lib/actr_data.py` verifies that checkout's `HEAD` against `dataset_pins.actr_paper_commit` and **refuses to run on drift** — those numbers are only meaningful against the artifacts that kit actually published. The USS MiniLM embeddings are vendored here (`data/embeddings/`, git-lfs) so the manuscript's satisfaction table is self-contained; only rounds 1-3 still need the sibling.

Every headline result is `sentence-transformers/all-MiniLM-L6-v2` (384-d), L2-normalized; `BAAI/bge-large-en-v1.5` (1024-d) repeats the main result under a strong encoder. **Vendored here**: the canonical result artifacts (`data/results/experiments/`, 57 JSONs), the USS embeddings, the pre-registrations, and the reference figures and tables. Raw corpora and experiment caches are downloaded or regenerated, never tracked.

## Claim to Script to Artifact

Every table and figure traces to one generator and to canonical artifacts under `data/results/experiments/`. Scripts live flat in `src/`; run them from `src/` (`cd src && python <script>`), or let `main.py` orchestrate everything. Each artifact's `_provenance` block names the commit, script sha256, argv, and seed that produced it. `manuscript_tables.py` renders the manuscript's tables from those artifacts and computes nothing itself, so the "produced by" column is where the number is actually made.

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

The investigation ran as pre-registered rounds. Each has a `PREREGISTRATION-R*.md` closed before its run, and rounds 1-18 each have a section in `paper/reproduction_record.md` (round 19's frontier is reported in the manuscript). The manuscript curates the surviving results; the record keeps the rest, including what died — this table is the map between them.

| Round | Corpus / question | Outcome |
|---|---|---|
| 1-3 | USS, PRISM: shared clustering, endpoint descriptives, sub-turn ladders | Zone framing loses to a plain head; "ends close" is a goodbye detector; structure-beyond-content null; one inversion found and retracted by its own registered follow-up |
| 4 | ABCD, SWE-agent | The head replicates on dialogue, collapses to chance on code agents |
| 5 | SpokenWOZ; the Goodhart attack | Survives voice transfer, attenuated; gameable by selection (ABCD 84%, ToolBench 50%) |
| 6-7 | Reward-arm matrix; shape-target arm judged by per-turn humans | Governance separates but leaves a residual; the trend reward is Goodhart-divergent |
| 8 | Convergence-phase follow-ups | Recorded in the reproduction record, §12 |
| 9-13 | The tool-anchoring road | Tethering and relational binding fail; fixed roles + conditional typicality succeed — un-gameable in substance iff task-conditioned |
| 14 | Tool-result controls | The fixed-role recovery is not error-counting; enriched tool-identity encodings add little |
| 15 | Structure beyond the mean | Three orthogonal attempts each add +0.005 to +0.009, none clearing the 0.02 threshold |
| 16 | Degree-1 role-split | The level-vs-velocity dissociation |
| 17 | The two-mode deployment | Genuine early warning at matched false-alarm rate; one degree-0 head read at two positions |
| 18-19 | Adversary strength to the selection ceiling; the accuracy/gameability frontier | The typicality residual's anti-gameability is not structural; the frontier maps the design space |

Rounds 6-7 and 12 run a design inherited from an earlier, non-public repository (issue shorthand `turn-encoding#92`, which is how the pre-registrations and two source files — `toolbench_preprocess.py`, `tool_relational.py` — refer to it): reward governance via channel-factorizable features, with its comparator matrix, minimum discriminating arm set, and matched-predictor reporting rule. The one deliberate change is the instrument: that design's own spike found its LLM judge more arc-biased than the reward was hackable, so these rounds judge against verified labels and per-turn human ratings instead. Its ToolBench preprocess is vendored here in full (source commit pinned in `KIT_PROVENANCE.json` as `toolbench_turn_encoding_commit`), and the citing files are left byte-for-byte as they were when they produced their artifacts: a pre-registration edited after its round is no longer evidence of what was fixed in advance, and the two scripts still hash to the `script_sha256` recorded in their results.

## How the kit keeps the paper honest

The premise: a paper should be confident it reproduces itself *while being developed*, not just at the end. That matters here specifically — a manipulability number is a claim that someone *could not* game a reward, accumulated over many rounds. Two rules are the paper's own; the rest is the harness, enforced by the pipeline rather than left to discipline.

**Pre-registration and adversarial judging.** Researcher degrees of freedom are fixed in a committed `PREREGISTRATION-R*.md` before any outcome-correlated result is seen, kill conditions and confound bars in writing. Manipulability is measured with a weak selection adversary (so the figures are lower bounds) and judged against verified outcome labels and per-turn human ratings — never an LLM judge, which shares the embedding proxies' blind spots; round 7 documents a case where the judge was more biased toward the exploit than the reward was exploitable.

**The stage contract.** Every script in `src/` is an importable module exporting up to three callables: `run(**params) -> dict` computes and persists via `lib.campaign.save_result()`; `run_sample(max_samples, ...)` is an optional ephemeral sandbox alias that never writes results or caches; `main(argv)` is a thin CLI shim. Importing a module never executes work — which is what lets `main.py` compile the stage DAG, and `src_check.py` resolve every claim in the paper to a script, without running anything.

**Resumability.** `lib/pipeline.py` is a detached, restart-recoverable runner; stage state lives in `.reproduce/main/` (compiled stage list, per-stage status, per-stage logs). A file lock refuses a second concurrent supervisor, `lib/ioutils.py` writes every file atomically so an interrupted write cannot leave a partial file that `exists()` later reports as good, and `lib/runtime.py` caps CPU, memory, and process priority so a long run does not starve its host. A killed run is always safe to rerun.

**Caching that expires itself.** Expensive stages checkpoint under `data/experiment-caches/<script>/` through `lib.campaign.Checkpoint`, stamped in `producer.json` with the producing script's **logic digest**: a hash of the parsed AST, not of the file bytes, so comment- and formatting-only edits do not wipe cached checkpoints while a change to the code itself re-invalidates. That digest is for *cache correctness*; the file sha256 in each artifact's `_provenance` is for *audit*. **Never hand-delete a cache to force freshness**: validity derives from the code, not from memory of what you last edited — which, with 1.6GB of caches over encode-once corpora, is what makes editing a late-stage script cheap and safe.

**Provenance.** `lib/provenance.py` stamps every artifact with the git commit, dirty flag, script sha256, and argv. `save_result()` writes both the canonical flat copy the tables read (`data/results/experiments/<name>.json`) and an immutable stamped copy in the run history (`.../<name>/<dataset>/<config>/`). Verify the set with `cd src && python -m lib.campaign verify`: `git_dirty` must be `false`, and the commit and script sha256 present. **A number from a dirty tree does not count** — the same evidence rule the pre-registrations are written against. Provenance fixes *which code* ran; the pinned single-thread BLAS fixes *what it computes*.

**Integrity checking.** Every generated figure and table carries a `<!-- src: script.py -->` tag naming its producing script, which is what makes gates 2 and 4 mechanical rather than manual. The PDF gate is content-fidelity, not a byte diff — XeLaTeX output is not byte-reproducible across TeX Live versions, so it compares the *multiset of words* in the rebuild against the committed PDF; metadata, whitespace, hyphenation, word order, and pagination all normalize away, so it fails exactly when a word changed. The gated record is unusually large — `paper/reproduction_record.md` carries 22 sections and 45 tables, every cell verified against its artifact — and the manuscript curates from it, so the manuscript cannot print a number the record does not hold.

**External data is never fetched implicitly.** Every corpus revision is pinned in `KIT_PROVENANCE.json` and every fetch is gated behind `--fetch-data`. The sibling checkout is treated the same way: its `HEAD` is verified against the pin at load time and the loaders refuse to run on drift.

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

For `src/lib/`:
- `gram.py` is the Gram fit; its degree-0 term is this paper's head input.
- `actr_data.py` loads the sibling kit's committed artifacts, verifying that checkout's `HEAD` against the pin and refusing to run on drift.
- `campaign.py` holds the stage contract, `save_result()`, and `Checkpoint` — read its docstring before writing a stage.
- `pipeline.py` (the restart-recoverable runner), `provenance.py`, `results.py` (registry), `runtime.py` (CPU/memory/priority caps), `ioutils.py` (atomic writes), and `device.py` (GPU-if-present) are the harness.

## Troubleshooting

- **`actr-paper HEAD ... != pinned ...`**: the sibling checkout has drifted from `dataset_pins.actr_paper_commit`. Check out the pinned commit in `../actr-paper`, or update the pin deliberately if the drift is intended.
- **Stages missing from `status`**: their data has not been fetched. The DAG is data-gated; run `python main.py --fetch-data`.
- **Manipulability numbers differ in the last decimals**: you ran a stage directly rather than through the pipeline, so BLAS was multithreaded. Trust `./run.sh check`.
- **`data/embeddings/*.npz` are tiny text files**: git-lfs is not installed or not pulled — run `git lfs pull`.
- **PDF stage fails**: `./install_pdf_tools.sh`, then re-run — it resumes at that stage.
- **Provenance errors about git**: you are not in a git clone, or the repo has no commits — re-clone, don't unzip.
- **A gate fails after an honest edit**: expected — the committed artifact or PDF is now stale. Re-run the pipeline and commit what it regenerates.

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
