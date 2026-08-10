#!/usr/bin/env python3
"""One-command reproduction driver - the pattern to extend for a new paper.

Composes the paper's evidence chain into a single resumable pipeline and
runs it. Out of the box this ships two toy stages (src/example_stage_a.py,
src/example_stage_b.py) plus the table/PDF tail, so a fresh clone of this
template reproduces end to end with zero external data and zero network
access - that is the point: prove the harness works before you have real
experiments to hang off it, then replace the toy stages with your own.

Resumable. Stage status is persisted by lib/pipeline.py to
.reproduce/main/pipeline.state.json after every transition; re-run this
driver and completed stages are skipped, an interrupted stage re-runs. A
second concurrent supervisor on the same state is refused (a file lock).

    python main.py               run / resume
    python main.py launch        detached run (survives the shell)
    python main.py status        stage table
    python main.py reset         forget stage state
    python main.py --fetch-data  download external corpora first, then run
                                  (no-op until you wire up a real dataset -
                                  see fetch_dataset() below)

See README.md for the full contract (draft-vs-artifact gate, provenance,
run modes, `./run.sh check`).
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
DATA = REPO / "data"
RAW = DATA / "raw"
STATE_DIR = REPO / ".reproduce" / "main"
SPEC_PATH = STATE_DIR / "pipeline.json"
STATE_PATH = STATE_DIR / "pipeline.state.json"

PIPELINE_NAME = "reproduce"

# BLAS pinned to one thread so sklearn's joblib (or your own parallelism)
# owns it - an unpinned BLAS oversubscribing cores under joblib is a real
# way to wedge a stage for an hour. Add your own knobs here as needed.
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
       # 62 GB machine; the SWE battery peaks ~16 GB (prefix matrix +
       # sklearn float64 copies) which the 25%-of-RAM default cap kills.
       "EXPERIMENT_MAX_RAM_GB": "28"}


def _load_module(name, path):
    """Load a module by file path, bypassing the eager lib/__init__ (keeps
    status/reset dependency-free: lib/pipeline.py is stdlib-only)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinned_clone(url, dest, commit):
    """Shallow-fetch a pinned commit of a dataset repo. Falls back to a
    plain shallow clone and warns loudly if its HEAD drifts from the pin.

    Reusable boilerplate for a real --fetch-data stage: pin the exact
    commit/revision of every external corpus in KIT_PROVENANCE.json, and
    call this (or the HuggingFace-hub equivalent) from fetch_dataset()
    below so two reproductions see byte-identical raw data."""
    dest = Path(dest)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin",
                        url], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1",
                        "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q",
                        "FETCH_HEAD"], check=True)
        return
    except subprocess.CalledProcessError:
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[fetch] pinned fetch of {commit[:12]} failed; falling back "
              f"to a plain shallow clone of {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        print(f"[fetch] WARNING: {url} HEAD {head[:12]} differs from the "
              f"pinned commit {commit[:12]} - upstream moved; the data may "
              "not reproduce the paper's numbers exactly")


TOOLBENCH_ANSWER = RAW / "toolbench" / "answer"


def _fetch_abcd(pins):
    """ABCD at the pinned commit: the dataset gzip plus the guideline/kb
    files the TRACER label rule reads. sha256 recorded into
    KIT_PROVENANCE.json on first fetch, verified on later ones."""
    import hashlib
    import urllib.request
    dest = RAW / "abcd"
    files = ["abcd_v1.1.json.gz", "guidelines.json", "kb.json",
             "ontology.json"]
    if all((dest / f).exists() for f in files):
        print("[fetch] abcd already present")
        return
    dest.mkdir(parents=True, exist_ok=True)
    base = (f"https://raw.githubusercontent.com/{pins['abcd_repo']}/"
            f"{pins['abcd_commit']}/data/")
    for f in files:
        if (dest / f).exists():
            continue
        print(f"[fetch] abcd/{f}")
        urllib.request.urlretrieve(base + f, dest / f)
    digest = hashlib.sha256((dest / "abcd_v1.1.json.gz").read_bytes()
                            ).hexdigest()
    prov_path = REPO / "KIT_PROVENANCE.json"
    prov = json.loads(prov_path.read_text())
    pinned = prov["dataset_pins"].get("abcd_sha256")
    if pinned is None:
        prov["dataset_pins"]["abcd_sha256"] = digest
        prov_path.write_text(json.dumps(prov, indent=1) + "\n")
        print(f"[fetch] abcd sha256 recorded: {digest[:16]}...")
    elif pinned != digest:
        raise RuntimeError(f"abcd_v1.1.json.gz sha256 {digest[:16]} != "
                           f"pinned {pinned[:16]}")


def fetch_dataset(fetch: bool):
    """Pinned external corpora (KIT_PROVENANCE.json). Gated: nothing
    downloads unless --fetch-data was passed."""
    if not fetch:
        return
    pins = json.loads((REPO / "KIT_PROVENANCE.json").read_text())["dataset_pins"]
    from huggingface_hub import snapshot_download
    _fetch_abcd(pins)
    if not (RAW / "swe_trajectories" / "data").exists():
        print(f"[fetch] SWE trajectories @ "
              f"{pins['swe_trajectories_hf_revision'][:12]}")
        snapshot_download(repo_id=pins["swe_trajectories_hf_repo"],
                          repo_type="dataset",
                          revision=pins["swe_trajectories_hf_revision"],
                          local_dir=RAW / "swe_trajectories")
    else:
        print("[fetch] swe_trajectories already present")
    for key, rev_key, sub in (
            ("spokenwoz_train_text_hf", "spokenwoz_train_text_revision",
             "spokenwoz/train_text"),
            ("spokenwoz_test_text_hf", "spokenwoz_test_text_revision",
             "spokenwoz/test_text")):
        dest = RAW / sub
        if dest.exists():
            print(f"[fetch] {sub} already present")
            continue
        print(f"[fetch] {pins[key]} @ {pins[rev_key][:12]}")
        snapshot_download(repo_id=pins[key], repo_type="dataset",
                          revision=pins[rev_key], local_dir=dest)
    # Completeness check, not existence: an interrupted snapshot_download
    # leaves a partial answer/ tree, and the download is resumable.
    expected = {"G1_answer": 5000, "G2_answer": 4985, "G3_answer": 5000}
    if all((TOOLBENCH_ANSWER / d).is_dir()
           and len(list((TOOLBENCH_ANSWER / d).glob("*.json"))) >= n
           for d, n in expected.items()):
        print(f"[fetch] {TOOLBENCH_ANSWER} already complete")
        return
    pins = json.loads((REPO / "KIT_PROVENANCE.json").read_text())["dataset_pins"]
    from huggingface_hub import snapshot_download
    print(f"[fetch] ToolBench answer/ from {pins['toolbench_hf_repo']} "
          f"@ {pins['toolbench_hf_revision'][:12]}")
    snapshot_download(repo_id=pins["toolbench_hf_repo"],
                      repo_type="dataset",
                      revision=pins["toolbench_hf_revision"],
                      allow_patterns=["answer/*"],
                      local_dir=RAW / "toolbench")


# (name, argv, deps) - simple stage list. For a stage that regenerates
# something expensive (embeddings, a scraped corpus), add an "outputs"
# check like this and skip appending the stage when every output already
# exists - see actr-paper/gp-paper's main.py (EMBED_STAGES) for the full
# pattern if your pipeline grows past a handful of stages.
STAGES = [
    ("spike1_zone_clustering", ["spike1_zone_clustering.py"], []),
    ("spike2_endpoint_descriptive", ["spike2_endpoint_descriptive.py"], []),
    # Spike 3 needs the heavy stack (requirements.txt: torch +
    # sentence-transformers); 3a's product is checkpointed so 3b edits
    # never re-encode.
    ("spike3a_prism_encode", ["spike3a_prism_encode.py"], []),
    ("spike3b_subturn_ladder", ["spike3b_subturn_ladder.py"],
     ["spike3a_prism_encode"]),
    ("paper_tables", ["paper_tables.py"],
     ["spike1_zone_clustering", "spike2_endpoint_descriptive",
      "spike3b_subturn_ladder"]),
]

# ToolBench stages join the pipeline only once the gated fetch has run -
# a plain `python main.py` on a clone without the data stays green.
if TOOLBENCH_ANSWER.exists():
    STAGES = STAGES[:-1] + [
        ("toolbench_preprocess", ["toolbench_preprocess.py"], []),
        ("toolbench_encode", ["toolbench_encode.py"],
         ["toolbench_preprocess"]),
        # Round-2 arms, sequenced cheapest-first off one checkpointed
        # feature producer (PREREGISTRATION-R2.md).
        ("tb_features", ["tb_features.py"], ["toolbench_encode"]),
        ("tb_endpoint", ["tb_endpoint.py"], ["tb_features"]),
        ("tb_ladder", ["tb_ladder.py"], ["tb_endpoint"]),
        ("tb_earlywarn", ["tb_earlywarn.py"], ["tb_ladder"]),
        # Round-3 full experiments (PREREGISTRATION-R3.md)
        ("gram_equivalence", ["gram_equivalence.py"], ["tb_features"]),
        ("tb3_error", ["tb3_error.py"], ["tb_features"]),
        ("tb3_confound", ["tb3_confound.py"], ["tb3_error"]),
        ("tb3_goalhug", ["tb3_goalhug.py"], ["tb_features"]),
        ("tb3_head", ["tb3_head.py"], ["tb3_error"]),
    ] + STAGES[-1:]
    STAGES = [(nm, argv, deps + ["tb_earlywarn", "tb3_head", "tb3_confound",
                                 "tb3_goalhug", "gram_equivalence"])
              if nm == "paper_tables" else (nm, argv, deps)
              for nm, argv, deps in STAGES]

# Round-4 corpora (PREREGISTRATION-R4.md), gated on their fetched data.
_R4 = []
if (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
    _R4 += [("abcd_preprocess", ["abcd_preprocess.py"], []),
            ("abcd_escalation", ["abcd_escalation.py"], ["abcd_preprocess"]),
            ("ds_encode_abcd", ["ds_encode.py", "abcd"], ["abcd_preprocess"]),
            ("ds_experiments_abcd", ["ds_experiments.py", "abcd"],
             ["ds_encode_abcd"])]
if (RAW / "swe_trajectories" / "data").exists():
    _R4 += [("swe_preprocess", ["swe_preprocess.py"], []),
            ("ds_encode_swe", ["ds_encode.py", "swe"], ["swe_preprocess"]),
            ("ds_experiments_swe", ["ds_experiments.py", "swe"],
             ["ds_encode_swe"])]
# Round 5: SpokenWOZ voice-transfer arm + Goodhart attack (targets the
# round-4 corpora heads, so it depends on their encodings).
if (RAW / "spokenwoz" / "train_text" / "data.json").exists():
    _R4 += [("spokenwoz_preprocess", ["spokenwoz_preprocess.py"], []),
            ("ds_encode_spokenwoz", ["ds_encode.py", "spokenwoz"],
             ["spokenwoz_preprocess"]),
            ("ds_experiments_spokenwoz", ["ds_experiments.py", "spokenwoz"],
             ["ds_encode_spokenwoz"])]
if (EMB_DIR := RAW.parent / "embeddings").exists() or True:
    _R4 += [("goodhart_toolbench", ["goodhart_attack.py", "toolbench"],
             ["toolbench_encode"]),
            ("goodhart_abcd", ["goodhart_attack.py", "abcd"],
             ["ds_encode_abcd"]),
            # Round 6: the predecessor design's rung-1 reward-arm matrix
            ("goodhart_matrix_toolbench",
             ["goodhart_matrix.py", "toolbench"], ["toolbench_encode"]),
            ("goodhart_matrix_abcd", ["goodhart_matrix.py", "abcd"],
             ["ds_encode_abcd"]),
            # Round 7: shape-target reward, per-turn human judge (USS)
            ("shape_reward_sgd", ["shape_reward.py", "sgd"], []),
            ("shape_reward_mwoz", ["shape_reward.py", "mwoz"], []),
            # Round 8: SpokenWOZ manipulability + endpoints-vs-shape
            ("endpoint_shape_sgd", ["endpoint_shape.py", "sgd"], []),
            ("endpoint_shape_mwoz", ["endpoint_shape.py", "mwoz"], [])]
if (RAW / "spokenwoz" / "train_text" / "data.json").exists():
    _R4 += [("goodhart_matrix_spokenwoz",
             ["goodhart_matrix.py", "spokenwoz"],
             ["ds_encode_spokenwoz"])]
# Round 9: tool-anchored representation (ToolBench) + role-split(1) addendum
# Round 12: relational binding of content against the tool grounding
if (RAW / "toolbench" / "conversations.json").exists():
    _R4 += [("tool_anchored", ["tool_anchored.py"], ["toolbench_encode"]),
            ("tool_anchored_g1", ["tool_anchored_g1.py"], ["tool_anchored"]),
            ("tool_relational", ["tool_relational.py"], ["tool_anchored"]),
            ("skeleton_roles", ["skeleton_roles.py"], ["tool_anchored"]),
            ("skeleton_roles_bge", ["skeleton_roles_bge.py"],
             ["toolbench_encode"]),
            # Round 14: tool-result content baselines, enriched skeleton,
            # structured vs semantic tool result (PREREGISTRATION-R14.md).
            ("round14", ["round14.py"], ["toolbench_encode"]),
            # Round 15: order/recency, goal-conditioning, cross-channel
            # consistency on the fixed channels (PREREGISTRATION-R15.md).
            ("round15", ["round15.py"], ["toolbench_encode"]),
            # Round 16: degree-1 role-split vs degree-0, outcome (ToolBench)
            # and satisfaction (USS via the actr-paper checkout).
            ("round16", ["round16.py"], ["toolbench_encode"])]
# Round 17: the two-mode deployment. Direction A's lead-time replication reads
# ABCD, so gate on both corpora.
if (RAW / "toolbench" / "conversations.json").exists() \
        and (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
    _R4 += [("round17", ["round17.py"],
             ["toolbench_encode", "ds_encode_abcd"])]
# Round 18: adversary-strength sweep to the selection ceiling. Reuses the
# Round-13 setup (imports skeleton_roles unchanged); the n=16 point nests the
# published manipulability under a built-in guard (PREREGISTRATION-R18.md).
# R19's frontier subsumes the typicality arm; the naive-head (flat) budget
# curve is this stage's alone (reproduction_record.md section 22).
if (RAW / "toolbench" / "conversations.json").exists():
    _R4 += [("adversary_sweep_toolbench",
             ["adversary_sweep.py", "toolbench"], ["skeleton_roles"])]
# Round 19: accuracy/gameability frontier (PREREGISTRATION-R19.md); feeds the
# manuscript's Table 8. Reuses the round-13 setups.
if (RAW / "toolbench" / "conversations.json").exists():
    _R4 += [("frontier_sweep_toolbench",
             ["frontier_sweep.py", "toolbench"], ["skeleton_roles"]),
            ("frontier_sweep_bge",
             ["frontier_sweep.py", "bge"], ["skeleton_roles_bge"])]
if (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
    _R4 += [("frontier_sweep_abcd",
             ["frontier_sweep.py", "abcd"], ["skeleton_roles_abcd"])]
# Round 10: skeleton restriction on ABCD
if (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
    _R4 += [("skeleton_abcd", ["skeleton_abcd.py"], ["ds_encode_abcd"]),
            ("skeleton_roles_abcd", ["skeleton_roles_abcd.py"],
             ["skeleton_abcd"])]
# Round 11: policy-choice adversary (determinism + real-label attack)
if (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
    _R4 += [("skeleton_determinism_abcd",
             ["skeleton_determinism.py", "abcd"], ["ds_encode_abcd"]),
            ("policy_choice_abcd", ["policy_choice_abcd.py"],
             ["skeleton_abcd"])]
if (RAW / "toolbench" / "conversations.json").exists():
    # Figure 3 (the construction schematic) reads skeleton_roles.json and, when
    # present, policy_choice_abcd.json - depend on them so a full run regenerates
    # the figure after those numbers. skeleton_roles shares this ToolBench gate;
    # policy_choice_abcd is ABCD-gated, so add it only when ABCD data is present.
    _fig_deps = ["toolbench_encode", "skeleton_roles"]
    if (RAW / "abcd" / "abcd_v1.1.json.gz").exists():
        _fig_deps.append("policy_choice_abcd")
    _R4 += [("skeleton_determinism_toolbench",
             ["skeleton_determinism.py", "toolbench"], ["toolbench_encode"]),
            # Manuscript figures (paper/paper.md Figs 1-3), from one real
            # ToolBench conversation plus the construction artifacts.
            # Deterministic; must run before build_pdf.
            ("paper_figures", ["paper_figures.py"], _fig_deps)]
if _R4:
    tail = {"paper_tables", "verify_draft_tables", "build_pdf"}
    r4_names = [nm for nm, _, _ in _R4
                if nm.startswith("ds_experiments")
                or nm == "abcd_escalation"
                or nm.startswith("goodhart_")
                or nm.startswith("goodhart_matrix_")
                or nm.startswith("shape_reward_")
                or nm.startswith("endpoint_shape_")
                or nm == "tool_anchored"
                or nm == "tool_anchored_g1"
                or nm == "tool_relational"
                or nm == "skeleton_roles"
                or nm == "skeleton_roles_bge"
                or nm == "skeleton_roles_abcd"
                or nm == "skeleton_abcd"
                or nm.startswith("skeleton_determinism_")
                or nm == "policy_choice_abcd"
                or nm == "round14"
                or nm == "round15"
                or nm == "round16"
                or nm == "round17"
                or nm.startswith("adversary_sweep")
                or nm.startswith("frontier_sweep")]
    head_stages = [s for s in STAGES if s[0] not in tail]
    tail_stages = []
    for nm, argv, deps in STAGES:
        if nm not in tail:
            continue
        if nm == "paper_tables":
            deps = deps + r4_names
        tail_stages.append((nm, argv, deps))
    STAGES = head_stages + _R4 + tail_stages

# The manuscript embeds figures/*.svg, so the PDF build depends on the
# figure stage whenever it is in play (ToolBench data present).
_fig_dep = ["paper_figures"] if any(nm == "paper_figures"
                                    for nm, _, _ in STAGES) else []
# Manuscript tables (paper/paper.md Tables 1-9) are generated from the flat
# artifacts and gated against paper.md, the same way actr-paper/gp-paper do.
_r4dep = r4_names if _R4 else []
STAGES += [
    ("manuscript_tables", ["manuscript_tables.py"], _r4dep),
    # Draft-vs-artifact gate: fails the run if a regenerated table cell
    # diverges from its draft (reproduction_record.md for the spike tables,
    # paper.md for the manuscript tables). Runs after the tables regenerate.
    ("verify_draft_tables", ["verify_draft_tables.py"],
     ["paper_tables", "manuscript_tables"]),
    # Terminal stage: the preview PDF. Fails resumably with install
    # instructions when the pandoc/xelatex toolchain is absent.
    ("build_pdf", ["build_pdf.py"],
     ["paper_tables", "manuscript_tables", "verify_draft_tables"] + _fig_dep),
]
PDF_TOOLS = ("pandoc", "xelatex", "rsvg-convert")


def build_spec():
    stages = [{"name": name, "cmd": [sys.executable] + argv,
              "depends_on": deps} for name, argv, deps in STAGES]
    return {"name": PIPELINE_NAME, "workdir": str(SRC),
            "log_dir": str(STATE_DIR / "logs"), "env": ENV, "stages": stages}


# ------------------------------------------------------------------ preflight

def _pending(names):
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    return [n for n in names if state.get(n, {}).get("status") != "done"]


def preflight(fetch, stages):
    errors = []
    fetch_dataset(fetch)

    for mod in ("numpy",):
        if importlib.util.find_spec(mod) is None:
            errors.append(f"missing package '{mod}' "
                          "(pip install -r requirements.txt)")

    if _pending(["build_pdf"]):
        missing = [t for t in PDF_TOOLS if shutil.which(t) is None]
        if missing:
            print("[preflight] warning: PDF tools missing "
                  f"({', '.join(missing)}) - every artifact and table "
                  "still builds; the terminal build_pdf stage will "
                  "fail with install instructions until they are present "
                  "(run ./install_pdf_tools.sh)")

    return errors


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reproduce every cited artifact of the paper "
                    "(resumable; see README.md)")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "launch", "status", "reset"])
    ap.add_argument("--fetch-data", action="store_true",
                    help="download external corpora (if any) if missing, "
                    "then run")
    args = ap.parse_args(argv)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR.parent / ".gitignore").write_text("*\n")

    spec = build_spec()
    SPEC_PATH.write_text(json.dumps(spec, indent=1))

    pipeline = _load_module("pipeline", SRC / "lib" / "pipeline.py")

    if args.command == "status":
        return pipeline.status(SPEC_PATH)
    if args.command == "reset":
        if STATE_PATH.exists():
            STATE_PATH.unlink()
            print(f"cleared {STATE_PATH} - next run re-executes every stage "
                  "(experiment caches and saved artifacts are untouched, so "
                  "most stages replay their caches)")
        else:
            print("no state to clear")
        return 0

    errors = preflight(args.fetch_data, spec["stages"])
    if errors:
        print("[preflight] FAILED:", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 2

    n_pending = len(_pending([s["name"] for s in spec["stages"]]))
    print(f"[reproduce] {len(spec['stages'])} stages ({n_pending} pending); "
          f"state: {STATE_PATH}; logs: {STATE_DIR / 'logs'}")
    if args.command == "launch":
        return pipeline.launch(SPEC_PATH)
    return pipeline.run(SPEC_PATH)


if __name__ == "__main__":
    sys.exit(main())
