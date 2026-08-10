"""SWE-agent-trajectories -> normalized schema, with the pre-registered
instance-level subsample (sample instance_ids uniformly, seed 42, until
>= 12,000 trajectories; every trajectory of a sampled instance included).

Role mapping (PREREGISTRATION-R4.md): system dropped; first user turn ->
'user' (carries the goal); later user turns -> 'tool_result'
(observations); ai -> 'assistant'. goal_text = the ISSUE block of the
first user message; step texts truncated at 500 chars. Label = `target`
(PR-test-verified). group_id = instance_id.

Writes data/raw/swe_trajectories/normalized.json.

    python swe_preprocess.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "swe_trajectories"
OUT = RAW / "normalized.json"

SEED = 42
TARGET_TRAJ = 12000
TRUNC = 500
ISSUE_RE = re.compile(r"ISSUE:\s*(.*?)(?:\n[A-Z][A-Z ]{3,}:|\Z)", re.DOTALL)


def goal_from_first_user(text):
    m = ISSUE_RE.search(text or "")
    return (m.group(1).strip() if m else (text or "").strip())[:2000]


def run():
    import pyarrow.parquet as pq
    files = sorted((RAW / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no parquet under data/raw/swe_trajectories"
                                "/data - python main.py --fetch-data")

    # pass 1: instance ids per row (cheap column read)
    ids = []
    for f in files:
        ids.append(pq.read_table(f, columns=["instance_id"])
                   .column(0).to_pylist())
    counts = {}
    for chunk in ids:
        for i in chunk:
            counts[i] = counts.get(i, 0) + 1
    rng = np.random.default_rng(SEED)
    instances = np.array(sorted(counts))
    rng.shuffle(instances)
    chosen, total = set(), 0
    for inst in instances:
        chosen.add(inst)
        total += counts[inst]
        if total >= TARGET_TRAJ:
            break

    out = []
    for f, chunk in zip(files, ids):
        keep_rows = [i for i, inst in enumerate(chunk) if inst in chosen]
        if not keep_rows:
            continue
        tbl = pq.read_table(f).to_pylist()
        for i in keep_rows:
            row = tbl[i]
            traj = row["trajectory"] or []
            turns = []
            seen_user = False
            for step in traj:
                role = step.get("role")
                text = (step.get("text") or "").strip()
                if role == "system" or not text:
                    continue
                if role == "user":
                    if not seen_user:
                        turns.append({"t": "user", "c": text[:TRUNC]})
                        seen_user = True
                    else:
                        turns.append({"t": "tool_result", "c": text[:TRUNC]})
                elif role == "ai":
                    turns.append({"t": "assistant", "c": text[:TRUNC]})
            if len(turns) < 3:
                continue
            first_user = next((s.get("text") for s in traj
                               if s.get("role") == "user"), "")
            out.append({"id": f"{row['instance_id']}::{len(out)}",
                        "group_id": row["instance_id"],
                        "goal_text": goal_from_first_user(first_user),
                        "label": bool(not row["target"]),  # has_failure
                        "turns": turns})
        del tbl

    OUT.write_text(json.dumps(out))
    n_fail = sum(c["label"] for c in out)
    res = {"n_trajectories": len(out),
           "n_instances": len(chosen),
           "failure_rate": n_fail / max(len(out), 1),
           "avg_turns": sum(len(c["turns"]) for c in out) / max(len(out), 1),
           "total_turns": sum(len(c["turns"]) for c in out)}
    save_result("swe_preprocess", res, dataset="swe",
                params={"seed": SEED, "target_traj": TARGET_TRAJ,
                        "trunc": TRUNC})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
