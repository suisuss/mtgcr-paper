"""Generic encoder: normalized.json -> the kit's npz schema
(embeddings + offsets + goal_embeddings; meta with turn_types,
has_failure, group_id, goal_text, n_turns). Conversations with label
None (no label rule) are dropped at encode time so every downstream
stage sees labeled data only.

    python ds_encode.py abcd
    python ds_encode.py swe
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
EMB_DIR = REPO / "data" / "embeddings"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCES = {"abcd": REPO / "data" / "raw" / "abcd" / "normalized.json",
           "swe": REPO / "data" / "raw" / "swe_trajectories"
           / "normalized.json",
           "spokenwoz": REPO / "data" / "raw" / "spokenwoz"
           / "normalized.json"}


def run(dataset):
    convs = [c for c in json.loads(SOURCES[dataset].read_text())
             if c["label"] is not None]
    texts, offsets = [], [0]
    for c in convs:
        texts.extend(t["c"] for t in c["turns"])
        offsets.append(len(texts))
    goals = [c["goal_text"] for c in convs]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)
    gemb = model.encode(goals, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_DIR / f"{dataset}_minilm.npz", embeddings=emb,
             offsets=np.array(offsets), goal_embeddings=gemb)
    meta = {"id": [c["id"] for c in convs],
            "group_id": [c["group_id"] for c in convs],
            "turn_types": [[t["t"] for t in c["turns"]] for c in convs],
            "has_failure": [bool(c["label"]) for c in convs],
            "goal_text": goals,
            "n_turns": [len(c["turns"]) for c in convs]}
    if dataset == "abcd":
        meta["subflow"] = [c["subflow"] for c in convs]
        meta["flow"] = [c["flow"] for c in convs]
        meta["action_names"] = [[t.get("a", "") for t in c["turns"]]
                                for c in convs]
    (EMB_DIR / f"{dataset}_meta.json").write_text(json.dumps(meta))

    res = {"dataset": dataset, "n_conversations": len(convs),
           "n_turns_total": int(len(texts)),
           "failure_rate": float(np.mean(meta["has_failure"])),
           "model": MODEL}
    save_result(f"ds_encode_{dataset}", res, dataset=dataset,
                params={"model": MODEL})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
