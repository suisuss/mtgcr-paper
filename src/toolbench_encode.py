"""ToolBench MiniLM encoding with the first-class goal embedding.

Encodes every turn of every preprocessed ToolBench conversation with
all-MiniLM-L6-v2 (matching the USS/PRISM encodings), plus one goal
embedding per conversation from `goal_text` - the data amendment that
enables the goal-distance baselines across the charters.

Products (regenerated, not tracked):
    data/embeddings/toolbench_minilm.npz   embeddings, offsets, goal_embeddings
    data/embeddings/toolbench_meta.json    turn_types, has_failure, group,
                                           goal_text, n_turns, dialogue_id

    python toolbench_encode.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
EMB_DIR = REPO / "data" / "embeddings"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256


def run():
    if not CONV.exists():
        raise FileNotFoundError(
            f"{CONV} missing - run toolbench_preprocess.py first")
    conversations = json.loads(CONV.read_text())

    texts, offsets = [], [0]
    for c in conversations:
        texts.extend(t["content"] for t in c["turns"])
        offsets.append(len(texts))
    goals = [c["goal_text"] for c in conversations]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)
    goal_emb = model.encode(goals, batch_size=BATCH, show_progress_bar=True,
                            normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_DIR / "toolbench_minilm.npz", embeddings=emb,
             offsets=np.array(offsets), goal_embeddings=goal_emb)
    meta = {
        "dialogue_id": [c["dialogue_id"] for c in conversations],
        "turn_types": [[t["turn_type"] for t in c["turns"]]
                       for c in conversations],
        "step_types": [[t["metadata"].get("step_type", "")
                        for t in c["turns"]] for c in conversations],
        "has_failure": [c["has_failure"] for c in conversations],
        "group": [c["group"] for c in conversations],
        "goal_text": goals,
        "n_turns": [c["n_turns"] for c in conversations],
    }
    (EMB_DIR / "toolbench_meta.json").write_text(json.dumps(meta))

    res = {"n_conversations": len(conversations),
           "n_turns_total": int(len(texts)),
           "failure_rate": float(np.mean(meta["has_failure"])),
           "model": MODEL, "dim": int(emb.shape[1])}
    save_result("toolbench_encode", res, dataset="toolbench",
                params={"model": MODEL, "batch": BATCH})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
