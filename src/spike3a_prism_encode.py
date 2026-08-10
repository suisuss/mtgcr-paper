"""Spike 3a: segment PRISM model responses into sentences and encode them
with all-MiniLM-L6-v2. Segmentation and inclusion criteria are fixed in
PREREGISTRATION.md (split on ./!/? + whitespace, segments >= 10 chars,
responses with >= 3 surviving segments). The expensive product (sentence
embeddings) is checkpointed; feature construction lives in spike3b so its
edits do not invalidate this cache.

    python spike3a_prism_encode.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import Checkpoint, save_result
from lib.actr_data import prism_path

SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_SEG_CHARS = 10
MIN_SEGS = 3
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256


def segment(text):
    return [s.strip() for s in SPLIT.split(text or "")
            if len(s.strip()) >= MIN_SEG_CHARS]


def collect():
    sents, offsets, scores, convs, chosen = [], [0], [], [], []
    with open(prism_path()) as f:
        for ci, line in enumerate(f):
            rec = json.loads(line)
            for t in rec["conversation_history"]:
                if t.get("role") != "model" or "score" not in t:
                    continue
                segs = segment(t["content"])
                if len(segs) < MIN_SEGS:
                    continue
                sents.extend(segs)
                offsets.append(len(sents))
                scores.append(float(t["score"]))
                convs.append(ci)
                chosen.append(bool(t.get("if_chosen", False)))
    return (sents, np.array(offsets), np.array(scores),
            np.array(convs), np.array(chosen))


def run():
    ckpt = Checkpoint("spike3a_prism_encode", script=__file__)
    sents, offsets, scores, convs, chosen = collect()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    revision = getattr(getattr(model, "model_card_data", None),
                       "base_model_revision", None)
    emb = model.encode(sents, batch_size=BATCH, show_progress_bar=True,
                       normalize_embeddings=True,
                       convert_to_numpy=True).astype(np.float32)

    ckpt.save("sentences", {"emb": emb, "offsets": offsets,
                            "scores": scores, "convs": convs,
                            "chosen": chosen})
    res = {"n_responses": int(len(scores)), "n_sentences": int(len(sents)),
           "n_conversations": int(len(set(convs.tolist()))),
           "model": MODEL, "model_revision": revision,
           "mean_score": float(scores.mean()),
           "sd_score": float(scores.std(ddof=1))}
    save_result("spike3a_prism_encode", res, dataset="prism",
                params={"min_seg_chars": MIN_SEG_CHARS,
                        "min_segs": MIN_SEGS, "model": MODEL})
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
