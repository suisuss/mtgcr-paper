"""Round-3 stage A: corrected true-error baseline (repairs the void Bar 1).
Flag definition and Bar 1 test fixed in PREREGISTRATION-R3.md. Per-turn
flags are checkpointed for stages B-D.

    python tb3_error.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
EMB = REPO / "data" / "embeddings"

SEED = 42
N_BOOT = 1000
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]

ERROR_PATTERNS = [
    re.compile(r'"error"\s*:\s*"[^"]'),
    re.compile(r"status_code=4"),
    re.compile(r"status_code=5"),
    re.compile(r"HTTPSConnectionPool"),
    re.compile(r"Function executing from"),
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"timed out", re.IGNORECASE),
]


def true_error(content):
    return any(p.search(content or "") for p in ERROR_PATTERNS)


def _auc_by_tau(y, score, tau):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi)
        out.append(float(roc_auc_score(y[m], score[m]))
                   if m.sum() and len(set(y[m])) == 2 else None)
    return out


def run():
    raw = json.loads(CONV.read_text())
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)

    # per-turn true-error flags, aligned to the turn stream
    flags, offsets = [], [0]
    for c in raw:
        for t in c["turns"]:
            flags.append(t["turn_type"] == "tool_result"
                         and true_error(t["content"]))
        offsets.append(len(flags))
    flags = np.array(flags, dtype=np.float32)
    offsets = np.array(offsets)

    ckpt = Checkpoint("tb3_error", script=__file__)
    ckpt.save("flags", {"flags": flags, "offsets": offsets})

    # running true-error share at the R2 prefix samples
    p = Checkpoint("tb_features").load("prefix")
    tau, sconv = p["tau"], p["conv"]
    sy = y[sconv]
    share = np.zeros(len(sy), dtype=np.float32)
    for i, (c, t_tau) in enumerate(zip(sconv, tau)):
        f = flags[offsets[c]:offsets[c + 1]]
        tt = np.array(meta["turn_types"][c])
        t = int(round(t_tau * (len(f) - 1)))
        tr = np.where(tt[:t + 1] == "tool_result")[0]
        share[i] = f[tr].mean() if len(tr) else 0.0

    err_auc = _auc_by_tau(sy, share, tau)
    # per-sample share checkpointed; the Bar 1 head-vs-error CI test runs
    # in tb3_head, where the OOF head scores exist
    ckpt.save("prefix_share", {"share": share})

    # per-conversation stats for stages B-D reuse
    rate = np.array([flags[offsets[i]:offsets[i + 1]].sum()
                     / max((np.array(meta["turn_types"][i])
                            == "tool_result").sum(), 1)
                     for i in range(len(y))])

    res = {"n_turn_flags": int(flags.sum()),
           "turn_flag_rate_over_tool_results": float(
               flags.sum() / max((flags >= 0).sum(), 1)),
           "mean_conv_true_error_rate": float(rate.mean()),
           "conv_error_rate_auc": float(roc_auc_score(y, rate)),
           "err_share_auc_by_tau": err_auc,
           "n_samples": int(len(sy))}
    save_result("tb3_error", res, dataset="toolbench",
                params={"patterns": [p.pattern for p in ERROR_PATTERNS],
                        "seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
