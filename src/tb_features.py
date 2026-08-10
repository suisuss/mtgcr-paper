"""Round-2 shared feature stage (label-blind): per-conversation typed
blocks, goal distances, thought recurrence, and the prefix samples for the
early-warning arm. All three ToolBench arms read this checkpoint
passively; edits to arm stages never re-trigger it. Definitions fixed in
PREREGISTRATION-R2.md. This stage never reads has_failure.

    python tb_features.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"

TYPES = ["user", "assistant", "tool_call", "tool_result"]
ERR_KEYWORDS = ["error", "invalid", "not found", "fail", "timeout",
                "exceeded", "unauthorized", "forbidden", "rate limit",
                "400", "401", "403", "404", "429", "500", "502", "503"]
DIM = 384


def _slope(E):
    m = len(E)
    t = np.arange(m, dtype=np.float32) / max(m - 1, 1)
    A = np.stack([np.ones(m, dtype=np.float32), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return coef[1]


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off, goals = z["embeddings"], z["offsets"], z["goal_embeddings"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    raw = json.loads(CONV.read_text())
    n = len(goals)

    overall_mean = np.zeros((n, DIM), np.float32)
    n_turns = np.zeros(n, np.float32)
    type_means = np.zeros((n, 4, DIM), np.float32)
    type_present = np.zeros((n, 4), np.float32)
    type_slopes = np.zeros((n, 4, DIM), np.float32)
    recurrence = np.zeros((n, 3), np.float32)
    goal_dist_types = np.zeros((n, 4), np.float32)
    goal_dist_overall = np.zeros(n, np.float32)

    px, perr, pgoal, pstep, ptau, pconv = [], [], [], [], [], []

    for i in range(n):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        st = np.array(meta["step_types"][i])
        g = goals[i]
        m = len(E)
        n_turns[i] = m
        overall_mean[i] = E.mean(0)
        goal_dist_overall[i] = 1.0 - float(E.mean(0) @ g
                                           / max(np.linalg.norm(E.mean(0)), 1e-9))
        for k, ty in enumerate(TYPES):
            idx = np.where(tt == ty)[0]
            if len(idx):
                type_means[i, k] = E[idx].mean(0)
                type_present[i, k] = 1.0
                goal_dist_types[i, k] = float(np.mean(1.0 - E[idx] @ g))
            if len(idx) >= 2:
                type_slopes[i, k] = _slope(E[idx])
        th = np.where((tt == "assistant") & (st == "thought"))[0]
        if len(th) >= 2:
            sims = (E[th] @ E[th].T)[np.triu_indices(len(th), k=1)]
            recurrence[i] = [sims.mean(), sims.max(), len(th)]
        else:
            recurrence[i] = [0.0, 0.0, len(th)]

        # prefix samples at tool_result turns
        err_flags = np.array(
            [any(k in (t["content"] or "").lower() for k in ERR_KEYWORDS)
             if t["turn_type"] == "tool_result" else False
             for t in raw[i]["turns"]], dtype=np.float32)
        type_onehot = np.stack([(tt == ty).astype(np.float32)
                                for ty in TYPES], axis=1)         # (m, 4)
        csum = np.cumsum(E[:, None, :] * type_onehot[:, :, None], axis=0)
        ccnt = np.cumsum(type_onehot, axis=0)                     # (m, 4)
        err_csum = np.cumsum(err_flags)
        tr_ccnt = ccnt[:, 3]
        for t in np.where(tt == "tool_result")[0]:
            means_t = csum[t] / np.maximum(ccnt[t][:, None], 1.0)  # (4, 384)
            px.append(means_t.reshape(-1))
            perr.append(err_csum[t] / max(tr_ccnt[t], 1.0))
            am = means_t[1]
            nrm = np.linalg.norm(am)
            pgoal.append(1.0 - float(am @ g / nrm) if nrm > 1e-9 else 1.0)
            pstep.append(tr_ccnt[t])
            ptau.append(t / max(m - 1, 1))
            pconv.append(i)

    ckpt = Checkpoint("tb_features", script=__file__)
    ckpt.save("conv", {
        "overall_mean": overall_mean, "n_turns": n_turns,
        "type_means": type_means, "type_present": type_present,
        "type_slopes": type_slopes, "recurrence": recurrence,
        "goal_dist_types": goal_dist_types,
        "goal_dist_overall": goal_dist_overall})
    ckpt.save("prefix", {
        "X": np.array(px, np.float32), "err_share": np.array(perr, np.float32),
        "goal_dist": np.array(pgoal, np.float32),
        "step": np.array(pstep, np.float32), "tau": np.array(ptau, np.float32),
        "conv": np.array(pconv)})

    res = {"n_conversations": int(n), "n_prefix_samples": int(len(px)),
           "mean_thoughts": float(recurrence[:, 2].mean()),
           "prefix_err_share_mean": float(np.mean(perr))}
    save_result("tb_features", res, dataset="toolbench",
                params={"types": TYPES, "err_keywords": ERR_KEYWORDS})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
