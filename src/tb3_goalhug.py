"""Round-3 stage C: goal-hugging mechanism tests (a) quote-the-query
exclusion, (b) group stratification, (c) USS transfer proxy, (d) the
descriptive tau-decile curve. Kill rules fixed in PREREGISTRATION-R3.md.

    python tb3_goalhug.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from lib.campaign import Checkpoint, save_result
from lib.actr_data import load_uss, band

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]


def _auc_by_tau(y, score, tau, mask=None):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi)
        if mask is not None:
            m &= mask
        valid = m & ~np.isnan(score)
        out.append(float(roc_auc_score(y[valid], score[valid]))
                   if valid.sum() and len(set(y[valid])) == 2 else None)
    return out


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off, goals = z["embeddings"], z["offsets"], z["goal_embeddings"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    groups = np.array(meta["group"])
    n = len(y)

    p = Checkpoint("tb_features").load("prefix")
    tau, sconv = p["tau"], p["conv"]
    sy = y[sconv]

    # (a) exclusion of the k most recent assistant turns from the running
    # mean, recomputed at every prefix sample
    excl = {1: np.full(len(sy), np.nan), 2: np.full(len(sy), np.nan)}
    base_score = np.full(len(sy), np.nan)
    for i, (c, t_tau) in enumerate(zip(sconv, tau)):
        E = emb[off[c]:off[c + 1]]
        tt = np.array(meta["turn_types"][c])
        t = int(round(t_tau * (len(E) - 1)))
        a_idx = np.where(tt[:t + 1] == "assistant")[0]
        g = goals[c]

        def dist(idx):
            if not len(idx):
                return np.nan
            m = E[idx].mean(0)
            nrm = np.linalg.norm(m)
            return 1.0 - float(m @ g / nrm) if nrm > 1e-9 else 1.0

        base_score[i] = dist(a_idx)
        for k in (1, 2):
            excl[k][i] = dist(a_idx[:-k]) if len(a_idx) > k else np.nan

    res_a = {"base": _auc_by_tau(sy, base_score, tau),
             "excl1": _auc_by_tau(sy, excl[1], tau),
             "excl2": _auc_by_tau(sy, excl[2], tau)}

    # (b) group stratification
    sgroups = groups[sconv]
    res_b = {g: _auc_by_tau(sy, base_score, tau, mask=(sgroups == g))
             for g in ["G1_answer", "G2_answer", "G3_answer"]}

    # (c) USS transfer proxy: goal proxy = first user turn embedding
    res_c = {}
    for corpus in ("sgd", "mwoz"):
        convs = load_uss(corpus)
        feats, labels, taus = [], [], []
        for conv in convs:
            b = band(conv)
            if b is None:
                continue
            E, roles = conv["emb"], conv["roles"]
            u = np.where(roles == "USER")[0]
            proxy = E[u[0]] if len(u) else E[0]
            for t in range(len(E)):
                if roles[t] != "USER":
                    continue
                s = np.where(roles[:t + 1] == "SYSTEM")[0]
                if not len(s):
                    continue
                m = E[s].mean(0)
                nrm = np.linalg.norm(m)
                feats.append(1.0 - float(m @ proxy / nrm)
                             if nrm > 1e-9 else 1.0)
                labels.append(b)
                taus.append(t / max(len(E) - 1, 1))
        feats, labels, taus = (np.array(feats), np.array(labels),
                               np.array(taus))
        res_c[corpus] = _auc_by_tau(labels, feats, taus)

    # (d) descriptive curve: assistant-turn goal distance by tau decile
    deciles = np.linspace(0, 1, 11)
    curve = {"fail": [], "success": []}
    sums = {("fail", j): [] for j in range(10)}
    sums.update({("success", j): [] for j in range(10)})
    for i in range(n):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        a_idx = np.where(tt == "assistant")[0]
        if not len(a_idx):
            continue
        d = 1.0 - E[a_idx] @ goals[i]
        t = a_idx / max(len(E) - 1, 1)
        key = "fail" if y[i] else "success"
        for j in range(10):
            m = (t >= deciles[j]) & (t < deciles[j + 1] + (j == 9) * 1e-9)
            if m.any():
                sums[(key, j)].append(float(d[m].mean()))
    for key in ("fail", "success"):
        curve[key] = [float(np.mean(sums[(key, j)]))
                      if sums[(key, j)] else None for j in range(10)]

    killed = all(v is None or 0.45 <= v <= 0.55
                 for v in res_a["excl2"][1:])
    res = {"a_exclusion": res_a,
           "a_kill_rule_fired": bool(killed),
           "b_by_group": res_b,
           "c_uss_transfer": res_c,
           "d_decile_curve": curve,
           "n_samples": int(len(sy))}
    save_result("tb3_goalhug", res, dataset="toolbench", params={})
    print(json.dumps({k: v for k, v in res.items() if k != "d_decile_curve"},
                     indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
