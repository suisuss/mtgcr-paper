"""Round-2 arm 1: goal-distance endpoint functionals (E1-E5) vs
has_failure. Family, direction, and FDR fixed in PREREGISTRATION-R2.md.
The endpoint charter's last arm: a null here parks the project.

    python tb_endpoint.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"


def _bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    passed = np.zeros(len(p), dtype=bool)
    below = p[order] <= q * (np.arange(1, len(p) + 1)) / len(p)
    if below.any():
        passed[order[:np.max(np.where(below)[0]) + 1]] = True
    return passed.tolist()


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off, goals = z["embeddings"], z["offsets"], z["goal_embeddings"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    n = len(y)

    vals = {k: np.full(n, np.nan) for k in
            ["E1_final_turn", "E2_final_thought", "E3_final_tool_result",
             "E4_min_goal_dist", "E5_min_tau"]}
    for i in range(n):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        st = np.array(meta["step_types"][i])
        d = 1.0 - E @ goals[i]
        m = len(E)
        vals["E1_final_turn"][i] = d[-1]
        th = np.where((tt == "assistant") & (st == "thought"))[0]
        if len(th):
            vals["E2_final_thought"][i] = d[th[-1]]
        tr = np.where(tt == "tool_result")[0]
        if len(tr):
            vals["E3_final_tool_result"][i] = d[tr[-1]]
        vals["E4_min_goal_dist"][i] = d.min()
        vals["E5_min_tau"][i] = np.argmin(d) / max(m - 1, 1)

    stats, pvals = {}, []
    for k, v in vals.items():
        mask = ~np.isnan(v)
        auc = float(roc_auc_score(y[mask], v[mask]))
        p = float(mannwhitneyu(v[mask][y[mask] == 1],
                               v[mask][y[mask] == 0],
                               alternative="two-sided").pvalue)
        stats[k] = {"auc": auc, "p": p, "n": int(mask.sum())}
        pvals.append(p)
    for k, ok in zip(stats, _bh_fdr(pvals)):
        stats[k]["fdr_pass"] = bool(ok)

    res = {"n": int(n), "failure_rate": float(y.mean()),
           "functionals": stats}
    save_result("tb_endpoint", res, dataset="toolbench", params={})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
