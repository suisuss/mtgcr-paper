"""Round-3 stage B: E1/E3 error-content confound. Stratifies the endpoint
goal-distance AUCs by whether the final tool_result carries a true error;
interpretation rule fixed in PREREGISTRATION-R3.md.

    python tb3_confound.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off, goals = z["embeddings"], z["offsets"], z["goal_embeddings"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    n = len(y)

    f = Checkpoint("tb3_error").load("flags")
    flags, foff = f["flags"], f["offsets"]

    e1 = np.full(n, np.nan)
    e3 = np.full(n, np.nan)
    final_err = np.zeros(n, dtype=int)
    for i in range(n):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        d = 1.0 - E @ goals[i]
        e1[i] = d[-1]
        tr = np.where(tt == "tool_result")[0]
        if len(tr):
            e3[i] = d[tr[-1]]
            final_err[i] = int(flags[foff[i]:foff[i + 1]][tr[-1]] > 0)

    def stats(vals, mask):
        m = mask & ~np.isnan(vals)
        if len(set(y[m])) < 2:
            return {"auc": None, "p": None, "n": int(m.sum())}
        auc = float(roc_auc_score(y[m], vals[m]))
        p = float(mannwhitneyu(vals[m][y[m] == 1], vals[m][y[m] == 0],
                               alternative="two-sided").pvalue)
        return {"auc": auc, "p": p, "n": int(m.sum())}

    all_mask = np.ones(n, dtype=bool)
    no_err = final_err == 0
    with_err = final_err == 1

    def verdict(s):
        if s["auc"] is None:
            return "no-data"
        if s["auc"] < 0.55:
            return "error-detector"
        if s["auc"] >= 0.60 and s["p"] < 1e-3:
            return "signal-beyond-error"
        return "inconclusive"

    res = {
        "n": int(n),
        "final_err_rate": float(final_err.mean()),
        "pure_error_classifier_auc": float(roc_auc_score(y, final_err)),
        "E1": {"all": stats(e1, all_mask),
               "no_error_final": stats(e1, no_err),
               "error_final": stats(e1, with_err)},
        "E3": {"all": stats(e3, all_mask),
               "no_error_final": stats(e3, no_err),
               "error_final": stats(e3, with_err)},
    }
    res["E1"]["verdict"] = verdict(res["E1"]["no_error_final"])
    res["E3"]["verdict"] = verdict(res["E3"]["no_error_final"])
    save_result("tb3_confound", res, dataset="toolbench", params={})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
