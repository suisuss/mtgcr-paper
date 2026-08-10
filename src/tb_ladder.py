"""Round-2 arm 2: typed sub-turn ablation ladder on ToolBench, predicting
has_failure. Rungs, keep bar (+0.01 OOF AUC, bootstrap CI > 0), and
attribution rule fixed in PREREGISTRATION-R2.md. The reasoning-monitoring
line opens only on R3/R4 lift; R2 lift is typed content.

    python tb_ladder.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"

SEED = 42
N_BOOT = 1000
KEEP_DELTA = 0.01


def _oof(X, y):
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run():
    d = Checkpoint("tb_features").load("conv")
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    n = len(y)

    flat_means = d["type_means"].reshape(n, -1)
    flat_slopes = d["type_slopes"].reshape(n, -1)
    rungs = {}
    rungs["R1_content_length"] = np.hstack(
        [d["overall_mean"], d["n_turns"][:, None]])
    rungs["R2_typed_content"] = np.hstack(
        [rungs["R1_content_length"], flat_means, d["type_present"]])
    rungs["R3_typed_structure"] = np.hstack(
        [rungs["R2_typed_content"], flat_slopes])
    rungs["R4_dynamics_goal"] = np.hstack(
        [rungs["R3_typed_structure"], d["recurrence"],
         d["goal_dist_types"], d["goal_dist_overall"][:, None]])

    oof = {name: _oof(X, y) for name, X in rungs.items()}
    auc = {name: float(roc_auc_score(y, o)) for name, o in oof.items()}

    rng = np.random.default_rng(SEED)
    names = list(rungs)
    deltas = {f"{b}-{a}": [] for a, b in zip(names, names[1:])}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        rs = {nm: roc_auc_score(y[idx], oof[nm][idx]) for nm in names}
        for a, b in zip(names, names[1:]):
            deltas[f"{b}-{a}"].append(rs[b] - rs[a])

    ladder = {}
    for key, v in deltas.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        hi_name, lo_name = key.split("-")
        point = auc[hi_name] - auc[lo_name]
        ladder[key] = {"delta": float(point), "ci": [float(lo), float(hi)],
                       "keep": bool(point >= KEEP_DELTA and lo > 0)}

    res = {"n": int(n), "failure_rate": float(y.mean()),
           "oof_auc": auc, "ladder": ladder}
    save_result("tb_ladder", res, dataset="toolbench",
                params={"seed": SEED, "n_boot": N_BOOT,
                        "keep_delta": KEEP_DELTA})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
