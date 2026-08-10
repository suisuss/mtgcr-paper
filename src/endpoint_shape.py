"""Round-8 Arm B: endpoint functionals against the degree-matched SHAPE
targets (y1 trend, y2 recovery), not just the degree-0 mean. The endpoint
charter's "ends close" died against satisfaction-mean and against binary
outcomes; this asks whether it predicts satisfaction TREND/RECOVERY.
PREREGISTRATION-R8.md.

Role-split degree-1 gap curve (canonical; gram==lstsq at degree 1, closed
round 3). Family: E1 raw final gap, E2 fitted ||d(1)||, E3 closest
approach, E4 initial gap. Spearman vs y0/y1/y2, BH-FDR across the 4x3
family per corpus.

    python endpoint_shape.py sgd
    python endpoint_shape.py mwoz
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from lib.campaign import save_result
from lib.actr_data import load_uss, load_uss_meta
from lib.gram import gram_fit

TAU_GRID = np.linspace(0.0, 1.0, 101)
FUNCTIONALS = ["E1_raw_final", "E2_fitted_end", "E3_closest", "E4_initial"]
TARGETS = ["y0_mean", "y1_trend", "y2_recovery"]


def _line(E):
    """Degree-1 least-squares line a + b*tau over the role's own tau."""
    m = len(E)
    t = np.arange(m, dtype=np.float64) / max(m - 1, 1)
    A = np.stack([np.ones(m), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, E.astype(np.float64), rcond=None)
    return coef[0], coef[1]


def _functionals(U, S):
    au, bu = _line(U)
    am, bm = _line(S)
    gap = (au - am)[None, :] + TAU_GRID[:, None] * (bu - bm)[None, :]
    norms = np.linalg.norm(gap, axis=1)
    return {"E1_raw_final": 1.0 - float(U[-1] @ S[-1]),
            "E2_fitted_end": float(norms[-1]),
            "E3_closest": float(norms.min()),
            "E4_initial": float(norms[0])}


def _series(turn_ratings):
    return np.array([float(np.mean(r)) for r in turn_ratings if r],
                    dtype=np.float64)


def _bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals)
    order = np.argsort(p)
    passed = np.zeros(len(p), dtype=bool)
    below = p[order] <= q * (np.arange(1, len(p) + 1)) / len(p)
    if below.any():
        passed[order[:np.max(np.where(below)[0]) + 1]] = True
    return passed


def run(corpus):
    convs = load_uss(corpus)
    meta = load_uss_meta(corpus)
    tr = meta["turn_ratings"]

    F = {k: [] for k in FUNCTIONALS}
    Y = {k: [] for k in TARGETS}
    for i, c in enumerate(convs):
        U = c["emb"][c["roles"] == "USER"].astype(np.float64)
        S = c["emb"][c["roles"] == "SYSTEM"].astype(np.float64)
        series = _series(tr[i])
        if len(U) < 2 or len(S) < 2 or len(series) < 3:
            continue
        fs = _functionals(U, S)
        yc = gram_fit(series[:, None], degree=2)
        for k in FUNCTIONALS:
            F[k].append(fs[k])
        Y["y0_mean"].append(yc[0, 0])
        Y["y1_trend"].append(yc[1, 0])
        Y["y2_recovery"].append(yc[2, 0])

    F = {k: np.array(v) for k, v in F.items()}
    Y = {k: np.array(v) for k, v in Y.items()}
    n = len(F["E1_raw_final"])

    cells = []
    for f in FUNCTIONALS:
        for t in TARGETS:
            rho, p = spearmanr(F[f], Y[t])
            cells.append({"functional": f, "target": t,
                          "rho": float(rho), "p": float(p)})
    passed = _bh_fdr([c["p"] for c in cells])
    for c, ok in zip(cells, passed):
        c["fdr_pass"] = bool(ok)

    res = {"corpus": corpus, "n": int(n), "cells": cells}
    save_result(f"endpoint_shape_{corpus}", res, dataset=corpus, params={})
    print(json.dumps({"corpus": corpus, "n": n,
                      "hits": [(c["functional"], c["target"],
                                round(c["rho"], 3))
                               for c in cells if c["fdr_pass"]]}, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
