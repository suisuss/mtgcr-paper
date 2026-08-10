"""Spike 2: endpoint descriptive functionals (Q1) with the closing-ritual
controls. Functional family, distance, FDR, keyword list, and the
interpretation rule are fixed in PREREGISTRATION.md.

    python spike2_endpoint_descriptive.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr, rankdata

from lib.campaign import save_result
from lib.actr_data import load_uss, load_uss_text

TAU_GRID = np.linspace(0.0, 1.0, 101)
RITUAL_KEYWORDS = ["thank", "thanks", "goodbye", "bye", "welcome",
                   "good day", "great day", "nice day", "that's all",
                   "that is all", "have a good", "appreciate"]
FUNCTIONALS = ["F1_raw_final", "F2_fitted_end", "F3_closest_depth",
               "F4_closest_tau", "F5_initial_gap"]


def _fit_line(E):
    """Least-squares line a + b*tau per embedding dim over the role's own
    normalized positions; returns (a, b)."""
    m = len(E)
    t = np.arange(m) / max(m - 1, 1)
    A = np.stack([np.ones(m), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return coef[0], coef[1]


def _functionals(conv, strip=0):
    emb, roles = conv["emb"], conv["roles"]
    u_idx = np.where(roles == "USER")[0]
    s_idx = np.where(roles == "SYSTEM")[0]
    if strip:
        u_idx, s_idx = u_idx[:-strip], s_idx[:-strip]
    if len(u_idx) < 2 or len(s_idx) < 2:
        return None
    U, S = emb[u_idx], emb[s_idx]
    f1 = 1.0 - float(U[-1] @ S[-1])
    au, bu = _fit_line(U)
    am, bm = _fit_line(S)
    gap = (au - am)[None, :] + TAU_GRID[:, None] * (bu - bm)[None, :]
    norms = np.linalg.norm(gap, axis=1)
    return {"F1_raw_final": f1,
            "F2_fitted_end": float(norms[-1]),
            "F3_closest_depth": float(norms.min()),
            "F4_closest_tau": float(TAU_GRID[np.argmin(norms)]),
            "F5_initial_gap": float(norms[0])}


def _bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg pass/fail within the family."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    passed = np.zeros(len(p), dtype=bool)
    thresh = q * (np.arange(1, len(p) + 1)) / len(p)
    below = p[order] <= thresh
    if below.any():
        cut = np.max(np.where(below)[0])
        passed[order[:cut + 1]] = True
    return passed.tolist()


def _ritual_covariate(texts, roles):
    """Binary: final user turn OR final system turn contains a ritual
    keyword (case-insensitive substring)."""
    u = [t for t, r in zip(texts, roles) if r == "USER"]
    s = [t for t, r in zip(texts, roles) if r == "SYSTEM"]
    last = " ".join([u[-1] if u else "", s[-1] if s else ""]).lower()
    return any(k in last for k in RITUAL_KEYWORDS)


def _partial_spearman(x, y, cov):
    """Spearman controlling a binary covariate via rank residualization."""
    rx, ry = rankdata(x), rankdata(y)
    c = np.asarray(cov, dtype=float)
    A = np.stack([np.ones(len(c)), c], axis=1)

    def resid(v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    ex, ey = resid(rx), resid(ry)
    denom = np.linalg.norm(ex) * np.linalg.norm(ey)
    return float(ex @ ey / denom) if denom else float("nan")


def run(corpora=("sgd", "mwoz")):
    all_results = {}
    for corpus in corpora:
        convs = load_uss(corpus)
        texts = load_uss_text(corpus)
        outcome = np.array([c["overall_mean"] for c in convs])

        rows = [_functionals(c) for c in convs]
        keep = [i for i, r in enumerate(rows) if r is not None]
        F = {f: np.array([rows[i][f] for i in keep]) for f in FUNCTIONALS}
        yv = outcome[keep]

        stats = {}
        pvals = []
        for f in FUNCTIONALS:
            rho, p = spearmanr(F[f], yv)
            stats[f] = {"rho": float(rho), "p": float(p)}
            pvals.append(p)
        for f, ok in zip(FUNCTIONALS, _bh_fdr(pvals)):
            stats[f]["fdr_pass"] = bool(ok)

        ritual = np.array([_ritual_covariate(texts[i],
                                             convs[i]["roles"].tolist())
                           for i in keep])
        partial_f1 = _partial_spearman(F["F1_raw_final"], yv, ritual)

        strips = {}
        for k in (1, 2):
            srows = [_functionals(convs[i], strip=k) for i in keep]
            skeep = [j for j, r in enumerate(srows) if r is not None]
            sv = np.array([srows[j]["F1_raw_final"] for j in skeep])
            rho, p = spearmanr(sv, yv[skeep])
            strips[f"strip{k}"] = {"rho": float(rho), "p": float(p),
                                   "n": len(skeep)}

        res = {
            "corpus": corpus, "n": len(keep),
            "functionals": stats,
            "ritual_share": float(ritual.mean()),
            "f1_partial_ritual_rho": partial_f1,
            "f1_strip": strips,
        }
        save_result(f"spike2_endpoint_{corpus}", res, dataset=corpus,
                    params={"keywords": RITUAL_KEYWORDS})
        all_results[corpus] = res
    return all_results


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
