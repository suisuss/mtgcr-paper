"""Round-3 stage E: canonical Gram parameterization port + equivalence
check against the lstsq stand-in used by rounds 1-3. Pass criterion (max
|delta| < 1e-4) fixed in PREREGISTRATION-R3.md.

    python gram_equivalence.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result
from lib.gram import gram_fit, gram_curve
from lib.actr_data import load_uss

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
TAU_GRID = np.linspace(0.0, 1.0, 101)
TYPES = ["user", "assistant", "tool_call", "tool_result"]


def _lstsq_curve(E, grid):
    m = len(E)
    t = np.arange(m, dtype=np.float64) / max(m - 1, 1)
    A = np.stack([np.ones(m), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, E.astype(np.float64), rcond=None)
    return coef[0][None, :] + grid[:, None] * coef[1][None, :]


def run(max_conversations=2000):
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    n = min(len(off) - 1, max_conversations)

    max_g0 = 0.0
    max_curve = 0.0
    checked = 0
    for i in range(n):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = np.array(meta["turn_types"][i])
        for ty in TYPES:
            idx = np.where(tt == ty)[0]
            if len(idx) < 2:
                continue
            S = E[idx]
            coeffs = gram_fit(S, degree=1)
            max_g0 = max(max_g0, float(np.abs(coeffs[0] - S.mean(0)).max()))
            gcurve = gram_curve(coeffs[:2], len(S), TAU_GRID)
            lcurve = _lstsq_curve(S, TAU_GRID)
            max_curve = max(max_curve, float(np.abs(gcurve - lcurve).max()))
            checked += 1

    # USS-SGD role gap curves (spike 2's object)
    max_gap = 0.0
    for conv in load_uss("sgd")[:500]:
        E, roles = conv["emb"].astype(np.float64), conv["roles"]
        curves = {}
        for role in ("USER", "SYSTEM"):
            idx = np.where(roles == role)[0]
            if len(idx) < 2:
                curves = None
                break
            S = E[idx]
            curves[role] = (gram_curve(gram_fit(S, degree=1)[:2],
                                       len(S), TAU_GRID),
                            _lstsq_curve(S, TAU_GRID))
        if curves is None:
            continue
        ggap = curves["USER"][0] - curves["SYSTEM"][0]
        lgap = curves["USER"][1] - curves["SYSTEM"][1]
        max_gap = max(max_gap, float(np.abs(
            np.linalg.norm(ggap, axis=1) - np.linalg.norm(lgap, axis=1)
        ).max()))

    passed = max(max_g0, max_curve, max_gap) < 1e-4
    res = {"sequences_checked": int(checked),
           "max_abs_g0_vs_mean": max_g0,
           "max_abs_curve_delta": max_curve,
           "max_abs_uss_gapnorm_delta": max_gap,
           "pass": bool(passed)}
    save_result("gram_equivalence", res, dataset="toolbench+sgd",
                params={"tolerance": 1e-4,
                        "max_conversations": max_conversations})
    print(json.dumps(res, indent=1))
    if not passed:
        raise SystemExit("gram_equivalence FAILED tolerance")
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
