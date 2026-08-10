"""Spike 3b: the pre-registered ablation ladder over PRISM within-response
sentence geometry. Rungs (PREREGISTRATION.md): R1 content+length (mean
embedding + n_sentences), R3 + degree-1 drift vector, R4 + causal surprise
profile [mean, max, last]. Ridge (alpha in {0.1, 1, 10} selected on train
folds), GroupKFold by conversation, OOF Spearman; rung deltas judged by
conversation-level paired bootstrap (1000 draws, seed 42), keep = +0.02
with the 95% CI excluding zero.

Features are standardized per train fold (one shared scaler per rung);
this applies identically to every rung so ladder deltas are unaffected.

    python spike3b_subturn_ladder.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.campaign import Checkpoint, save_result

SEED = 42
N_BOOT = 1000
ALPHAS = (0.1, 1.0, 10.0)
KEEP_DELTA = 0.02


def _features(emb, offsets):
    """Per-response mean, drift slope vector, surprise profile."""
    n = len(offsets) - 1
    mean = np.zeros((n, emb.shape[1]), dtype=np.float32)
    drift = np.zeros((n, emb.shape[1]), dtype=np.float32)
    surp = np.zeros((n, 3), dtype=np.float32)
    n_sent = np.zeros(n, dtype=np.float32)
    for r in range(n):
        E = emb[offsets[r]:offsets[r + 1]]
        m = len(E)
        n_sent[r] = m
        mean[r] = E.mean(0)
        t = np.arange(m, dtype=np.float32)
        A = np.stack([np.ones(m, dtype=np.float32), t], axis=1)
        coef, *_ = np.linalg.lstsq(A, E, rcond=None)
        drift[r] = coef[1]
        s = []
        for i in range(2, m):
            Ai = A[:i]
            ci, *_ = np.linalg.lstsq(Ai, E[:i], rcond=None)
            pred = ci[0] + ci[1] * i
            denom = np.linalg.norm(pred) * np.linalg.norm(E[i])
            s.append(1.0 - float(pred @ E[i] / denom) if denom else 1.0)
        s = np.array(s)
        surp[r] = [s.mean(), s.max(), s[-1]]
    return mean, drift, surp, n_sent


def _oof_predict(X, y, groups):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        model = make_pipeline(StandardScaler(),
                              RidgeCV(alphas=ALPHAS))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return oof


def run():
    ckpt = Checkpoint("spike3a_prism_encode")
    d = ckpt.load("sentences")
    emb, offsets = d["emb"], d["offsets"]
    y, groups = d["scores"], d["convs"]

    mean, drift, surp, n_sent = _features(emb, offsets)
    rungs = {
        "R1_content_length": np.hstack([mean, n_sent[:, None]]),
        "R3_plus_drift": np.hstack([mean, n_sent[:, None], drift]),
        "R4_plus_surprise": np.hstack([mean, n_sent[:, None], drift, surp]),
    }

    oof = {name: _oof_predict(X, y, groups) for name, X in rungs.items()}
    rho = {name: float(spearmanr(o, y).statistic) for name, o in oof.items()}

    rng = np.random.default_rng(SEED)
    conv_ids = np.unique(groups)
    by_conv = {c: np.where(groups == c)[0] for c in conv_ids}
    names = list(rungs)
    deltas = {f"{b}-{a}": [] for a, b in zip(names, names[1:])}
    for _ in range(N_BOOT):
        sample = rng.choice(conv_ids, size=len(conv_ids), replace=True)
        idx = np.concatenate([by_conv[c] for c in sample])
        rs = {n: spearmanr(oof[n][idx], y[idx]).statistic for n in names}
        for a, b in zip(names, names[1:]):
            deltas[f"{b}-{a}"].append(rs[b] - rs[a])

    ladder = {}
    for key, vals in deltas.items():
        v = np.array(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        point = rho[key.split("-")[0]] - rho[key.split("-")[1]]
        ladder[key] = {"delta": float(point), "ci": [float(lo), float(hi)],
                       "keep": bool(point >= KEEP_DELTA and lo > 0)}

    res = {"n_responses": int(len(y)),
           "n_conversations": int(len(conv_ids)),
           "oof_spearman": rho, "ladder": ladder,
           "median_n_sentences": float(np.median(n_sent))}
    save_result("spike3_subturn_prism", res, dataset="prism",
                params={"alphas": list(ALPHAS), "seed": SEED,
                        "n_boot": N_BOOT, "keep_delta": KEEP_DELTA})
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
