"""Round-7: shape-target reward manipulability, judged by a
position-neutral per-turn HUMAN judge (USS turn_ratings). Completes the
predecessor design's shape-target arm its arc-biased LLM judge could not.
PREREGISTRATION-R7.md.

Satisfaction series = mean-annotator rating per user turn; targets y0/y1/y2
are its degree-0/1/2 Gram coefficients. Reward heads predict y0/y1/y2 from
the SYSTEM channel Gram coefficients (system channel only). Best-of-N
selection over OOF reward scores; judges = equal-weight per-turn human
mean (position-neutral) and last-turn human rating (peak-end proxy).

    python shape_reward.py sgd
    python shape_reward.py mwoz
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result
from lib.actr_data import load_uss, load_uss_meta
from lib.gram import gram_fit

SEED = 42
NS = [1, 2, 4, 8, 16, 32]
T_TRIALS = 2000
ALPHAS = (0.1, 1.0, 10.0)


def _series(turn_ratings):
    return np.array([float(np.mean(r)) for r in turn_ratings if r],
                    dtype=np.float64)


def _oof(X, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RidgeCV(alphas=ALPHAS)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def run(corpus):
    convs = load_uss(corpus)
    meta = load_uss_meta(corpus)
    turn_ratings = meta["turn_ratings"]

    g0s, g1s, g2s = [], [], []
    u1s, u2s = [], []
    y0, y1, y2, jfinal = [], [], [], []
    for i, c in enumerate(convs):
        S = c["emb"][c["roles"] == "SYSTEM"].astype(np.float64)
        U = c["emb"][c["roles"] == "USER"].astype(np.float64)
        series = _series(turn_ratings[i])
        if len(S) < 3 or len(U) < 3 or len(series) < 3:
            continue
        gc = gram_fit(S, degree=2)
        uc = gram_fit(U, degree=2)
        sc = gram_fit(series[:, None], degree=2)
        g0s.append(gc[0]); g1s.append(gc[1]); g2s.append(gc[2])
        u1s.append(uc[1]); u2s.append(uc[2])
        y0.append(sc[0, 0]); y1.append(sc[1, 0]); y2.append(sc[2, 0])
        jfinal.append(series[-1])
    g0s, g1s, g2s = np.array(g0s), np.array(g1s), np.array(g2s)
    u1s, u2s = np.array(u1s), np.array(u2s)
    y0, y1, y2 = np.array(y0), np.array(y1), np.array(y2)
    jmean, jfinal = y0.copy(), np.array(jfinal)
    n = len(y0)
    groups = np.arange(n)

    heads = {"level": _oof(g0s, y0, groups),
             "trend": _oof(g1s, y1, groups),
             "recovery": _oof(g2s, y2, groups)}
    targets = {"level": y0, "trend": y1, "recovery": y2}
    accuracy = {k: float(spearmanr(heads[k], targets[k]).statistic)
                for k in heads}

    # Safety-vs-strength diagnostic: the mandated SYSTEM-only channel
    # vs the full both-role representation (Paper 3's setup). Shows how much
    # shape signal the manipulation-safe restriction discards.
    def acc(X, y):
        return float(spearmanr(_oof(X, y, groups), y).statistic)
    channel = {
        "trend_system_only": accuracy["trend"],
        "trend_user_only": acc(u1s, y1),
        "trend_both_roles": acc(np.hstack([g1s, u1s]), y1),
        "recovery_system_only": accuracy["recovery"],
        "recovery_both_roles": acc(np.hstack([g2s, u2s]), y2)}

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    rewards = {"level": heads["level"], "trend": heads["trend"],
               "recovery": heads["recovery"],
               "combo": z(heads["level"]) + z(heads["trend"])
               + z(heads["recovery"])}

    rng = np.random.default_rng(SEED)
    draws = {N: rng.integers(0, n, size=(T_TRIALS, N)) for N in NS}
    probe = {}
    for name, r in rewards.items():
        by_n = {}
        for N in NS:
            idx = draws[N]
            pick = idx[np.arange(T_TRIALS), np.argmax(r[idx], axis=1)]
            by_n[N] = {"judge_mean": float(jmean[pick].mean()),
                       "judge_final": float(jfinal[pick].mean()),
                       "reward_own": float(r[pick].mean())}
        base = by_n[1]
        probe[name] = {
            "by_n": {str(N): by_n[N] for N in NS},
            "delta_judge_mean": by_n[32]["judge_mean"] - base["judge_mean"],
            "delta_judge_final": by_n[32]["judge_final"]
            - base["judge_final"],
            "delta_reward_own": by_n[32]["reward_own"] - base["reward_own"]}

    res = {"corpus": corpus, "n": int(n),
           "head_accuracy_oof_spearman": accuracy,
           "channel_strength": channel,
           "judge_mean_grand": float(jmean.mean()),
           "judge_final_grand": float(jfinal.mean()),
           "probe": probe}
    save_result(f"shape_reward_{corpus}", res, dataset=corpus,
                params={"seed": SEED, "ns": NS, "trials": T_TRIALS})
    print(json.dumps({"corpus": corpus, "n": n, "accuracy": accuracy,
                      "deltas": {k: {"judge_mean": round(v["delta_judge_mean"], 3),
                                     "judge_final": round(v["delta_judge_final"], 3),
                                     "reward_own": round(v["delta_reward_own"], 3)}
                                 for k, v in probe.items()}}, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
