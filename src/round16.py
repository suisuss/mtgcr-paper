"""Round 16: degree-1 role-split, and does velocity predict satisfaction?
PREREGISTRATION-R16.md (+ 2026-07-24 amendment).

Degree-0 (the per-role mean) is the whole paper. Degree-1 keeps the mean AND the
per-role slope - the direction each channel moves through semantic space. The
hypothesis: level geometry predicts whether a call resolves (outcome); velocity
geometry predicts how it is going (satisfaction).

  A  ToolBench outcome (4-role fixed): degree-0 (drop_assistant) vs + per-role
     degree-1 slope, with a leak-free slope-shuffle control. Expected null (R15).
  B  USS satisfaction (2-role): per-turn Spearman, degree-0 vs degree-1, with a
     same-turn-content baseline (last USER-turn embedding; VADER unavailable) and
     a lead-1 requirement. The asymmetry test.
  C  USS dialogue-act channel (running success/failure + act distribution) as a
     weak tool-outcome proxy for satisfaction.

Degree-1 on the fixed roles stays un-gameable; robustness inherited from round 13.

    python round16.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.actr_data import actr_root
from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
DIM = 384
SEED = 42
BOOT = 1000
AUC_MARGIN = 0.02        # Direction A (outcome)
RHO_MARGIN = 0.03        # Direction B (satisfaction)
TB_FIXED = ["user", "tool_call", "tool_result"]


def _fit01(seq, shuffle_rng=None):
    """Degree-0 (mean) and degree-1 (slope) role coefficients of a prefix
    seq (k, DIM). Positions are linspace(-1, 1, k), symmetric, so the intercept
    is the mean and the slope is sum(x*e)/sum(x^2). Slope 0 for k < 2 (frozen
    short-role fallback). shuffle_rng permutes the prefix rows (destroys the
    temporal direction; still causal - only the seen prefix is used)."""
    k = len(seq)
    a0 = seq.mean(0)
    if k < 2:
        return a0, np.zeros(DIM)
    E = seq if shuffle_rng is None else seq[shuffle_rng.permutation(k)]
    x = np.linspace(-1.0, 1.0, k)
    a1 = (x[:, None] * E).sum(0) / (x * x).sum()
    return a0, a1


def _boot_ci(metric, pred, y, conv, boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uc = np.unique(conv)
    by = {c: np.where(conv == c)[0] for c in uc}
    vals = []
    for _ in range(boot):
        pts = np.concatenate([by[c] for c in rng.choice(uc, len(uc), True)])
        ys = y[pts]
        if np.ptp(ys) == 0:
            continue
        vals.append(metric(ys, pred[pts]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


# ---------------------------------------------------------------- Direction A
def _tb_collect(emb, off, meta):
    cols = {f"{r}_{c}": [] for r in TB_FIXED for c in ("a0", "a1", "a1s")}
    conv = []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        buf = {r: [] for r in TB_FIXED}
        rng = np.random.default_rng([SEED, i])
        for t in range(len(E)):
            r = tt[t]
            if r in buf:
                buf[r].append(E[t])
            if r == "tool_result":
                for rr in TB_FIXED:
                    seq = np.array(buf[rr]) if buf[rr] else np.zeros((1, DIM))
                    a0, a1 = _fit01(seq)
                    _, a1s = _fit01(seq, shuffle_rng=rng)
                    cols[f"{rr}_a0"].append(a0)
                    cols[f"{rr}_a1"].append(a1)
                    cols[f"{rr}_a1s"].append(a1s)
                conv.append(i)
    out = {k: np.array(v, dtype=np.float32) for k, v in cols.items()}
    out["conv"] = np.array(conv)
    return out


def _direction_a():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    d = _tb_collect(emb, off, meta)
    conv = d["conv"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(d["user_a0"], sy, conv))
    cte = conv[te]
    base = np.hstack([d[f"{r}_a0"] for r in TB_FIXED])

    def arm(X):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[tr], sy[tr])
        s = clf.decision_function(X[te])
        return round(float(roc_auc_score(sy[te], s)), 4), \
            _boot_ci(roc_auc_score, s, sy[te], cte)

    base_auc, base_ci = arm(base)
    slope = np.hstack([base] + [d[f"{r}_a1"] for r in TB_FIXED])
    shuf = np.hstack([base] + [d[f"{r}_a1s"] for r in TB_FIXED])
    sl_auc, sl_ci = arm(slope)
    sh_auc, sh_ci = arm(shuf)
    keep = bool(sl_auc - base_auc >= AUC_MARGIN and not _overlap(sl_ci, base_ci)
                and sl_auc - sh_auc >= AUC_MARGIN)
    return {
        "target": "toolbench_outcome", "metric": "AUC",
        "base_degree0": {"auc": base_auc, "ci95": base_ci},
        "+degree1_slope": {"auc": sl_auc, "ci95": sl_ci,
                           "delta_vs_base": round(sl_auc - base_auc, 4)},
        "+slope_shuffled": {"auc": sh_auc, "ci95": sh_ci,
                            "delta_vs_base": round(sh_auc - base_auc, 4)},
        "keep_degree1": keep,
        "verdict": ("KEEP: velocity adds outcome signal" if keep else
                    "KILL: degree-0 mean-pooling is sufficient for outcome"),
    }


# ------------------------------------------------------------- Directions B, C
def _uss_collect(corpus, lead):
    root = actr_root()
    z = np.load(root / "data" / "embeddings" / f"uss_{corpus}_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((root / "data" / "embeddings"
                       / f"uss_{corpus}_meta.json").read_text())
    roles, ratings, actions = (meta["roles"], meta["turn_ratings"],
                               meta["actions"])
    vocab = sorted({a for conv in actions for a in conv})
    aidx = {a: k for k, a in enumerate(vocab)}
    cols = {k: [] for k in ("u_a0", "u_a1", "u_a1s", "s_a0", "s_a1", "s_a1s",
                            "last_user", "acts")}
    y, conv = [], []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        rl, rt, ac = roles[i], ratings[i], actions[i]
        rng = np.random.default_rng([SEED, i])
        rated = [t for t in range(len(E))
                 if rl[t] == "USER" and t < len(rt) and rt[t]]
        for pos, t in enumerate(rated):
            if lead:
                if pos + 1 >= len(rated):
                    continue
                target_t = rated[pos + 1]      # predict the NEXT rating...
            else:
                target_t = t                    # ...or the current one
            uu = E[[j for j in range(t + 1) if rl[j] == "USER"]]
            ss = E[[j for j in range(t + 1) if rl[j] == "SYSTEM"]] \
                if any(rl[j] == "SYSTEM" for j in range(t + 1)) \
                else np.zeros((1, DIM))
            ua0, ua1 = _fit01(uu)
            _, ua1s = _fit01(uu, shuffle_rng=rng)
            sa0, sa1 = _fit01(ss)
            _, sa1s = _fit01(ss, shuffle_rng=rng)
            adist = np.zeros(len(vocab))
            for j in range(t + 1):
                if j < len(ac) and ac[j] in aidx:
                    adist[aidx[ac[j]]] += 1
            adist /= max(adist.sum(), 1)
            cols["u_a0"].append(ua0)
            cols["u_a1"].append(ua1)
            cols["u_a1s"].append(ua1s)
            cols["s_a0"].append(sa0)
            cols["s_a1"].append(sa1)
            cols["s_a1s"].append(sa1s)
            cols["last_user"].append(E[t])
            cols["acts"].append(adist)
            y.append(float(np.mean(rt[target_t])))
            conv.append(i)
    out = {k: np.array(v, dtype=np.float32) for k, v in cols.items()}
    out["y"] = np.array(y)
    out["conv"] = np.array(conv)
    return out


def _uss_eval(corpus, lead=False):
    d = _uss_collect(corpus, lead)
    conv, y = d["conv"], d["y"]
    tr, te = next(GroupKFold(n_splits=5).split(d["u_a0"], y, conv))
    cte = conv[te]

    def arm(X):
        clf = Ridge(alpha=1.0).fit(X[tr], y[tr])
        p = clf.predict(X[te])
        return round(float(spearmanr(y[te], p).statistic), 4), \
            _boot_ci(lambda a, b: spearmanr(a, b).statistic, p, y[te], cte)

    base = np.hstack([d["u_a0"], d["s_a0"]])
    slope = np.hstack([base, d["u_a1"], d["s_a1"]])
    shuf = np.hstack([base, d["u_a1s"], d["s_a1s"]])
    b_rho, b_ci = arm(base)
    sl_rho, sl_ci = arm(slope)
    sh_rho, sh_ci = arm(shuf)
    sent_rho, sent_ci = arm(d["last_user"])
    acts_rho, acts_ci = arm(np.hstack([base, d["acts"]]))
    keep = bool(sl_rho - b_rho >= RHO_MARGIN and not _overlap(sl_ci, b_ci)
                and sl_rho - sh_rho >= RHO_MARGIN and sl_rho - sent_rho >= 0)
    return {
        "n_points_test": int(len(te)),
        "base_degree0": {"rho": b_rho, "ci95": b_ci},
        "+degree1_slope": {"rho": sl_rho, "ci95": sl_ci,
                           "delta_vs_base": round(sl_rho - b_rho, 4)},
        "+slope_shuffled": {"rho": sh_rho, "ci95": sh_ci},
        "sentiment_baseline": {"rho": sent_rho, "ci95": sent_ci},
        "+acts": {"rho": acts_rho, "ci95": acts_ci,
                  "delta_vs_base": round(acts_rho - b_rho, 4)},
        "keep_degree1": keep,
    }


def run():
    direction_a = _direction_a()
    direction_b = {}
    for corpus in ("sgd", "mwoz"):
        direction_b[corpus] = {
            "lead0": _uss_eval(corpus, lead=False),
            "lead1": _uss_eval(corpus, lead=True),
        }
    # asymmetry: degree-1 delta on satisfaction (best USS lead0) vs outcome
    sat_deltas = [direction_b[c]["lead0"]["+degree1_slope"]["delta_vs_base"]
                  for c in ("sgd", "mwoz")]
    out_delta = direction_a["+degree1_slope"]["delta_vs_base"]
    res = {
        "seed": SEED,
        "direction_a_outcome": direction_a,
        "direction_bc_satisfaction": direction_b,
        "asymmetry": {
            "outcome_degree1_delta_auc": out_delta,
            "satisfaction_degree1_delta_rho": {"sgd": sat_deltas[0],
                                               "mwoz": sat_deltas[1]},
            "note": ("velocity helps satisfaction more than outcome iff the "
                     "rho deltas exceed the auc delta and clear the bar"),
        },
    }
    save_result("round16", res, dataset="mixed", params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
