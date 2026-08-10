"""Round-3 stage D: the early-warning head as study object. Recomputes the
R2 OOF head persisting per-sample scores, then calibration (reliability +
ECE), the alert-protocol precision/lead-time curve with pre-registered
operating points, and the repaired Bar 1 test against the true-error
baseline. Protocol fixed in PREREGISTRATION-R3.md.

    python tb3_head.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
SEED = 42
N_BOOT = 1000
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]


def _auc_by_tau(y, score, tau):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi)
        out.append(float(roc_auc_score(y[m], score[m]))
                   if m.sum() and len(set(y[m])) == 2 else None)
    return out


def run():
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    p = Checkpoint("tb_features").load("prefix")
    X, tau, sconv, step = p["X"], p["tau"], p["conv"], p["step"]
    sy = y[sconv]

    ckpt = Checkpoint("tb3_head", script=__file__)
    cached = ckpt.load("oof")
    if cached is not None:
        oof = cached["oof"]
    else:
        oof = np.full(len(sy), np.nan)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(X, sy, sconv):
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000)
            clf.fit(X[tr], sy[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        ckpt.save("oof", {"oof": oof})

    # calibration
    bins = np.linspace(0, 1, 11)
    reliability = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (oof >= lo) & (oof < hi if hi < 1 else oof <= 1)
        if m.sum():
            conf, acc = float(oof[m].mean()), float(sy[m].mean())
            reliability.append({"bin": [float(lo), float(hi)],
                                "n": int(m.sum()), "mean_score": conf,
                                "frac_fail": acc})
            ece += m.sum() / len(oof) * abs(conf - acc)

    # alert protocol
    order = np.argsort(sconv, kind="stable")
    thresholds = np.percentile(oof, np.arange(50, 100, 2))
    conv_ids = np.unique(sconv)
    by_conv = {c: np.where(sconv == c)[0] for c in conv_ids}
    n_fail_convs = int(y[conv_ids].sum())
    n_succ_convs = len(conv_ids) - n_fail_convs
    curve = []
    for th in thresholds:
        tp = fa = 0
        leads_steps, leads_tau = [], []
        for c in conv_ids:
            idx = by_conv[c]
            fired = np.where(oof[idx] >= th)[0]
            if not len(fired):
                continue
            first = idx[fired[0]]
            if y[c]:
                tp += 1
                leads_steps.append(float(step[idx[-1]] - step[first]))
                leads_tau.append(float(1.0 - tau[first]))
            else:
                fa += 1
        alerted = tp + fa
        curve.append({
            "threshold": float(th),
            "precision": tp / alerted if alerted else None,
            "recall": tp / n_fail_convs,
            "false_alarm_rate": fa / n_succ_convs,
            "median_lead_steps": float(np.median(leads_steps))
            if leads_steps else None,
            "median_lead_tau": float(np.median(leads_tau))
            if leads_tau else None})

    def operating_point(target):
        cands = [c for c in curve
                 if c["precision"] is not None and c["precision"] >= target]
        return min(cands, key=lambda c: c["threshold"]) if cands else None

    # Bar 1: head vs true-error share, conv-level bootstrap of AUC diff
    share = Checkpoint("tb3_error").load("prefix_share")["share"]
    head_auc = _auc_by_tau(sy, oof, tau)
    err_auc = _auc_by_tau(sy, share, tau)
    rng = np.random.default_rng(SEED)
    diffs = {q: [] for q in range(4)}
    for _ in range(N_BOOT):
        cs = rng.choice(conv_ids, size=len(conv_ids), replace=True)
        idx = np.concatenate([by_conv[c] for c in cs])
        for q, (lo, hi) in enumerate(zip(TAU_BINS[:-1], TAU_BINS[1:])):
            m = (tau[idx] >= lo) & (tau[idx] < hi)
            sub = idx[m]
            if len(set(sy[sub])) == 2:
                diffs[q].append(roc_auc_score(sy[sub], oof[sub])
                                - roc_auc_score(sy[sub], share[sub]))
    bar1 = []
    for q in range(4):
        v = np.array(diffs[q])
        lo, hi = np.percentile(v, [2.5, 97.5])
        bar1.append({"quartile": q + 1,
                     "head_auc": head_auc[q], "err_auc": err_auc[q],
                     "diff": (head_auc[q] - err_auc[q])
                     if head_auc[q] and err_auc[q] else None,
                     "ci": [float(lo), float(hi)],
                     "pass": bool(lo > 0)})

    res = {"overall_auc": float(roc_auc_score(sy, oof)),
           "head_auc_by_tau": head_auc, "ece": float(ece),
           "reliability": reliability,
           "alert_curve": curve,
           "op_p50": operating_point(0.5),
           "op_p70": operating_point(0.7),
           "bar1": bar1,
           "n_samples": int(len(sy)),
           "n_conversations": int(len(conv_ids))}
    save_result("tb3_head", res, dataset="toolbench",
                params={"seed": SEED, "n_boot": N_BOOT})
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("reliability", "alert_curve")}, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
