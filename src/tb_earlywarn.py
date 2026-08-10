"""Round-2 arm 3: zones vs head early warning on ToolBench (protocol shape
of round-1 spike 1, balanced-ish binary labels). Clustering, lift/HR
criteria, lead-time definition, OOF comparison and baselines fixed in
PREREGISTRATION-R2.md. Bar 1: beat the error-keyword share at lead time.
Bar 2: dwell vs head decides whether the zone framing survives.

    python tb_earlywarn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import beta
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.campaign import Checkpoint, save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"

K, SEED, K_SENS, SEED_SENS = 64, 42, 16, 123
N_BOOT = 1000
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]


def _counts(assign, cid, n_conv, k):
    T = np.zeros((n_conv, k))
    np.add.at(T, (cid, assign), 1)
    return T


def _lift(T, y):
    tot, bad = T.sum(0), T[y == 1].sum(0)
    base = bad.sum() / tot.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0, bad / np.maximum(tot, 1), np.nan)
    jlo = beta.ppf(0.025, bad + 0.5, tot - bad + 0.5)
    return share / base, jlo / base, base


def _boot_lift(T, y, rng, k):
    n = len(T)
    lifts = np.full((N_BOOT, k), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        tot, bad = T[idx].sum(0), T[idx][y[idx] == 1].sum(0)
        base = bad.sum() / max(tot.sum(), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            lifts[b] = np.where(tot > 0, bad / np.maximum(tot, 1),
                                np.nan) / base
    return (np.nanpercentile(lifts, 2.5, axis=0),
            np.nanpercentile(lifts, 97.5, axis=0))


def _auc_by_tau(y, score, tau):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi)
        out.append(float(roc_auc_score(y[m], score[m]))
                   if m.sum() and len(set(y[m])) == 2 else None)
    return out


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    n = len(y)
    cid = np.repeat(np.arange(n), np.diff(off))
    rng = np.random.default_rng(SEED)

    ckpt = Checkpoint("tb_earlywarn", script=__file__)
    cached = ckpt.load("assign_primary")
    if cached is not None:
        assign = cached["assign"]
    else:
        assign = KMeans(n_clusters=K, n_init=10,
                        random_state=SEED).fit_predict(emb)
        ckpt.save("assign_primary", {"assign": assign})

    T = _counts(assign, cid, n, K)
    lift, jlo, base = _lift(T, y)
    blo, bhi = _boot_lift(T, y, rng, K)
    hr = np.where((lift >= 2.0) & (blo > 1.0))[0]
    top = int(np.nanargmax(lift))

    finals = assign[off[1:] - 1]
    def tv(labels):
        p = np.bincount(finals[labels == 1], minlength=K) / max((labels == 1).sum(), 1)
        q = np.bincount(finals[labels == 0], minlength=K) / max((labels == 0).sum(), 1)
        return 0.5 * np.abs(p - q).sum()
    obs_tv = tv(y)
    perms = np.array([tv(rng.permutation(y)) for _ in range(N_BOOT)])
    geo = {"tv": float(obs_tv), "perm_p95": float(np.percentile(perms, 95)),
           "perm_pvalue": float((perms >= obs_tv).mean())}

    hr_set = set(hr.tolist())
    lead_rows = []
    for i in np.where(y == 1)[0]:
        a = assign[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        in_hr = np.where(np.isin(a, list(hr_set)))[0]
        if not len(in_hr):
            continue
        first = in_hr[0]
        tr = np.where(tt == "tool_result")[0]
        cutoff = tr[-2] if len(tr) >= 2 else (tr[-1] if len(tr) else 0)
        m = len(a)
        lead_rows.append({"tau": first / max(m - 1, 1),
                          "early": bool(first < cutoff)})
    lead = ({"n_entering": len(lead_rows),
             "median_first_tau": float(np.median([r["tau"] for r in lead_rows])),
             "early_share": float(np.mean([r["early"] for r in lead_rows]))}
            if lead_rows else {"n_entering": 0, "median_first_tau": None,
                               "early_share": None})

    p = Checkpoint("tb_features").load("prefix")
    X, tau, sconv = p["X"], p["tau"], p["conv"]
    sy = y[sconv]
    head_oof = np.full(len(sy), np.nan)
    dwell = np.full(len(sy), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr_idx, te_idx in gkf.split(X, sy, sconv):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000)
        clf.fit(X[tr_idx], sy[tr_idx])
        head_oof[te_idx] = clf.predict_proba(X[te_idx])[:, 1]

        tr_convs = np.unique(sconv[tr_idx])
        mask = np.isin(cid, tr_convs)
        Tf = _counts(assign[mask], cid[mask], n, K)
        lf, jf, _ = _lift(Tf[tr_convs], y[tr_convs])
        hr_f = set(np.where((lf >= 2.0) & (jf > 1.0))[0].tolist())
        for c in np.unique(sconv[te_idx]):
            a = assign[off[c]:off[c + 1]]
            cum = np.cumsum(np.isin(a, list(hr_f)))
            for s in te_idx[sconv[te_idx] == c]:
                t = int(round(tau[s] * (len(a) - 1)))
                dwell[s] = cum[t] / (t + 1)

    aucs = {"head": _auc_by_tau(sy, head_oof, tau),
            "dwell": _auc_by_tau(sy, dwell, tau),
            "err_share": _auc_by_tau(sy, p["err_share"], tau),
            "goal_dist": _auc_by_tau(sy, p["goal_dist"], tau),
            "step_count": _auc_by_tau(sy, p["step"], tau)}

    sens = {}
    for tag, kk, sd in (("k16", K_SENS, SEED), ("seed123", K, SEED_SENS)):
        a2 = KMeans(n_clusters=kk, n_init=10, random_state=sd).fit_predict(emb)
        T2 = _counts(a2, cid, n, kk)
        l2, _, _ = _lift(T2, y)
        b2lo, _ = _boot_lift(T2, y, rng, kk)
        sens[tag] = {"max_lift": float(np.nanmax(l2)),
                     "n_highrisk": int(((l2 >= 2.0) & (b2lo > 1.0)).sum())}

    res = {"n": int(n), "failure_rate": float(y.mean()), "k": K,
           "base_rate": float(base), "max_lift": float(lift[top]),
           "max_lift_boot_ci": [float(blo[top]), float(bhi[top])],
           "n_highrisk": int(len(hr)), "endpoint_geography": geo,
           "lead_time": lead, "auc_by_tau": aucs,
           "n_samples": int(len(sy)), "sensitivities": sens}
    save_result("tb_earlywarn", res, dataset="toolbench",
                params={"k": K, "seed": SEED, "n_boot": N_BOOT})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
