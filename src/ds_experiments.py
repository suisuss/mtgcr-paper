"""Round-4 Arm B: generic replication battery, protocols identical to
rounds 2-3, parameterized per corpus (PREREGISTRATION-R4.md).

Components: early-warning head (prefix per-type running means, logistic,
GroupKFold by group_id, OOF AUC per tau quartile); baselines (step count,
running goal distance, and for SWE the fixed error-pattern share);
endpoint functionals E1/E3 with BH-FDR and (SWE) the error-content
stratification; typed ladder with canonical Gram slopes (SWE only);
thought-presence transfer check (SWE).

    python ds_experiments.py abcd
    python ds_experiments.py swe
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.campaign import save_result
from lib.gram import gram_fit

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
SEED = 42
N_BOOT = 1000
KEEP_DELTA = 0.01
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]

CONFIG = {
    "abcd": {"types": ["customer", "agent", "action"],
             "decision": "customer", "agentlike": "agent",
             "final_e3": "action", "error_patterns": None, "ladder": False},
    "swe": {"types": ["user", "assistant", "tool_result"],
            "decision": "tool_result", "agentlike": "assistant",
            "final_e3": "tool_result",
            "error_patterns": re.compile(
                r"Traceback|error|FAILED|Exception|fatal", re.I),
            "ladder": True},
    "spokenwoz": {"types": ["user", "system"], "decision": "user",
                  "agentlike": "system", "final_e3": "system",
                  "error_patterns": None, "ladder": False},
}


def _auc_by_tau(y, score, tau):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi) & ~np.isnan(score)
        out.append(float(roc_auc_score(y[m], score[m]))
                   if m.sum() and len(set(y[m])) == 2 else None)
    return out


def _oof(X, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run(dataset):
    cfg = CONFIG[dataset]
    z = np.load(EMB / f"{dataset}_minilm.npz")
    emb, off, goals = z["embeddings"], z["offsets"], z["goal_embeddings"]
    meta = json.loads((EMB / f"{dataset}_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    gid = np.array(meta["group_id"])
    n = len(y)
    types = cfg["types"]
    k_types = len(types)

    # ---- prefix samples + per-conversation features in one pass
    px, ptau, pconv, pgoal, pstep, perr = [], [], [], [], [], []
    overall_mean = np.zeros((n, 384), np.float32)
    type_means = np.zeros((n, k_types, 384), np.float32)
    type_present = np.zeros((n, k_types), np.float32)
    type_slopes = np.zeros((n, k_types, 384), np.float32)
    recurrence = np.zeros((n, 3), np.float32)
    goal_dist_types = np.zeros((n, k_types), np.float32)
    e1 = np.full(n, np.nan)
    e3 = np.full(n, np.nan)
    final_err = np.zeros(n, dtype=int)
    asst_share = np.zeros(n, np.float32)

    # need raw text only for the SWE error baseline
    raw_text = None
    has_err = False
    if cfg["error_patterns"] is not None:
        src = {"swe": REPO / "data" / "raw" / "swe_trajectories"
               / "normalized.json"}[dataset]
        raw = [c for c in json.loads(src.read_text())
               if c["label"] is not None]
        raw_text = [[t["c"] for t in c["turns"]] for c in raw]
        has_err = True

    for i in range(n):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        g = goals[i]
        m = len(E)
        d = 1.0 - E @ g
        overall_mean[i] = E.mean(0)
        e1[i] = d[-1]
        fe = np.where(tt == cfg["final_e3"])[0]
        if len(fe):
            e3[i] = d[fe[-1]]
            if raw_text is not None:
                final_err[i] = int(bool(cfg["error_patterns"].search(
                    raw_text[i][fe[-1]])))
        for k, ty in enumerate(types):
            idx = np.where(tt == ty)[0]
            if len(idx):
                type_means[i, k] = E[idx].mean(0)
                type_present[i, k] = 1.0
                goal_dist_types[i, k] = float(d[idx].mean())
            if len(idx) >= 2:
                type_slopes[i, k] = gram_fit(
                    E[idx].astype(np.float64), degree=1)[1]
        a_idx = np.where(tt == cfg["agentlike"])[0]
        asst_share[i] = len(a_idx) / m
        if len(a_idx) >= 2:
            sims = (E[a_idx] @ E[a_idx].T)[
                np.triu_indices(len(a_idx), k=1)]
            recurrence[i] = [sims.mean(), sims.max(), len(a_idx)]
        else:
            recurrence[i] = [0.0, 0.0, len(a_idx)]

        onehot = np.stack([(tt == ty).astype(np.float32) for ty in types],
                          axis=1)
        csum = np.cumsum(E[:, None, :] * onehot[:, :, None], axis=0)
        ccnt = np.cumsum(onehot, axis=0)
        if raw_text is not None:
            err_flags = np.array(
                [bool(cfg["error_patterns"].search(txt))
                 if ty == "tool_result" else False
                 for txt, ty in zip(raw_text[i], tt)], dtype=np.float32)
            err_csum = np.cumsum(err_flags)
            tr_ccnt = ccnt[:, types.index("tool_result")]
        dec = np.where(tt == cfg["decision"])[0]
        agent_k = types.index(cfg["agentlike"])
        for step_no, t in enumerate(dec):
            means_t = csum[t] / np.maximum(ccnt[t][:, None], 1.0)
            px.append(means_t.reshape(-1))
            ptau.append(t / max(m - 1, 1))
            pconv.append(i)
            pstep.append(step_no + 1)
            am = means_t[agent_k]
            nrm = np.linalg.norm(am)
            pgoal.append(1.0 - float(am @ g / nrm) if nrm > 1e-9
                         else np.nan)
            perr.append(err_csum[t] / max(tr_ccnt[t], 1.0)
                        if raw_text is not None else np.nan)

    px = np.array(px, np.float32)
    ptau, pconv = np.array(ptau), np.array(pconv)
    pgoal, pstep = np.array(pgoal), np.array(pstep, np.float32)
    perr = np.array(perr)
    sy, sgroups = y[pconv], gid[pconv]

    head_oof = _oof(px, sy, sgroups)
    head = {"auc_by_tau": _auc_by_tau(sy, head_oof, ptau),
            "overall_auc": float(roc_auc_score(sy, head_oof)),
            "n_samples": int(len(sy))}
    baselines = {"goal_dist": _auc_by_tau(sy, pgoal, ptau),
                 "step_count": _auc_by_tau(sy, pstep, ptau)}
    if raw_text is not None:
        baselines["err_share"] = _auc_by_tau(sy, perr, ptau)
    # the prefix matrix and raw text are the stage's biggest tenants;
    # release them before the ladder allocates its rung matrices
    del px, head_oof, raw_text
    import gc
    gc.collect()

    # ---- endpoints
    def mwu(vals):
        mask = ~np.isnan(vals)
        auc = float(roc_auc_score(y[mask], vals[mask]))
        p = float(mannwhitneyu(vals[mask][y[mask] == 1],
                               vals[mask][y[mask] == 0],
                               alternative="two-sided").pvalue)
        return {"auc": auc, "p": p, "n": int(mask.sum())}

    endpoints = {"E1_final_turn": mwu(e1), "E3_final_typed": mwu(e3)}
    ps = [endpoints[k]["p"] for k in endpoints]
    bh = np.argsort(ps)
    passed = [False, False]
    for rank, idx in enumerate(bh):
        if ps[idx] <= 0.05 * (rank + 1) / 2:
            passed[idx] = True
    for k, ok in zip(endpoints, passed):
        endpoints[k]["fdr_pass"] = bool(ok)
    if has_err:
        for k, vals in (("E1_final_turn", e1), ("E3_final_typed", e3)):
            for strat, mask in (("no_error_final", final_err == 0),
                                ("error_final", final_err == 1)):
                mm = mask & ~np.isnan(vals)
                endpoints[k][strat] = (
                    {"auc": float(roc_auc_score(y[mm], vals[mm])),
                     "n": int(mm.sum())}
                    if len(set(y[mm])) == 2 else {"auc": None,
                                                 "n": int(mm.sum())})

    # ---- ladder (SWE only)
    ladder = None
    if cfg["ladder"]:
        flat_means = type_means.reshape(n, -1)
        flat_slopes = type_slopes.reshape(n, -1)
        n_turns = np.array(meta["n_turns"], np.float32)[:, None]
        blocks = {
            "R1_content_length": [overall_mean, n_turns],
            "R2_typed_content": [overall_mean, n_turns, flat_means,
                                 type_present],
            "R3_typed_structure": [overall_mean, n_turns, flat_means,
                                   type_present, flat_slopes],
            "R4_dynamics_goal": [overall_mean, n_turns, flat_means,
                                 type_present, flat_slopes, recurrence,
                                 goal_dist_types],
        }
        oof = {}
        for name, parts in blocks.items():
            X = np.hstack(parts).astype(np.float32)
            o = np.full(n, np.nan)
            for tr, te in GroupKFold(n_splits=5).split(X, y, gid):
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000))
                clf.fit(X[tr], y[tr])
                o[te] = clf.predict_proba(X[te])[:, 1]
            oof[name] = o
            del X
        rungs = blocks  # names only, for the delta loop below
        auc = {nm: float(roc_auc_score(y, o)) for nm, o in oof.items()}
        rng = np.random.default_rng(SEED)
        ug = np.unique(gid)
        by_g = {g_: np.where(gid == g_)[0] for g_ in ug}
        names = list(rungs)
        deltas = {f"{b}-{a}": [] for a, b in zip(names, names[1:])}
        for _ in range(N_BOOT):
            gs = rng.choice(ug, size=len(ug), replace=True)
            idx = np.concatenate([by_g[g_] for g_ in gs])
            if len(set(y[idx])) < 2:
                continue
            rs = {nm: roc_auc_score(y[idx], oof[nm][idx]) for nm in names}
            for a, b in zip(names, names[1:]):
                deltas[f"{b}-{a}"].append(rs[b] - rs[a])
        ladder = {"oof_auc": auc, "rungs": {}}
        for key, v in deltas.items():
            v = np.array(v)
            lo, hi = np.percentile(v, [2.5, 97.5])
            hi_n, lo_n = key.split("-")
            point = auc[hi_n] - auc[lo_n]
            ladder["rungs"][key] = {
                "delta": float(point), "ci": [float(lo), float(hi)],
                "keep": bool(point >= KEEP_DELTA and lo > 0)}

    res = {"dataset": dataset, "n": int(n),
           "failure_rate": float(y.mean()),
           "head": head, "baselines": baselines,
           "endpoints": endpoints, "ladder": ladder,
           "assistant_share_auc": float(roc_auc_score(y, asst_share))}
    save_result(f"ds_experiments_{dataset}", res, dataset=dataset,
                params={"seed": SEED, "n_boot": N_BOOT})
    print(json.dumps(res, indent=1)[:2500])
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
