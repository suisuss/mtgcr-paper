"""Round 15: structure beyond the mean - order/recency, goal-conditioning, and
cross-channel consistency, all on the fixed (un-gameable) channels.
PREREGISTRATION-R15.md.

Round 14 showed the per-role mean has nearly exhausted the fixed channels (the
payload embedding beat parsed flags by only +0.012). Round 15 adds the structure
the mean discards - all functions of the fixed channels only (user, tool-call
skeleton, tool-result, goal), never the narration, so every arm is invariant
under the round-9 attack by construction (robustness inherited, not the point).

Predictive lane: each arm ADDS features to the drop_assistant base (fixed roles,
~0.769) and must beat it by >= 0.02 AUC with disjoint conversation-bootstrap CIs.

  A  order/recency over the tool-result sub-trajectory (last / recency / delta),
     with an order-shuffle negative control.
  B  goal-conditioned alignment (cosine of the goal to the result trajectory),
     with a mismatched-goal (deranged) topicality control.
  C  cross-channel consistency among fixed roles (request-result, call-result,
     repetition, novelty) - interactions a linear head over means cannot form.

    python round15.py
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
from tool_anchored import _service_embeddings

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
DECISION = "tool_result"
SEED = 42
BOOT = 1000
MARGIN = 0.02
LAMBDA = 0.5              # recency decay (frozen)
DIM = 384


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _derange(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    return p


def _order_feats(seq):
    """Per-step last / recency-weighted mean / consecutive delta over an ordered
    result sequence seq (K, DIM). Returns three (K, DIM) arrays."""
    K = len(seq)
    last = np.zeros((K, DIM))
    rec = np.zeros((K, DIM))
    delta = np.zeros((K, DIM))
    R = np.zeros(DIM)
    D = 0.0
    for k in range(K):
        last[k] = seq[k]
        R = LAMBDA * R + seq[k]
        D = LAMBDA * D + 1.0
        rec[k] = R / D
        delta[k] = seq[k] - seq[k - 1] if k > 0 else 0.0
    return last, rec, delta


def _goal_embeddings(conversations):
    texts = [c.get("goal_text", "") or "no goal" for c in conversations]
    ckpt = Checkpoint("round15", script=__file__)
    cached = ckpt.load("goals")
    if cached is not None and int(cached["n"]) == len(texts):
        return cached["vecs"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(texts, batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)
    ckpt.save("goals", {"n": np.array(len(texts)), "vecs": vecs})
    return vecs


def _collect(emb, off, meta, conversations, svc_vecs, svc_idx, per_svc,
             goals, mism):
    """One record per decision point with the base role means plus every R15
    feature (fixed channels only)."""
    cols = {k: [] for k in (
        "u", "tc", "tr", "last", "rec", "delta",
        "last_sh", "rec_sh", "delta_sh")}
    sca = {k: [] for k in (
        "goal_res", "goal_prog", "goal_mis", "xreq", "xcall", "xrep",
        "xnov")}
    conv = []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        g, gm = goals[i].astype(np.float64), mism[i].astype(np.float64)
        # pass 1: running context recorded at each tool_result turn
        usum = np.zeros(DIM)
        ucnt = 0
        tcsum = np.zeros(DIM)
        tccnt = 0
        sksum = np.zeros(DIM)
        skcnt = 0
        seen = set()
        ncall = nrep = 0
        ctx = []
        res_seq = []
        tcptr = 0
        for t in range(len(E)):
            r = tt[t]
            if r == "user":
                usum += E[t]
                ucnt += 1
            elif r == "tool_call":
                tcsum += E[t]
                tccnt += 1
                if tcptr < len(per_svc[i]):
                    tn = per_svc[i][tcptr]
                    sksum += svc_vecs[svc_idx[tn]]
                    skcnt += 1
                    ncall += 1
                    if tn in seen:
                        nrep += 1
                    seen.add(tn)
                    tcptr += 1
            elif r == DECISION:
                res_seq.append(E[t])
                ctx.append((usum / max(ucnt, 1), tcsum / max(tccnt, 1),
                            sksum / max(skcnt, 1), nrep / max(ncall, 1)))
        if not res_seq:
            continue
        # pass 2: ordered result features (causal) and cross-channel scalars.
        # The shuffle control permutes only the ALREADY-SEEN prefix at each
        # step (destroys order, keeps the seen set, leaks no future result) -
        # a full-sequence permutation would leak future results into early
        # decision points and inflate the control.
        res_seq = np.array(res_seq)
        last, rec, delta = _order_feats(res_seq)
        rng_c = np.random.default_rng([SEED, i])
        csum = np.zeros(DIM)
        gres_prev = None
        nov_sum = 0.0
        for k in range(len(res_seq)):
            csum += res_seq[k]
            trmean = csum / (k + 1)
            u, tc, skel, xrep = ctx[k]
            sh = res_seq[:k + 1][rng_c.permutation(k + 1)]
            R = np.zeros(DIM)
            D = 0.0
            for j in range(k + 1):
                R = LAMBDA * R + sh[j]
                D = LAMBDA * D + 1.0
            sh_rec = R / D
            gres = _cos(g, trmean)
            gprog = 0.0 if gres_prev is None else gres - gres_prev
            gres_prev = gres
            if k > 0:
                nov_sum += 1.0 - _cos(res_seq[k], res_seq[k - 1])
            xnov = nov_sum / max(k, 1)
            cols["u"].append(u)
            cols["tc"].append(tc)
            cols["tr"].append(trmean)
            cols["last"].append(last[k])
            cols["rec"].append(rec[k])
            cols["delta"].append(delta[k])
            cols["last_sh"].append(sh[-1])
            cols["rec_sh"].append(sh_rec)
            cols["delta_sh"].append(sh[-1] - sh[-2] if k > 0
                                    else np.zeros(DIM))
            sca["goal_res"].append(gres)
            sca["goal_prog"].append(gprog)
            sca["goal_mis"].append(_cos(gm, trmean))
            sca["xreq"].append(_cos(u, trmean))
            sca["xcall"].append(_cos(skel, trmean))
            sca["xrep"].append(xrep)
            sca["xnov"].append(xnov)
            conv.append(i)
    out = {k: np.array(v, dtype=np.float32) for k, v in cols.items()}
    for k, v in sca.items():
        out[k] = np.array(v, dtype=np.float64)[:, None]
    out["conv"] = np.array(conv)
    return out


def _auc_ci(scores, labels, conv, boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uc = np.unique(conv)
    by = {c: np.where(conv == c)[0] for c in uc}
    aucs = []
    for _ in range(boot):
        samp = rng.choice(uc, size=len(uc), replace=True)
        pts = np.concatenate([by[c] for c in samp])
        ys = labels[pts]
        if ys.min() == ys.max():
            continue
        aucs.append(roc_auc_score(ys, scores[pts]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    conversations = json.loads(CONV.read_text())
    svc_vecs, svc_idx, per_svc = _service_embeddings(conversations)
    goals = _goal_embeddings(conversations)
    mism = goals[_derange(len(conversations), SEED)]

    d = _collect(emb, off, meta, conversations, svc_vecs, svc_idx, per_svc,
                 goals, mism)
    conv = d["conv"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(d["u"], sy, conv))
    conv_te = conv[te]
    base = np.hstack([d["u"], d["tc"], d["tr"]])

    def evalX(X):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[tr], sy[tr])
        s = clf.decision_function(X[te])
        return round(float(roc_auc_score(sy[te], s)), 4), \
            _auc_ci(s, sy[te], conv_te)

    base_auc, base_ci = evalX(base)

    def arm(*extra):
        return evalX(np.hstack([base] + list(extra)))

    def rec(auc_ci):
        auc, ci = auc_ci
        return {"auc": auc, "ci95": ci, "delta_vs_base": round(auc - base_auc,
                4), "beats_base": bool(auc - base_auc >= MARGIN
                                       and not _overlap(ci, base_ci))}

    # -------- Direction A: order / recency --------
    a_last = rec(arm(d["last"]))
    a_rec = rec(arm(d["rec"]))
    a_delta = rec(arm(d["delta"]))
    a_all = rec(arm(d["last"], d["rec"], d["delta"]))
    a_shuf = rec(arm(d["last_sh"], d["rec_sh"], d["delta_sh"]))
    best_a = max((a_last, a_rec, a_delta, a_all), key=lambda r: r["auc"])
    a_keep = bool(best_a["beats_base"]
                  and best_a["auc"] - a_shuf["auc"] >= MARGIN)
    direction_a = {
        "base": {"auc": base_auc, "ci95": base_ci},
        "+last": a_last, "+recency": a_rec, "+delta": a_delta, "+all": a_all,
        "+all_shuffled": a_shuf,
        "keep_order_carries_signal": a_keep,
        "verdict": ("KEEP: order in the result channel adds signal"
                    if a_keep else
                    "KILL: degree-0 on the results is near-optimal"),
    }

    # -------- Direction B: goal-conditioning --------
    b_res = rec(arm(d["goal_res"]))
    b_prog = rec(arm(d["goal_prog"]))
    b_all = rec(arm(d["goal_res"], d["goal_prog"]))
    b_mis = rec(arm(d["goal_mis"]))
    best_b = max((b_res, b_all), key=lambda r: r["auc"])
    b_keep = bool(best_b["beats_base"]
                  and best_b["auc"] - b_mis["auc"] >= MARGIN)
    direction_b = {
        "base": {"auc": base_auc, "ci95": base_ci},
        "+goal_result": b_res, "+goal_progress": b_prog, "+goal_all": b_all,
        "+goal_mismatch": b_mis,
        "keep_goal_conditioning": b_keep,
        "verdict": ("KEEP: goal alignment adds goal-specific signal"
                    if b_keep else
                    "KILL: dense goal-conditioning adds nothing over base "
                    "(or is topicality, not goal-specific)"),
    }

    # -------- Direction C: cross-channel consistency --------
    c_cons = rec(arm(d["xreq"], d["xcall"], d["xrep"], d["xnov"]))
    c_rep = rec(arm(d["xrep"]))
    rep_auc, rep_ci = evalX(d["xrep"])
    direction_c = {
        "base": {"auc": base_auc, "ci95": base_ci},
        "+consistency": c_cons, "+repetition": c_rep,
        "repetition_only": {"auc": rep_auc, "ci95": rep_ci,
                            "above_chance": bool(rep_ci[0] > 0.5)},
        "keep_consistency": bool(c_cons["beats_base"]),
        "verdict": ("KEEP: cross-channel consistency adds signal"
                    if c_cons["beats_base"] else
                    "KILL: role means already capture the cross-channel "
                    "information"),
    }

    res = {"dataset": "toolbench", "seed": SEED,
           "n_decision_points_test": int(len(te)),
           "n_test_conversations": int(len(np.unique(conv_te))),
           "base_auc": base_auc, "keep_margin": MARGIN,
           "direction_a": direction_a, "direction_b": direction_b,
           "direction_c": direction_c}
    save_result("round15", res, dataset="toolbench", params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
