"""Round 14: is fixed-roles just error-counting? does the tool identity encoding
help when enriched? structured flag vs semantic payload for the tool result.
PREREGISTRATION-R14.md (+ 2026-07-24 amendment).

Four directions, thresholds frozen before the run:

  A  error-rate baseline (the missing ToolBench process baseline): err_naive
     (running fraction of tool results with a non-empty `error` field) and
     err_strong (+ empty payload + a frozen failure-string set), each a scalar,
     scored by the same head; compared to drop_assistant with conversation
     bootstrap CIs. Keep if drop_assistant beats max(err) by >= 0.03 AUC with
     non-overlapping CIs; requalify Step 5 if err_strong is within 0.01.
  B  enriched skeleton: skel_service (baseline), skel_name (embed tool_name),
     skel_argkeys (embed the aggregated argument-key set), evaluated in-domain
     and on a frozen held-out-tool split. Null predicted in-domain, gain
     predicted out-of-domain.
  C  structured vs semantic tool result: struct (parsed flags), semantic (the
     tool_result embedding mean), both. Semantic predicted to beat struct.
  D  firewall: the success-channel arms (A, C) are invariant under the
     narration attack by construction - that 0 is NOT robustness. They are
     predictor-lane until the action-substitution adversary runs; the ABCD
     action-substitution slack (policy_choice) is the action-channel evidence.

`metadata.success` is BANNED as a feature (it is the conversation label copied
onto every call - see the amendment). All A/C signal is parsed from the
tool_result `content` text only.

    python round14.py
"""
import hashlib
import json
import re
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
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
POLICY = REPO / "data" / "results" / "experiments" / "policy_choice_abcd.json"
TYPES = ["user", "assistant", "tool_call", "tool_result"]
CONTEXT = ["user", "tool_call", "tool_result"]   # non-assistant (fixed) roles
DECISION = "tool_result"
SEED = 42
BOOT = 1000
KEEP_MARGIN = 0.03        # drop_assistant must beat err by this (Direction A)
NULL_MARGIN = 0.01        # in-domain "null" band (Direction B)

# frozen in PREREGISTRATION-R14.md, not tuned against outcomes
FAILSTR = ("disabled", "rate limit", "blocked", "not found",
           "no such function", "does not exist")
ERR_RE = re.compile(r'"error"\s*:\s*"((?:[^"\\]|\\.)*)"')
RESP_RE = re.compile(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"')
SVDIM = 9                 # structured per-result vector


def _svec(content):
    """Structured encoding of one tool_result, parsed from content text only.
    Returns (vector[SVDIM], naive_error_flag, strong_failure_flag)."""
    c = content or ""
    low = c.lower()
    m = ERR_RE.search(c)
    err = (m.group(1) if m else "").strip()
    m2 = RESP_RE.search(c)
    resp = m2.group(1) if m2 else None
    empty = 1.0 if (resp is not None
                    and resp.strip() in ("", "[]", "{}", "\\n", "none",
                                         "null")) else 0.0
    failstr = 1.0 if any(s in low for s in FAILSTR) else 0.0
    errored = 1.0 if err else 0.0
    rate = 1.0 if "rate limit" in low else 0.0
    disabled = 1.0 if "disabled" in low else 0.0
    server = 1.0 if ("status_code=5" in low or "data error" in low) else 0.0
    notfound = 1.0 if any(s in low for s in ("not found", "does not exist",
                                             "no such function",
                                             "blocked")) else 0.0
    ok = 1.0 if not (errored or empty or failstr) else 0.0
    length = min(len(resp) if resp is not None else len(c), 1000) / 1000.0
    strong = 1.0 if (errored or empty or failstr) else 0.0
    vec = np.array([errored, empty, failstr, rate, disabled, server, notfound,
                    ok, length], dtype=np.float64)
    return vec, errored, strong


def _tool_variants(conversations):
    """Three tool-identity embedding variants keyed by tool_name (all
    deterministic functions of tool id): service text, tool_name text, and the
    aggregated argument-key set text. Returns {variant: vecs}, name->row index,
    and the per-conversation tool_name sequence."""
    per_names = []
    svc_of, keys_of = {}, {}
    for c in conversations:
        seq = []
        for t in c["turns"]:
            if t["turn_type"] == "tool_call":
                md = t.get("metadata", {})
                tn = md.get("tool_name") or md.get("service") or "unknown"
                seq.append(tn)
                svc_of[tn] = md.get("service", "unknown")
                keys_of.setdefault(tn, set()).update(
                    (md.get("parameters") or {}).keys())
        per_names.append(seq)
    names = sorted(svc_of)
    texts = {
        "service": [svc_of[n].replace("_", " ") for n in names],
        "name": [n.replace("_", " ") for n in names],
        "argkeys": [(" ".join(sorted(keys_of[n])) or "no arguments")
                    for n in names],
    }
    ckpt = Checkpoint("round14", script=__file__)
    cached = ckpt.load("tool_variants")
    if cached is not None and list(cached["names"]) == names:
        variants = {v: cached[v] for v in texts}
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        variants = {v: model.encode(texts[v], batch_size=256,
                                    normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
                    for v in texts}
        ckpt.save("tool_variants", {"names": np.array(names), **variants})
    name_idx = {n: k for k, n in enumerate(names)}
    return variants, name_idx, per_names


def _collect(emb, off, meta, conversations, variants, name_idx, per_names):
    """One record per decision point (tool_result turn): per-role running means,
    the three skeleton-variant running means, err_naive / err_strong running
    fractions, and the structured running-mean vector."""
    role = {r: [] for r in TYPES}
    skel = {v: [] for v in variants}
    errN, errS, struct, conv = [], [], [], []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        turns = conversations[i]["turns"]
        rsum = {r: np.zeros(384) for r in TYPES}
        rcnt = {r: 0 for r in TYPES}
        sk = {v: np.zeros(384) for v in variants}
        skc = 0
        sv = np.zeros(SVDIM)
        nres = nnaive = nstrong = 0
        tc = 0
        seq = per_names[i]
        for t in range(len(E)):
            r = tt[t]
            rsum[r] += E[t]
            rcnt[r] += 1
            if r == "tool_call" and tc < len(seq):
                row = name_idx.get(seq[tc])
                if row is not None:
                    for v in variants:
                        sk[v] += variants[v][row]
                    skc += 1
                tc += 1
            if r == DECISION:
                content = turns[t].get("content", "") if t < len(turns) else ""
                vec, naive, strong = _svec(content)
                sv += vec
                nres += 1
                nnaive += naive
                nstrong += strong
                for rr in TYPES:
                    role[rr].append(rsum[rr] / max(rcnt[rr], 1))
                for v in variants:
                    skel[v].append(sk[v] / max(skc, 1))
                errN.append(nnaive / nres)
                errS.append(nstrong / nres)
                struct.append(sv / nres)
                conv.append(i)
    out = {r: np.array(role[r], dtype=np.float32) for r in TYPES}
    for v in variants:
        out["sk_" + v] = np.array(skel[v], dtype=np.float32)
    out["errN"] = np.array(errN)[:, None]
    out["errS"] = np.array(errS)[:, None]
    out["struct"] = np.array(struct, dtype=np.float32)
    out["conv"] = np.array(conv)
    return out


def _fit_auc(X, sy, conv, tr, te):
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000).fit(X[tr], sy[tr])
    s = clf.decision_function(X[te])
    return float(roc_auc_score(sy[te], s)), s


def _auc_ci(scores, labels, conv, boot=BOOT, seed=SEED):
    """Conversation-level bootstrap 95% CI for AUC over decision points."""
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
    """Do two [lo,hi] CIs overlap?"""
    return not (a[1] < b[0] or b[1] < a[0])


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    conversations = json.loads(CONV.read_text())

    variants, name_idx, per_names = _tool_variants(conversations)
    d = _collect(emb, off, meta, conversations, variants, name_idx, per_names)
    conv = d["conv"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(d["errN"], sy, conv))
    conv_te = conv[te]

    def arm(X):
        auc, s = _fit_auc(X, sy, conv, tr, te)
        return auc, _auc_ci(s, sy[te], conv_te)

    # -------- Direction A: error-rate baseline vs drop_assistant --------
    drop_assistant = np.hstack([d[r] for r in CONTEXT])
    da_auc, da_ci = arm(drop_assistant)
    n_auc, n_ci = arm(d["errN"])
    s_auc, s_ci = arm(d["errS"])
    best_err = max(n_auc, s_auc)
    beats = (da_auc - best_err) >= KEEP_MARGIN
    err_best_ci = n_ci if n_auc >= s_auc else s_ci
    disjoint = not _overlap(da_ci, err_best_ci)
    requalify = (da_auc - s_auc) <= NULL_MARGIN or _overlap(da_ci, s_ci)
    direction_a = {
        "drop_assistant": {"auc": round(da_auc, 4), "ci95": da_ci},
        "err_naive": {"auc": round(n_auc, 4), "ci95": n_ci},
        "err_strong": {"auc": round(s_auc, 4), "ci95": s_ci},
        "gap_drop_minus_best_err": round(da_auc - best_err, 4),
        "keep_drop_beats_err_by_margin": bool(beats and disjoint),
        "requalify_step5": bool(requalify),
        "verdict": ("KEEP: fixed-roles carry signal beyond error-counting"
                    if beats and disjoint else
                    "REQUALIFY: fixed-roles ~= error-counting" if requalify
                    else "INCONCLUSIVE (gap < margin but CIs disjoint)"),
    }

    # -------- Direction B: enriched skeleton, in-domain + held-out-tool ----
    def hashed_unseen(tn):
        h = int(hashlib.md5(tn.encode()).hexdigest(), 16)
        return h % 5 == 0            # ~20% of tool ids held out (frozen)
    tools_of = [set(s) for s in per_names]
    seen_conv = np.array([len(tools_of[i]) > 0
                          and not any(hashed_unseen(t) for t in tools_of[i])
                          for i in range(len(off) - 1)])
    unseen_conv = np.array([len(tools_of[i]) > 0
                            and all(hashed_unseen(t) for t in tools_of[i])
                            for i in range(len(off) - 1)])
    in_tr = np.isin(conv, np.where(seen_conv)[0])
    in_te = np.isin(conv, np.where(unseen_conv)[0])

    b_variants = {"skel_service": "sk_service", "skel_name": "sk_name",
                  "skel_argkeys": "sk_argkeys"}
    direction_b = {"in_domain": {}, "held_out_tool": {}}
    for label, key in b_variants.items():
        auc_id, _ = _fit_auc(d[key], sy, conv, tr, te)
        direction_b["in_domain"][label] = round(auc_id, 4)
        if in_tr.sum() and in_te.sum() and sy[in_te].min() != sy[in_te].max():
            auc_ood, _ = _fit_auc(d[key], sy, conv,
                                  np.where(in_tr)[0], np.where(in_te)[0])
            direction_b["held_out_tool"][label] = round(float(auc_ood), 4)
    base_id = direction_b["in_domain"]["skel_service"]
    direction_b["in_domain_gain_name"] = round(
        direction_b["in_domain"]["skel_name"] - base_id, 4)
    direction_b["in_domain_gain_argkeys"] = round(
        direction_b["in_domain"]["skel_argkeys"] - base_id, 4)
    direction_b["in_domain_null_confirmed"] = bool(
        abs(direction_b["in_domain_gain_name"]) <= NULL_MARGIN
        and abs(direction_b["in_domain_gain_argkeys"]) <= NULL_MARGIN)
    direction_b["n_pure_seen_conv"] = int(seen_conv.sum())
    direction_b["n_pure_unseen_conv"] = int(unseen_conv.sum())
    if direction_b["held_out_tool"]:
        hb = direction_b["held_out_tool"]
        direction_b["held_out_gain_name"] = round(
            hb.get("skel_name", 0) - hb.get("skel_service", 0), 4)
        direction_b["held_out_gain_argkeys"] = round(
            hb.get("skel_argkeys", 0) - hb.get("skel_service", 0), 4)

    # -------- Direction C: structured vs semantic tool result --------
    sk = d["sk_service"]
    struct = np.hstack([sk, d["struct"]])
    semantic = np.hstack([sk, d["tool_result"]])
    both = np.hstack([sk, d["struct"], d["tool_result"]])
    st_auc, st_ci = arm(struct)
    se_auc, se_ci = arm(semantic)
    bo_auc, bo_ci = arm(both)
    # empty-error-field failures: strong flag set but naive (error field) not
    n_empty_err_fail = 0
    for i in range(len(off) - 1):
        turns = conversations[i]["turns"]
        for t in turns:
            if t["turn_type"] == DECISION:
                _, naive, strong = _svec(t.get("content", ""))
                if strong and not naive:
                    n_empty_err_fail += 1
    direction_c = {
        "struct": {"auc": round(st_auc, 4), "ci95": st_ci},
        "semantic": {"auc": round(se_auc, 4), "ci95": se_ci},
        "both": {"auc": round(bo_auc, 4), "ci95": bo_ci},
        "semantic_minus_struct": round(se_auc - st_auc, 4),
        "both_minus_semantic": round(bo_auc - se_auc, 4),
        "keep_semantic": bool((se_auc - st_auc) >= KEEP_MARGIN
                              and not _overlap(se_ci, st_ci)),
        "n_empty_error_field_failures": n_empty_err_fail,
    }

    # -------- Direction D: firewall --------
    policy = json.loads(POLICY.read_text()) if POLICY.exists() else {}
    direction_d = {
        "lane": "predictor",
        "narration_attack_manipulability": 0.0,
        "note": ("success-channel arms (A err-rate, C struct) are invariant "
                 "under the narration-only selection attack by construction - "
                 "that 0 is NOT action-robustness and is not quoted as reward "
                 "robustness. They stay predictor-lane until the action-"
                 "substitution adversary runs."),
        "action_substitution_evidence_abcd": {
            "same_task_slack": policy.get("same_subflow", {}).get("slack_rate"),
            "any_task_slack": policy.get("any_subflow", {}).get("slack_rate"),
            "source": "policy_choice_abcd.json",
        },
    }

    res = {"dataset": "toolbench", "seed": SEED,
           "n_decision_points_test": int(len(te)),
           "n_test_conversations": int(len(np.unique(conv_te))),
           "direction_a": direction_a, "direction_b": direction_b,
           "direction_c": direction_c, "direction_d": direction_d}
    save_result("round14", res, dataset="toolbench", params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
