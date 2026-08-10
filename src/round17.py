"""Round 17: the two-mode deployment - running early-warning + terminal read.
PREREGISTRATION-R17.md.

On the leading method (the fixed-role degree-0 head), three questions:
  A  is there actionable lead? lead-time operating characteristic at conv-level
     FPR 0.10, two-mode (any decision point) vs terminal-only, versus a
     same-turn-content baseline. ToolBench + ABCD (lead headroom).
  B  does the running mode need degree-1? degree-0 vs degree-1 fixed-role AUC,
     stratified early vs late by within-conversation position, slope-shuffle.
  C  is the terminal read degree-0? degree-0 vs degree-1 at the terminal point.

All arms are fixed-role and narration-invariant; robustness inherited.

    python round17.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result
from round16 import _fit01

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
DIM = 384
SEED = 42
BOOT = 1000
FPR = 0.10
LEAD_MIN = 2
MIN_DP = 3
AUC_MARGIN = 0.02

CFG = {
    "toolbench": {"npz": "toolbench_minilm.npz", "meta": "toolbench_meta.json",
                  "fixed": ["user", "tool_call", "tool_result"],
                  "decision": "tool_result"},
    "abcd": {"npz": "abcd_minilm.npz", "meta": "abcd_meta.json",
             "fixed": ["customer", "action"], "decision": "customer"},
}


def _collect(corpus):
    cfg = CFG[corpus]
    z = np.load(EMB / cfg["npz"])
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / cfg["meta"]).read_text())
    y = np.array(meta["has_failure"], dtype=int)
    fixed, decision = cfg["fixed"], cfg["decision"]
    a0 = {r: [] for r in fixed}
    a1 = {r: [] for r in fixed}
    a1s = {r: [] for r in fixed}
    content, conv, dpi, K = [], [], [], []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        buf = {r: [] for r in fixed}
        rng = np.random.default_rng([SEED, i])
        rows = []
        for t in range(len(E)):
            if tt[t] in buf:
                buf[tt[t]].append(E[t])
            if tt[t] == decision:
                feat = {}
                for r in fixed:
                    seq = np.array(buf[r]) if buf[r] else np.zeros((1, DIM))
                    c0, c1 = _fit01(seq)
                    _, cs = _fit01(seq, shuffle_rng=rng)
                    feat[r] = (c0, c1, cs)
                rows.append((feat, E[t]))
        for j, (feat, last) in enumerate(rows):
            for r in fixed:
                a0[r].append(feat[r][0])
                a1[r].append(feat[r][1])
                a1s[r].append(feat[r][2])
            content.append(last)
            conv.append(i)
            dpi.append(j)
            K.append(len(rows))
    out = {"base": np.hstack([np.array(a0[r], dtype=np.float32)
                              for r in fixed]),
           "slope": np.hstack([np.array(a1[r], dtype=np.float32)
                               for r in fixed]),
           "slope_sh": np.hstack([np.array(a1s[r], dtype=np.float32)
                                  for r in fixed]),
           "content": np.array(content, dtype=np.float32),
           "conv": np.array(conv), "dpi": np.array(dpi),
           "K": np.array(K), "y": y}
    out["base"] = np.hstack([out["base"]])
    out["slope"] = np.hstack([out["base"], out["slope"]])
    out["slope_sh"] = np.hstack([out["base"], out["slope_sh"]])
    return out


def _fpr_thr(scores, conv, y, fpr=FPR):
    smax = [scores[conv == c].max() for c in np.unique(conv) if not y[c]]
    return float(np.quantile(smax, 1 - fpr))


def _lead_stats(scores, conv, y, thr):
    """Per-conversation first-alert lead. Lead stats restricted to failing
    conversations with >= MIN_DP decision points."""
    leads, any_f, term_f, nf = [], 0, 0, 0
    fpr_hit, ns = 0, 0
    for c in np.unique(conv):
        idx = np.where(conv == c)[0]            # in decision-point order
        s = scores[idx]
        alert = s >= thr
        if y[c]:
            if len(idx) < MIN_DP:
                continue
            nf += 1
            if alert.any():
                any_f += 1
                first = int(np.argmax(alert))
                leads.append((len(idx) - 1) - first)
            if alert[-1]:
                term_f += 1
        else:
            if len(idx) < MIN_DP:
                continue
            ns += 1
            if alert.any():
                fpr_hit += 1
    leads = np.array(leads)
    return {
        "n_fail": nf,
        "recall_anypoint": round(any_f / max(nf, 1), 3),
        "recall_terminal_only": round(term_f / max(nf, 1), 3),
        "median_lead": float(np.median(leads)) if len(leads) else 0.0,
        # absolute early-recall: fraction of ALL failures caught >= L steps
        # early (not conditional on being alerted), the honest early-detection
        # rate for the confound comparison.
        "early_recall_ge2": round(float((leads >= 2).sum()) / max(nf, 1), 3),
        "early_recall_ge1": round(float((leads >= 1).sum()) / max(nf, 1), 3),
        "fpr_conv": round(fpr_hit / max(ns, 1), 3),
    }


def _auc_ci(s, yv, conv, boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uc = np.unique(conv)
    by = {c: np.where(conv == c)[0] for c in uc}
    vals = []
    for _ in range(boot):
        pts = np.concatenate([by[c] for c in rng.choice(uc, len(uc), True)])
        yy = yv[pts]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, s[pts]))
    return round(float(np.percentile(vals, 2.5)), 4), \
        round(float(np.percentile(vals, 97.5)), 4)


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def _fit(X, sy, tr):
    return LogisticRegression(C=1.0, class_weight="balanced",
                              max_iter=2000).fit(X[tr], sy[tr])


def _direction_a(corpus):
    d = _collect(corpus)
    conv, y = d["conv"], d["y"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(d["base"], sy, conv))
    tr_conv, te_conv = conv[tr], conv[te]

    def head(feat):
        clf = _fit(d[feat], sy, tr)
        s_tr = clf.decision_function(d[feat][tr])
        s_te = clf.decision_function(d[feat][te])
        thr = _fpr_thr(s_tr, tr_conv, y)
        return _lead_stats(s_te, te_conv, y, thr)

    geom = head("base")
    cont = head("content")
    keep = bool(geom["early_recall_ge2"] >= 0.25
                and geom["early_recall_ge2"] > cont["early_recall_ge2"])
    return {"geometry": geom, "content_baseline": cont,
            "two_mode_gain_recall": round(geom["recall_anypoint"]
                                          - geom["recall_terminal_only"], 3),
            "keep_running_mode": keep}


def _direction_bc(corpus="toolbench"):
    d = _collect(corpus)
    conv, y, dpi, K = d["conv"], d["y"], d["dpi"], d["K"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(d["base"], sy, conv))
    early = dpi < (K / 2.0)
    terminal = dpi == (K - 1)
    masks = {"early": early[te], "late": ~early[te],
             "terminal": terminal[te]}

    heads = {f: _fit(d[f], sy, tr) for f in ("base", "slope", "slope_sh")}
    scores = {f: heads[f].decision_function(d[f][te]) for f in heads}
    yte, cte = sy[te], conv[te]

    out = {}
    for stratum, m in masks.items():
        row = {}
        for f, lbl in (("base", "degree0"), ("slope", "degree1"),
                       ("slope_sh", "degree1_shuffled")):
            auc = round(float(roc_auc_score(yte[m], scores[f][m])), 4)
            ci = _auc_ci(scores[f][m], yte[m], cte[m])
            row[lbl] = {"auc": auc, "ci95": ci}
        row["degree1_minus_degree0"] = round(
            row["degree1"]["auc"] - row["degree0"]["auc"], 4)
        row["degree1_minus_shuffle"] = round(
            row["degree1"]["auc"] - row["degree1_shuffled"]["auc"], 4)
        out[stratum] = row

    b_keep = bool(out["early"]["degree1_minus_degree0"] >= AUC_MARGIN
                  and not _overlap(out["early"]["degree1"]["ci95"],
                                   out["early"]["degree0"]["ci95"])
                  and out["early"]["degree1_minus_shuffle"] >= AUC_MARGIN)
    c_confirm = bool(out["terminal"]["degree1_minus_degree0"] < AUC_MARGIN)
    return out, b_keep, c_confirm


def run():
    a = {c: _direction_a(c) for c in ("toolbench", "abcd")}
    bc, b_keep, c_confirm = _direction_bc("toolbench")

    tb = a["toolbench"]["geometry"]
    if not a["toolbench"]["keep_running_mode"]:
        verdict = "ONE MODE: no early-warning value; terminal read only"
    elif b_keep:
        verdict = "TWO-MODE (two representations): degree-1 running, degree-0 terminal"
    else:
        verdict = ("TWO-MODE (two read-positions of ONE degree-0 head): "
                   "early flag + terminal level read, same representation")
    res = {
        "seed": SEED,
        "direction_a_leadtime": a,
        "direction_b_stratified": {"toolbench": bc},
        "direction_c_terminal_is_degree0": c_confirm,
        "b_running_needs_degree1": b_keep,
        "verdict": verdict,
    }
    save_result("round17", res, dataset="mixed", params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
