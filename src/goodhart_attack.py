"""Round-5 Arm G: Goodhart attack on the early-warning head (sandbox;
the anti-Goodhart charter's sanctioned violation - nothing here is
optimized against in production). PREREGISTRATION-R5.md.

Targets the per-corpus head on the two corpora where it works (toolbench,
abcd). Head + alert threshold are trained on GroupKFold fold-0 TRAIN;
every attack metric is on held-out failing conversations.

Attack 1: best-of-n selection - substitute each upcoming agentlike turn
with the pool candidate (real agentlike turns of successful training
conversations) that minimizes the head's current risk; measure alert
recall vs n. n=1 = random substitution (drift control).

Attack 2: white-box direction - block norms of the head weight, minimal
L2 shift of the agentlike block to cross the threshold at the median
failing prefix, and the nearest real agentlike turns to the exploit
direction (readability).

    python goodhart_attack.py toolbench
    python goodhart_attack.py abcd
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
SEED = 42
NS = [1, 2, 4, 8, 16]

CORPUS = {
    "toolbench": {"types": ["user", "assistant", "tool_call",
                            "tool_result"],
                  "decision": "tool_result", "agentlike": "assistant"},
    "abcd": {"types": ["customer", "agent", "action"],
             "decision": "customer", "agentlike": "agent"},
}


def _load(dataset):
    z = np.load(EMB / f"{dataset}_minilm.npz")
    meta = json.loads((EMB / f"{dataset}_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    gid = np.array(meta.get("group_id", np.arange(len(y))))
    return z["embeddings"], z["offsets"], y, gid, meta


def _prefix_samples(emb, off, meta, cfg):
    """Per-decision-turn prefix features (per-type running means) + the
    conversation index and tau."""
    types = cfg["types"]
    X, conv, tau = [], [], []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        m = len(E)
        onehot = np.stack([(tt == t).astype(np.float32) for t in types],
                          axis=1)
        csum = np.cumsum(E[:, None, :] * onehot[:, :, None], axis=0)
        ccnt = np.cumsum(onehot, axis=0)
        for t in np.where(tt == cfg["decision"])[0]:
            means = csum[t] / np.maximum(ccnt[t][:, None], 1.0)
            X.append(means.reshape(-1))
            conv.append(i)
            tau.append(t / max(m - 1, 1))
    return np.array(X, np.float32), np.array(conv), np.array(tau)


def _threshold(scores, conv, y_conv, target=0.5):
    """Lowest score threshold whose alerted-conversation precision >=
    target (conversation alerts if any prefix score >= threshold)."""
    grid = np.percentile(scores, np.arange(50, 100))
    best = None
    for thr in grid:
        alerted = np.unique(conv[scores >= thr])
        if not len(alerted):
            continue
        prec = y_conv[alerted].mean()
        if prec >= target:
            best = thr
            break
    return best if best is not None else grid[-1]


def _score(clf, feat):
    return clf.decision_function(feat.reshape(1, -1))[0]


def run(dataset):
    cfg = CORPUS[dataset]
    types = cfg["types"]
    ai = types.index(cfg["agentlike"])
    emb, off, y, gid, meta = _load(dataset)

    X, conv, tau = _prefix_samples(emb, off, meta, cfg)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(X, sy, gid[conv]))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(X[tr_idx], sy[tr_idx])

    tr_scores = clf.decision_function(X[tr_idx])
    tr_conv = conv[tr_idx]
    y_by_conv = {c: y[c] for c in np.unique(conv)}
    yc = np.array([y_by_conv[c] for c in np.unique(tr_conv)])
    remap = {c: j for j, c in enumerate(np.unique(tr_conv))}
    thr = _threshold(tr_scores, np.array([remap[c] for c in tr_conv]),
                     yc)

    train_convs = set(np.unique(conv[tr_idx]).tolist())
    test_convs = set(np.unique(conv[te_idx]).tolist())

    # candidate pool: agentlike turns of SUCCESSFUL training conversations
    pool = []
    for i in train_convs:
        if y[i]:
            continue
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        pool.append(E[tt == cfg["agentlike"]])
    pool = np.vstack([p for p in pool if len(p)]).astype(np.float32)

    rng = np.random.default_rng(SEED)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])

    def attack_conv(i, n):
        """Causal walk with best-of-n substitution of agentlike turns;
        returns whether the conversation still alerts at thr."""
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = np.array(meta["turn_types"][i])
        sums = np.zeros((len(types), 384))
        cnts = np.zeros(len(types))
        alerted = False
        for t in range(len(E)):
            ty = tt[t]
            if ty == cfg["agentlike"]:
                cand = pool[rng.integers(0, len(pool), size=n)]
                trial_mean = (sums[ai] + cand) / (cnts[ai] + 1)
                feats = np.tile(np.concatenate([
                    (sums[k] / max(cnts[k], 1)) for k in range(len(types))
                ]), (n, 1))
                feats[:, ai * 384:(ai + 1) * 384] = trial_mean
                if n == 1:
                    pick = 0
                else:
                    pick = int(np.argmin(feats @ w + b))
                chosen = cand[pick]
                sums[ai] += chosen
                cnts[ai] += 1
            else:
                k = types.index(ty) if ty in types else None
                if k is not None:
                    sums[k] += E[t]
                    cnts[k] += 1
            if ty == cfg["decision"]:
                feat = np.concatenate([sums[k] / max(cnts[k], 1)
                                       for k in range(len(types))])
                if feat @ w + b >= thr:
                    alerted = True
        return alerted

    fail_test = [i for i in test_convs if y[i]]
    recall_by_n = {}
    for n in NS:
        alerts = sum(attack_conv(i, n) for i in fail_test)
        recall_by_n[n] = alerts / max(len(fail_test), 1)

    # baseline recall (no attack) at thr on held-out failures
    base_alerts = 0
    for i in fail_test:
        m = conv == i
        if (clf.decision_function(X[m]) >= thr).any():
            base_alerts += 1
    base_recall = base_alerts / max(len(fail_test), 1)

    # Attack 2: white-box
    block_norms = [float(np.linalg.norm(w[k * 384:(k + 1) * 384]))
                   for k in range(len(types))]
    fail_feats = X[(sy == 1)]
    med = np.median(fail_feats, axis=0)
    logit_thr = thr
    wa = w[ai * 384:(ai + 1) * 384]
    score_med = med @ w + b
    min_shift = float(max(score_med - logit_thr, 0) / np.linalg.norm(wa))
    direction = -wa / np.linalg.norm(wa)
    pooln = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True),
                              1e-9)
    sims = pooln @ direction
    top5 = np.argsort(sims)[::-1][:5]

    degradation = ((base_recall - recall_by_n[NS[-1]]) / base_recall
                   if base_recall > 0 else 0.0)
    verdict = ("gameable" if degradation > 0.5 else
               "selection-robust" if degradation < 0.1 else "graded")
    res = {"dataset": dataset, "threshold": float(thr),
           "n_test_failures": len(fail_test),
           "base_recall": float(base_recall),
           "recall_by_n": {str(k): float(v) for k, v in recall_by_n.items()},
           "drift_control_n1_recall": float(recall_by_n[1]),
           "degradation_at_n16": float(degradation),
           "verdict": verdict,
           "block_norms": dict(zip(types, block_norms)),
           "agentlike_block": cfg["agentlike"],
           "min_L2_shift_median_failure": min_shift,
           "nearest_exploit_turn_sims": [float(sims[j]) for j in top5]}
    save_result(f"goodhart_{dataset}", res, dataset=dataset,
                params={"seed": SEED, "ns": NS})
    print(json.dumps({k: v for k, v in res.items()
                      if k != "nearest_exploit_turn_sims"}, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
