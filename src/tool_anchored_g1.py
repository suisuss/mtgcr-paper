"""Round-9 addendum: does anchoring the FULL role-split(1) (per-role g0+g1,
i.e. degree-1 Gram) with the tool-call skeleton reduce manipulability?
Round 9 used degree-0 running means; this tests whether the tethering-does-
not-help finding is an artifact of dropping the drift coefficient g1.

Arms (ToolBench, round-9 attack: best-of-16 substitution of assistant turns,
fold-0 train head + threshold, seed 42):
  gram1_flat     - per-role running [g0, g1] (role-split(1)), fully NL
  gram1_tethered - gram1_flat + skeleton + discrepancy
Compared against round-9's committed degree-0 arms (same fold/seed):
  A_flat (g0)     AUC 0.788 / manip 0.50
  B_skeleton      AUC 0.706 / manip 0.00
  C_tethered (g0) AUC 0.788 / manip 0.50

    python tool_anchored_g1.py
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
from tool_anchored import _service_embeddings

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
TYPES = ["user", "assistant", "tool_call", "tool_result"]
AGENTLIKE = "assistant"
DECISION = "tool_result"
SEED = 42


def _g01(E):
    """Closed-form degree-1 Gram coefficients [g0, g1] (768-d) for a role
    subsequence E (n, 384). g1 basis is the linear ramp 1 - 2i/(n-1); this
    equals lib.gram.gram_fit(E, degree=1)[:2] (validated round 3)."""
    n = len(E)
    if n == 0:
        return np.zeros(768)
    if n == 1:
        return np.concatenate([E[0], np.zeros(384)])
    ramp = 1.0 - 2.0 * np.arange(n) / (n - 1)
    g1 = (ramp @ E) / (ramp @ ramp)
    return np.concatenate([E.mean(0), g1])


class State:
    def __init__(self):
        self.embs = {t: [] for t in TYPES}
        self.skel_sum = np.zeros(384)
        self.skel_cnt = 0

    def ingest(self, e, ttype, svc=None):
        self.embs[ttype].append(e)
        if ttype == "tool_call" and svc is not None:
            self.skel_sum += svc
            self.skel_cnt += 1

    def gram1(self):
        return np.concatenate([_g01(np.array(self.embs[t]) if self.embs[t]
                                    else np.zeros((0, 384)))
                               for t in TYPES])

    def skeleton(self):
        return self.skel_sum / max(self.skel_cnt, 1)

    def feat(self, arm):
        g = self.gram1()
        if arm == "gram1_flat":
            return g
        sk = self.skeleton()
        ae = self.embs[AGENTLIKE]
        am = np.mean(ae, axis=0) if ae else np.zeros(384)
        disc = float(am @ sk / max(np.linalg.norm(am) * np.linalg.norm(sk),
                                   1e-9))
        return np.concatenate([g, sk, [disc]])


def _cand_scores(st, C, arm, w, b):
    """Vectorized head score for candidate assistant turns C (m, 384):
    only the assistant g0/g1 block (and, for tethered, the discrepancy)
    move; other role blocks and the skeleton are fixed."""
    m = len(C)
    E = np.array(st.embs[AGENTLIKE]) if st.embs[AGENTLIKE] \
        else np.zeros((0, 384))
    n = len(E)
    if n == 0:
        g0 = C
        g1 = np.zeros((m, 384))
    else:
        m1 = n + 1
        ramp = 1.0 - 2.0 * np.arange(m1) / (m1 - 1)
        denom = ramp @ ramp
        g0 = (E.sum(0)[None, :] + C) / m1
        g1 = (ramp[:n] @ E + ramp[n] * C) / denom
    a_block = np.hstack([g0, g1])                      # (m, 768)
    other = {t: _g01(np.array(st.embs[t]) if st.embs[t]
                     else np.zeros((0, 384)))
             for t in TYPES if t != AGENTLIKE}
    blocks = []
    for t in TYPES:
        blocks.append(a_block if t == AGENTLIKE
                      else np.tile(other[t], (m, 1)))
    feat = np.hstack(blocks)
    if arm == "gram1_tethered":
        sk = st.skeleton()
        am = (E.sum(0)[None, :] + C) / (n + 1)
        disc = (am @ sk) / np.maximum(
            np.linalg.norm(am, axis=1) * np.linalg.norm(sk), 1e-9)
        feat = np.hstack([feat, np.tile(sk, (m, 1)), disc[:, None]])
    return feat @ w + b


def _collect(emb, off, meta, svc_vecs, svc_idx, per_conv):
    feats = {"gram1_flat": [], "gram1_tethered": []}
    conv = []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        st = State()
        tc = 0
        seq = per_conv[i]
        for t in range(len(E)):
            sv = None
            if tt[t] == "tool_call" and tc < len(seq):
                sv = svc_vecs[svc_idx[seq[tc]]]
                tc += 1
            st.ingest(E[t].astype(np.float64), tt[t], sv)
            if tt[t] == DECISION:
                feats["gram1_flat"].append(st.feat("gram1_flat"))
                feats["gram1_tethered"].append(st.feat("gram1_tethered"))
                conv.append(i)
    return ({k: np.array(v, np.float32) for k, v in feats.items()},
            np.array(conv))


def _threshold(scores, conv, y):
    grid = np.percentile(scores, np.arange(50, 100))
    for thr in grid:
        al = np.unique(conv[scores >= thr])
        if len(al) and y[al].mean() >= 0.5:
            return thr
    return grid[-1]


def _attack(emb, off, meta, per_conv, svc_vecs, svc_idx, arm, clf, thr,
            pool, fail_convs):
    w, b = clf.coef_[0], float(clf.intercept_[0])
    rng = np.random.default_rng(SEED)
    suppressed = 0
    for i in fail_convs:
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        st = State()
        tc = 0
        seq = per_conv[i]
        alerted = False
        for t in range(len(E)):
            if tt[t] == AGENTLIKE:
                cand = pool[rng.integers(0, len(pool), size=16)]
                best = cand[int(np.argmin(
                    _cand_scores(st, cand, arm, w, b)))]
                st.ingest(best, AGENTLIKE)
            else:
                sv = (svc_vecs[svc_idx[seq[tc]]]
                      if tt[t] == "tool_call" and tc < len(seq) else None)
                if tt[t] == "tool_call" and tc < len(seq):
                    tc += 1
                st.ingest(E[t], tt[t], sv)
            if tt[t] == DECISION and clf.decision_function(
                    st.feat(arm).reshape(1, -1))[0] >= thr:
                alerted = True
        if not alerted:
            suppressed += 1
    return suppressed


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    conversations = json.loads(CONV.read_text())
    svc_vecs, svc_idx, per_conv = _service_embeddings(conversations)

    feats, conv = _collect(emb, off, meta, svc_vecs, svc_idx, per_conv)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        feats["gram1_flat"], sy, conv))
    train_convs = set(conv[tr_idx].tolist())
    fail_test = [i for i in set(conv[te_idx].tolist()) if y[i]]

    pool = []
    for i in train_convs:
        if y[i]:
            continue
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        p = E[tt == AGENTLIKE]
        if len(p):
            pool.append(p)
    pool = np.vstack(pool).astype(np.float64)

    arms = {}
    for arm in ("gram1_flat", "gram1_tethered"):
        X = feats[arm]
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[tr_idx], sy[tr_idx])
        acc = float(roc_auc_score(sy[te_idx],
                                  clf.decision_function(X[te_idx])))
        thr = _threshold(clf.decision_function(X[tr_idx]), conv[tr_idx], y)
        base = sum(1 for i in fail_test
                   if (clf.decision_function(feats[arm][conv == i]) >= thr
                       ).any())
        base_recall = base / max(len(fail_test), 1)
        supp = _attack(emb, off, meta, per_conv, svc_vecs, svc_idx, arm,
                       clf, thr, pool, fail_test)
        n16 = (len(fail_test) - supp) / max(len(fail_test), 1)
        manip = ((base_recall - n16) / base_recall
                 if base_recall > 0 else 0.0)
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "n16_recall": float(n16), "manipulability": float(manip)}

    res = {"dataset": "toolbench", "n_test_failures": len(fail_test),
           "arms": arms}
    save_result("tool_anchored_g1", res, dataset="toolbench",
                params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
