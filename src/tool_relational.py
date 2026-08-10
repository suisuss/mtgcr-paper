"""Round-12: relational binding of content (g0) against the tool grounding.
Rounds 9-10 showed ADDITIVE tethering does not inherit the skeleton's
robustness (the head keeps weight on the free NL block). This tests whether
letting the NL content enter ONLY as a quantity measured against the
non-spoofable tool skeleton - never as free absolute position - reduces
manipulability while recovering accuracy above the bare skeleton.
PREREGISTRATION-R12.md. (#92 "binding by parameterization", instantiated
against the tool grounding and tested under the selection attack.)

Arms (ToolBench, round-9 attack: best-of-16 substitution of assistant turns,
fold-0 train head + threshold, seed 42):
  skeleton    - skeleton mean only (invariant anchor; reproduces 0.706/0.00)
  relational  - [skeleton, cos(assistant g0, skel), cos(tool_result, skel),
                cos(user, skel)] - NO raw assistant g0
  gated       - [skeleton, gate * assistant g0], gate = clip(cos(assistant,
                skel), 0, 1): ungrounded NL is zeroed
Compared against round-9 committed: flat g0 0.788/0.50; tethered 0.788/0.50.

    python tool_relational.py
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
ARMS = ["skeleton", "relational", "gated"]


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


class State:
    def __init__(self):
        self.sums = {t: np.zeros(384) for t in TYPES}
        self.cnts = {t: 0 for t in TYPES}
        self.skel_sum = np.zeros(384)
        self.skel_cnt = 0

    def ingest(self, e, ttype, svc=None):
        self.sums[ttype] += e
        self.cnts[ttype] += 1
        if ttype == "tool_call" and svc is not None:
            self.skel_sum += svc
            self.skel_cnt += 1

    def _m(self, t):
        return self.sums[t] / max(self.cnts[t], 1)

    def skeleton(self):
        return self.skel_sum / max(self.skel_cnt, 1)

    def feat(self, arm):
        sk = self.skeleton()
        if arm == "skeleton":
            return sk
        am = self._m(AGENTLIKE)
        if arm == "relational":
            return np.concatenate([sk, [_cos(am, sk),
                                        _cos(self._m("tool_result"), sk),
                                        _cos(self._m("user"), sk)]])
        gate = float(np.clip(_cos(am, sk), 0.0, 1.0))
        return np.concatenate([sk, gate * am])


def _cand_scores(st, C, arm, w, b):
    """Score candidate assistant turns C (m,384): only the assistant mean
    (and features derived from it) move; skeleton and other role means fixed."""
    m = len(C)
    n = st.cnts[AGENTLIKE]
    am_new = (st.sums[AGENTLIKE][None, :] + C) / (n + 1)     # (m,384)
    sk = st.skeleton()
    nsk = np.linalg.norm(sk)
    cos_am = (am_new @ sk) / np.maximum(
        np.linalg.norm(am_new, axis=1) * nsk, 1e-9)          # (m,)
    if arm == "relational":
        cols = np.stack([cos_am,
                         np.full(m, _cos(st._m("tool_result"), sk)),
                         np.full(m, _cos(st._m("user"), sk))], axis=1)
        feat = np.hstack([np.tile(sk, (m, 1)), cols])
    else:  # gated
        gate = np.clip(cos_am, 0.0, 1.0)[:, None]
        feat = np.hstack([np.tile(sk, (m, 1)), gate * am_new])
    return feat @ w + b


def _collect(emb, off, meta, svc_vecs, svc_idx, per_conv):
    feats = {a: [] for a in ARMS}
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
                for a in ARMS:
                    feats[a].append(st.feat(a))
                conv.append(i)
    return {a: np.array(v) for a, v in feats.items()}, np.array(conv)


def _threshold(scores, conv, y):
    grid = np.percentile(scores, np.arange(50, 100))
    for thr in grid:
        al = np.unique(conv[scores >= thr])
        if len(al) and y[al].mean() >= 0.5:
            return thr
    return grid[-1]


def _attack(emb, off, meta, per_conv, svc_vecs, svc_idx, arm, clf, thr,
            pool, fail_convs):
    if arm == "skeleton":
        return None  # invariant under assistant substitution
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
                sv = None
                if tt[t] == "tool_call" and tc < len(seq):
                    sv = svc_vecs[svc_idx[seq[tc]]]
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
        feats["relational"], sy, conv))
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
    for arm in ARMS:
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
        if supp is None:
            n16, manip = base_recall, 0.0
        else:
            n16 = (len(fail_test) - supp) / max(len(fail_test), 1)
            manip = ((base_recall - n16) / base_recall
                     if base_recall > 0 else 0.0)
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "n16_recall": float(n16), "manipulability": float(manip)}

    res = {"dataset": "toolbench", "n_test_failures": len(fail_test),
           "arms": arms}
    save_result("tool_relational", res, dataset="toolbench",
                params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
