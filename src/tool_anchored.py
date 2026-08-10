"""Round-9: tool-anchored representation. Tests whether anchoring the
trajectory to the non-spoofable tool-call IDENTITY reduces
selection-manipulability vs a purely-NL head. PREREGISTRATION-R9.md.

Three arms scored at tool_result decision points, round-6 attack protocol:
  A flat NL (four per-role running means)  - fully spoofable
  B skeleton (mean of called-tool service-name embeddings) - invariant
  C tethered (A + skeleton + discrepancy)  - dense NL tethered to identity
The attack substitutes assistant (thought) turns only; tool calls cannot be
substituted, so the skeleton is invariant by construction.

    python tool_anchored.py
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
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
TYPES = ["user", "assistant", "tool_call", "tool_result"]
AGENTLIKE = "assistant"
DECISION = "tool_result"
SEED = 42


def _service_embeddings(conversations):
    """Embed each unique tool service name once; return {name: vec} and the
    per-conversation ordered service list aligned to turns."""
    per_conv = []
    vocab = set()
    for c in conversations:
        seq = []
        for t in c["turns"]:
            if t["turn_type"] == "tool_call":
                s = t["metadata"].get("service", "unknown")
                seq.append(s)
                vocab.add(s)
        per_conv.append(seq)
    vocab = sorted(vocab)
    ckpt = Checkpoint("tool_anchored", script=__file__)
    cached = ckpt.load("services")
    if cached is not None and list(cached["names"]) == vocab:
        vecs = cached["vecs"]
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vecs = model.encode([v.replace("_", " ") for v in vocab],
                            batch_size=256, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
        ckpt.save("services", {"names": np.array(vocab), "vecs": vecs})
    idx = {v: k for k, v in enumerate(vocab)}
    return vecs, idx, per_conv


class State:
    def __init__(self):
        self.sums = {t: np.zeros(384) for t in TYPES}
        self.cnts = {t: 0 for t in TYPES}
        self.skel_sum = np.zeros(384)
        self.skel_cnt = 0

    def ingest(self, e, ttype, svc_vec=None):
        self.sums[ttype] += e
        self.cnts[ttype] += 1
        if ttype == "tool_call" and svc_vec is not None:
            self.skel_sum += svc_vec
            self.skel_cnt += 1

    def _m(self, t):
        return self.sums[t] / max(self.cnts[t], 1)

    def skeleton(self):
        return self.skel_sum / max(self.skel_cnt, 1)

    def feat(self, arm):
        if arm == "B_skeleton":
            return self.skeleton()
        flat = np.concatenate([self._m(t) for t in TYPES])
        if arm == "A_flat":
            return flat
        sk = self.skeleton()
        am = self._m(AGENTLIKE)
        disc = float(am @ sk / max(np.linalg.norm(am) * np.linalg.norm(sk),
                                   1e-9))
        return np.concatenate([flat, sk, [disc]])


def _collect(emb, off, meta, svc_vecs, svc_idx, per_conv):
    feats = {a: [] for a in ("A_flat", "B_skeleton", "C_tethered")}
    conv = []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        st = State()
        tc_ptr = 0
        seq = per_conv[i]
        for t in range(len(E)):
            sv = None
            if tt[t] == "tool_call" and tc_ptr < len(seq):
                sv = svc_vecs[svc_idx[seq[tc_ptr]]]
                tc_ptr += 1
            st.ingest(E[t].astype(np.float64), tt[t], sv)
            if tt[t] == DECISION:
                for a in feats:
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


def _cand_scores(st, C, arm, w, b):
    """Head score for each candidate assistant turn C[j]; only the
    assistant block (and C's discrepancy) move, skeleton is fixed."""
    n = len(C)
    am_new = (st.sums[AGENTLIKE][None, :] + C) / (st.cnts[AGENTLIKE] + 1)
    ai = TYPES.index(AGENTLIKE)
    if arm == "A_flat":
        blocks = [np.tile(st._m(t), (n, 1)) if t != AGENTLIKE else am_new
                  for t in TYPES]
        return np.hstack(blocks) @ w + b
    # C_tethered
    blocks = [np.tile(st._m(t), (n, 1)) if t != AGENTLIKE else am_new
              for t in TYPES]
    sk = st.skeleton()
    disc = (am_new @ sk) / np.maximum(
        np.linalg.norm(am_new, axis=1) * np.linalg.norm(sk), 1e-9)
    feat = np.hstack(blocks + [np.tile(sk, (n, 1)), disc[:, None]])
    return feat @ w + b


def _attack(emb, off, meta, per_conv, svc_vecs, svc_idx, arm, clf, thr,
            pool, fail_convs):
    if arm == "B_skeleton":
        return None  # invariant; n16 recall == base recall
    w, b = clf.coef_[0], float(clf.intercept_[0])
    rng = np.random.default_rng(SEED)
    suppressed = 0
    for i in fail_convs:
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        st = State()
        tc_ptr = 0
        seq = per_conv[i]
        alerted = False
        for t in range(len(E)):
            if tt[t] == AGENTLIKE:
                cand = pool[rng.integers(0, len(pool), size=16)]
                best = cand[int(np.argmin(_cand_scores(st, cand, arm,
                                                       w, b)))]
                st.ingest(best, AGENTLIKE)
            else:
                sv = None
                if tt[t] == "tool_call" and tc_ptr < len(seq):
                    sv = svc_vecs[svc_idx[seq[tc_ptr]]]
                    tc_ptr += 1
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
        feats["A_flat"], sy, conv))
    test_convs = set(conv[te_idx].tolist())
    train_convs = set(conv[tr_idx].tolist())
    fail_test = [i for i in test_convs if y[i]]

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
    for arm in ("A_flat", "B_skeleton", "C_tethered"):
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
        if supp is None:  # invariant
            n16_recall, manip = base_recall, 0.0
        else:
            n16_recall = (len(fail_test) - supp) / max(len(fail_test), 1)
            manip = ((base_recall - n16_recall) / base_recall
                     if base_recall > 0 else 0.0)
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "n16_recall": float(n16_recall),
                     "manipulability": float(manip)}

    res = {"dataset": "toolbench", "n_test_failures": len(fail_test),
           "n_services": len(svc_idx), "arms": arms}
    save_result("tool_anchored", res, dataset="toolbench",
                params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
