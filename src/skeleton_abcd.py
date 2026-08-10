"""Round-10: skeleton restriction on ABCD (the most-gameable corpus).
Restricts the representation to the non-spoofable canonical AGENT-ACTION
channel and measures accuracy + manipulability. PREREGISTRATION-R10.md.

CIRCULARITY CAVEAT: ABCD's label (TRACER rule) is derived from the action
sequence, so skeleton accuracy is an upper bound inflated by label leakage;
the count-only arm (D) quantifies it, and the clean accuracy anchor is
ToolBench round-9 B=0.706. The manipulability result is clean (structural).

Arms: A flat NL (round-6 head), B skeleton (action-slug embeddings),
C tethered, D count-only. Attack substitutes agent turns; actions are
non-substitutable, so B and D are invariant.

    python skeleton_abcd.py
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
TYPES = ["customer", "agent", "action"]
AGENTLIKE = "agent"
DECISION = "customer"
SEED = 42


def _slug_embeddings(action_names):
    vocab = sorted({a for conv in action_names for a in conv if a})
    ckpt = Checkpoint("skeleton_abcd", script=__file__)
    cached = ckpt.load("slugs")
    if cached is not None and list(cached["names"]) == vocab:
        return cached["vecs"], {v: k for k, v in enumerate(vocab)}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode([v.replace("-", " ") for v in vocab],
                        batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)
    ckpt.save("slugs", {"names": np.array(vocab), "vecs": vecs})
    return vecs, {v: k for k, v in enumerate(vocab)}


class State:
    def __init__(self):
        self.sums = {t: np.zeros(384) for t in TYPES}
        self.cnts = {t: 0 for t in TYPES}
        self.skel_sum = np.zeros(384)
        self.skel_cnt = 0

    def ingest(self, e, ttype, slug_vec=None):
        self.sums[ttype] += e
        self.cnts[ttype] += 1
        if ttype == "action" and slug_vec is not None:
            self.skel_sum += slug_vec
            self.skel_cnt += 1

    def _m(self, t):
        return self.sums[t] / max(self.cnts[t], 1)

    def skeleton(self):
        return self.skel_sum / max(self.skel_cnt, 1)

    def feat(self, arm):
        if arm == "B_skeleton":
            return self.skeleton()
        if arm == "D_count":
            return np.array([float(self.skel_cnt)])
        flat = np.concatenate([self._m(t) for t in TYPES])
        if arm == "A_flat":
            return flat
        sk = self.skeleton()
        am = self._m(AGENTLIKE)
        disc = float(am @ sk / max(np.linalg.norm(am) * np.linalg.norm(sk),
                                   1e-9))
        return np.concatenate([flat, sk, [disc]])


def _collect(emb, off, meta, slug_vecs, slug_idx):
    arms = ("A_flat", "B_skeleton", "C_tethered", "D_count")
    feats = {a: [] for a in arms}
    conv = []
    action_names = meta["action_names"]
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        an = action_names[i]
        st = State()
        for t in range(len(E)):
            sv = None
            if tt[t] == "action" and an[t] in slug_idx:
                sv = slug_vecs[slug_idx[an[t]]]
            st.ingest(E[t].astype(np.float64), tt[t], sv)
            if tt[t] == DECISION:
                for a in arms:
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
    n = len(C)
    am_new = (st.sums[AGENTLIKE][None, :] + C) / (st.cnts[AGENTLIKE] + 1)
    blocks = [np.tile(st._m(t), (n, 1)) if t != AGENTLIKE else am_new
              for t in TYPES]
    if arm == "A_flat":
        return np.hstack(blocks) @ w + b
    sk = st.skeleton()
    disc = (am_new @ sk) / np.maximum(
        np.linalg.norm(am_new, axis=1) * np.linalg.norm(sk), 1e-9)
    return np.hstack(blocks + [np.tile(sk, (n, 1)), disc[:, None]]) @ w + b


def _attack(emb, off, meta, slug_vecs, slug_idx, arm, clf, thr, pool,
            fail_convs):
    if arm in ("B_skeleton", "D_count"):
        return None  # invariant under agent substitution
    w, b = clf.coef_[0], float(clf.intercept_[0])
    rng = np.random.default_rng(SEED)
    an = meta["action_names"]
    suppressed = 0
    for i in fail_convs:
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        st = State()
        alerted = False
        for t in range(len(E)):
            if tt[t] == AGENTLIKE:
                cand = pool[rng.integers(0, len(pool), size=16)]
                best = cand[int(np.argmin(_cand_scores(st, cand, arm,
                                                       w, b)))]
                st.ingest(best, AGENTLIKE)
            else:
                sv = (slug_vecs[slug_idx[an[i][t]]]
                      if tt[t] == "action" and an[i][t] in slug_idx
                      else None)
                st.ingest(E[t], tt[t], sv)
            if tt[t] == DECISION and clf.decision_function(
                    st.feat(arm).reshape(1, -1))[0] >= thr:
                alerted = True
        if not alerted:
            suppressed += 1
    return suppressed


def run():
    z = np.load(EMB / "abcd_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "abcd_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    slug_vecs, slug_idx = _slug_embeddings(meta["action_names"])

    feats, conv = _collect(emb, off, meta, slug_vecs, slug_idx)
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
    for arm in ("A_flat", "B_skeleton", "C_tethered", "D_count"):
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
        supp = _attack(emb, off, meta, slug_vecs, slug_idx, arm, clf, thr,
                       pool, fail_test)
        if supp is None:
            n16, manip = base_recall, 0.0
        else:
            n16 = (len(fail_test) - supp) / max(len(fail_test), 1)
            manip = ((base_recall - n16) / base_recall
                     if base_recall > 0 else 0.0)
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "n16_recall": float(n16), "manipulability": float(manip)}

    res = {"dataset": "abcd", "n_test_failures": len(fail_test),
           "n_slugs": len(slug_idx), "arms": arms}
    save_result("skeleton_abcd", res, dataset="abcd", params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
