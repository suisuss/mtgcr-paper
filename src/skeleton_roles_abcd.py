"""Round-13B: split-roles restriction + conditional typicality on ABCD
(replication of round 13). Adversary writes the AGENT role; customer and
action roles are fixed. Restriction to the fixed roles is invariant under
the agent-substitution attack. PREREGISTRATION-R13B.md.

CIRCULARITY CAVEAT: ABCD's label is action-derived, so accuracy leans on the
action channel the label is computed from (not clean evidence - ToolBench is
the clean accuracy anchor). The clean result here is the MANIPULABILITY
replication (structural, label-independent).

    python skeleton_roles_abcd.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result
from skeleton_abcd import _slug_embeddings

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
TYPES = ["customer", "agent", "action"]
AGENTLIKE = "agent"
DECISION = "customer"
CONTEXT = ["customer", "action"]
SEED = 42
INVARIANT = {"skeleton", "drop_agent", "drop_agent_skel"}


class State:
    def __init__(self):
        self.sums = {t: np.zeros(384) for t in TYPES}
        self.cnts = {t: 0 for t in TYPES}
        self.skel_sum = np.zeros(384)
        self.skel_cnt = 0

    def ingest(self, e, ttype, slug=None):
        self.sums[ttype] += e
        self.cnts[ttype] += 1
        if ttype == "action" and slug is not None:
            self.skel_sum += slug
            self.skel_cnt += 1

    def m(self, t):
        return self.sums[t] / max(self.cnts[t], 1)

    def skeleton(self):
        return self.skel_sum / max(self.skel_cnt, 1)

    def context(self):
        return np.concatenate([self.m(t) for t in CONTEXT]
                              + [self.skeleton()])

    def atoms(self):
        return {t: self.m(t) for t in TYPES} | {"skeleton": self.skeleton()}


def _feat(a, arm, ridge=None):
    if arm == "flat":
        return np.concatenate([a[t] for t in TYPES])
    if arm == "skeleton":
        return a["skeleton"]
    ctx = np.concatenate([a[t] for t in CONTEXT] + [a["skeleton"]])
    if arm == "drop_agent":
        return np.concatenate([a[t] for t in CONTEXT])
    if arm == "drop_agent_skel":
        return ctx
    pred = ridge.predict(ctx.reshape(1, -1))[0]
    return np.concatenate([ctx, [float(np.linalg.norm(a["agent"] - pred))]])


def _collect(emb, off, meta, slug_vecs, slug_idx):
    atoms = {k: [] for k in TYPES + ["skeleton"]}
    conv = []
    an = meta["action_names"]
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        st = State()
        for t in range(len(E)):
            slug = (slug_vecs[slug_idx[an[i][t]]]
                    if tt[t] == "action" and an[i][t] in slug_idx else None)
            st.ingest(E[t].astype(np.float64), tt[t], slug)
            if tt[t] == DECISION:
                a = st.atoms()
                for k in atoms:
                    atoms[k].append(a[k])
                conv.append(i)
    return {k: np.array(v) for k, v in atoms.items()}, np.array(conv)


def _threshold(scores, conv, y):
    grid = np.percentile(scores, np.arange(50, 100))
    for thr in grid:
        al = np.unique(conv[scores >= thr])
        if len(al) and y[al].mean() >= 0.5:
            return thr
    return grid[-1]


def _cand_scores(st, C, arm, w, b, ridge=None):
    m = len(C)
    n = st.cnts[AGENTLIKE]
    am_new = (st.sums[AGENTLIKE][None, :] + C) / (n + 1)
    if arm == "flat":
        blocks = [np.tile(st.m(t), (m, 1)) if t != AGENTLIKE else am_new
                  for t in TYPES]
        return np.hstack(blocks) @ w + b
    ctx = st.context()
    pred = ridge.predict(ctx.reshape(1, -1))[0]
    res = np.linalg.norm(am_new - pred, axis=1)
    return np.hstack([np.tile(ctx, (m, 1)), res[:, None]]) @ w + b


def _attack(emb, off, meta, slug_vecs, slug_idx, arm, clf, thr, pool,
            fail_convs, ridge=None):
    if arm in INVARIANT:
        return None
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
                best = cand[int(np.argmin(
                    _cand_scores(st, cand, arm, w, b, ridge)))]
                st.ingest(best, AGENTLIKE)
            else:
                slug = (slug_vecs[slug_idx[an[i][t]]]
                        if tt[t] == "action" and an[i][t] in slug_idx
                        else None)
                st.ingest(E[t], tt[t], slug)
            if tt[t] == DECISION:
                feat = _feat(st.atoms(), arm, ridge)
                if clf.decision_function(feat.reshape(1, -1))[0] >= thr:
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

    atoms, conv = _collect(emb, off, meta, slug_vecs, slug_idx)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        atoms["skeleton"], sy, conv))
    ctx_all = np.hstack([atoms[t] for t in CONTEXT] + [atoms["skeleton"]])
    ridge = Ridge(alpha=1.0).fit(ctx_all[tr_idx], atoms["agent"][tr_idx])
    resid = np.linalg.norm(atoms["agent"] - ridge.predict(ctx_all), axis=1)

    def build(arm):
        if arm == "flat":
            return np.hstack([atoms[t] for t in TYPES])
        if arm == "skeleton":
            return atoms["skeleton"]
        if arm == "drop_agent":
            return np.hstack([atoms[t] for t in CONTEXT])
        if arm == "drop_agent_skel":
            return ctx_all
        return np.hstack([ctx_all, resid[:, None]])

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
    for arm in ("flat", "skeleton", "drop_agent", "drop_agent_skel",
                "typicality"):
        X = build(arm)
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[tr_idx], sy[tr_idx])
        acc = float(roc_auc_score(sy[te_idx],
                                  clf.decision_function(X[te_idx])))
        thr = _threshold(clf.decision_function(X[tr_idx]), conv[tr_idx], y)
        base = sum(1 for i in fail_test
                   if (clf.decision_function(X[conv == i]) >= thr).any())
        base_recall = base / max(len(fail_test), 1)
        supp = _attack(emb, off, meta, slug_vecs, slug_idx, arm, clf, thr,
                       pool, fail_test, ridge)
        if supp is None:
            n16, manip = base_recall, 0.0
        else:
            n16 = (len(fail_test) - supp) / max(len(fail_test), 1)
            manip = ((base_recall - n16) / base_recall
                     if base_recall > 0 else 0.0)
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "n16_recall": float(n16), "manipulability": float(manip)}

    res = {"dataset": "abcd", "n_test_failures": len(fail_test), "arms": arms}
    save_result("skeleton_roles_abcd", res, dataset="abcd",
                params={"seed": SEED})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
