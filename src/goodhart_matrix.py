"""Round-6: the predecessor design's rung-1 reward-arm matrix as a
selection-pressure manipulability probe with a verified-label judge.
PREREGISTRATION-R6.md.

Four arms (its minimum discriminating set): floor (dense g0), foil
(TRACE-relational), method (Gram role-split degree-0), governed (Gram +
channel tripwire). Each: frozen head on fold-0 train, alert threshold at
train precision >= 0.5, best-of-n substitution of agentlike turns from the
successful-conversation pool on held-out failures. Reported as
(accuracy, manipulability) pairs per the matched-predictor rule.

Each arm is a FeatureState: incremental running state that ingests turns
(real or substituted) and emits (a) a fixed feature vector for the head at
a decision point, and (b) for the governed arm, tripwire values. This lets
the selection attack re-score every arm under substitution without
recomputing from scratch.

    python goodhart_matrix.py toolbench
    python goodhart_matrix.py abcd
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

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
SEED = 42
NS = [1, 16]
ARMS = ["floor_dense", "foil_trace", "method_gram", "governed_gram"]

CORPUS = {
    "toolbench": {"types": ["user", "assistant", "tool_call",
                            "tool_result"],
                  "decision": "tool_result", "agentlike": "assistant",
                  "userlike": "user"},
    "abcd": {"types": ["customer", "agent", "action"],
             "decision": "customer", "agentlike": "agent",
             "userlike": "customer"},
    "spokenwoz": {"types": ["user", "system"], "decision": "user",
                  "agentlike": "system", "userlike": "user"},
}


class State:
    """Incremental running features for all arms in one pass."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.types = cfg["types"]
        self.sums = {t: np.zeros(384) for t in self.types}
        self.cnts = {t: 0 for t in self.types}
        self.overall_sum = np.zeros(384)
        self.overall_cnt = 0
        self.first = None
        self.prev = None
        self.cos_sum = 0.0
        self.cos_sqsum = 0.0
        self.cos_n = 0

    def ingest(self, e, ttype):
        if ttype in self.sums:
            self.sums[ttype] += e
            self.cnts[ttype] += 1
        self.overall_sum += e
        self.overall_cnt += 1
        if self.first is None:
            self.first = e
        if self.prev is not None:
            c = 1.0 - float(self.prev @ e)
            self.cos_sum += c
            self.cos_sqsum += c * c
            self.cos_n += 1
        self.prev = e

    def _tmean(self, t):
        return self.sums[t] / max(self.cnts[t], 1)

    def dense(self):
        return self.overall_sum / max(self.overall_cnt, 1)

    def gram(self):
        return np.concatenate([self._tmean(t) for t in self.types])

    def trace(self):
        cur = self.overall_sum / max(self.overall_cnt, 1)
        roughness = self.cos_sum / max(self.cos_n, 1)
        drift = (1.0 - float(self.first @ cur)
                 if self.first is not None else 0.0)
        um = self._tmean(self.cfg["userlike"])
        sm = self._tmean(self.cfg["agentlike"])
        gap = 1.0 - float(um @ sm / max(np.linalg.norm(um)
                                        * np.linalg.norm(sm), 1e-9))
        var = (self.cos_sqsum / max(self.cos_n, 1)
               - roughness ** 2)
        vol = float(np.sqrt(max(var, 0.0)))
        return np.array([roughness, drift, gap, vol])

    def tripwire(self):
        um = self._tmean(self.cfg["userlike"])
        sm = self._tmean(self.cfg["agentlike"])
        div = 1.0 - float(um @ sm / max(np.linalg.norm(um)
                                        * np.linalg.norm(sm), 1e-9))
        energy = float(np.linalg.norm(sm))
        return np.array([div, energy])


def _feat(state, arm):
    if arm == "floor_dense":
        return state.dense()
    if arm == "foil_trace":
        return state.trace()
    return state.gram()  # method + governed share the head


def _candidate_scores(st, C, arm, w, b, cfg):
    """Vectorized head score for each candidate C[j] considered as the
    next agentlike turn, from current state st. Mirrors State's feature
    maps exactly, without mutating st."""
    ai = cfg["agentlike"]
    n = len(C)
    if arm == "floor_dense":
        feat = (st.overall_sum[None, :] + C) / (st.overall_cnt + 1)
        return feat @ w + b
    if arm == "method_gram":
        blocks = []
        for t in cfg["types"]:
            if t == ai:
                blocks.append((st.sums[t][None, :] + C) / (st.cnts[t] + 1))
            else:
                blocks.append(np.tile(st._tmean(t), (n, 1)))
        return np.hstack(blocks) @ w + b
    # foil_trace
    cur = (st.overall_sum[None, :] + C) / (st.overall_cnt + 1)
    if st.prev is not None:
        cnew = 1.0 - C @ st.prev
        rough = (st.cos_sum + cnew) / (st.cos_n + 1)
        sqnew = (st.cos_sqsum + cnew ** 2) / (st.cos_n + 1)
    else:
        rough = np.zeros(n)
        sqnew = np.zeros(n)
    drift = (1.0 - cur @ st.first if st.first is not None
             else np.zeros(n))
    um = st._tmean(cfg["userlike"])
    sm = (st.sums[ai][None, :] + C) / (st.cnts[ai] + 1)
    gap = 1.0 - (sm @ um) / np.maximum(
        np.linalg.norm(sm, axis=1) * np.linalg.norm(um), 1e-9)
    vol = np.sqrt(np.maximum(sqnew - rough ** 2, 0.0))
    feat = np.stack([rough, drift, gap, vol], axis=1)
    return feat @ w + b


def _collect(emb, off, meta, cfg):
    """Per-decision-point features for all arms + conv id + label."""
    rows = {a: [] for a in ("floor_dense", "foil_trace", "method_gram")}
    trip = []
    conv = []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        st = State(cfg)
        for t in range(len(E)):
            st.ingest(E[t].astype(np.float64), tt[t])
            if tt[t] == cfg["decision"]:
                rows["floor_dense"].append(st.dense())
                rows["foil_trace"].append(st.trace())
                rows["method_gram"].append(st.gram())
                trip.append(st.tripwire())
                conv.append(i)
    return ({a: np.array(v) for a, v in rows.items()},
            np.array(trip), np.array(conv))


def _threshold(scores, conv, y):
    grid = np.percentile(scores, np.arange(50, 100))
    for thr in grid:
        al = np.unique(conv[scores >= thr])
        if len(al) and y[al].mean() >= 0.5:
            return thr
    return grid[-1]


def _attack(emb, off, meta, cfg, arm, clf, thr, pool, rng,
            fail_convs, n, trip_env=None):
    """Returns (n suppressed, n suppressed-and-uncaught) over fail_convs."""
    ai = cfg["agentlike"]
    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    suppressed = uncaught = 0
    for i in fail_convs:
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        st = State(cfg)
        alerted = False
        caught_at_supp = False
        for t in range(len(E)):
            if tt[t] == ai:
                cand = pool[rng.integers(0, len(pool), size=n)]
                if n > 1:
                    best = cand[int(np.argmin(
                        _candidate_scores(st, cand, arm, w, b, cfg)))]
                else:
                    best = cand[0]
                st.ingest(best, ai)
            else:
                st.ingest(E[t], tt[t])
            if tt[t] == cfg["decision"]:
                score = _feat(st, arm) @ w + b
                if score >= thr:
                    alerted = True
                elif trip_env is not None:
                    tv = st.tripwire()
                    if np.any(tv < trip_env[0]) or np.any(tv > trip_env[1]):
                        caught_at_supp = True
        if not alerted:
            suppressed += 1
            if trip_env is None or not caught_at_supp:
                uncaught += 1
    return suppressed, uncaught




def run(dataset):
    cfg = CORPUS[dataset]
    z = np.load(EMB / f"{dataset}_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / f"{dataset}_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    gid = np.array(meta.get("group_id", np.arange(len(y))))

    feats, trip, conv = _collect(emb, off, meta, cfg)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        feats["method_gram"], sy, gid[conv]))
    train_convs = set(conv[tr_idx].tolist())
    test_convs = set(conv[te_idx].tolist())
    fail_test = [i for i in test_convs if y[i]]

    # candidate pool: agentlike turns of successful training conversations
    pool = []
    for i in train_convs:
        if y[i]:
            continue
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        p = E[tt == cfg["agentlike"]]
        if len(p):
            pool.append(p)
    pool = np.vstack(pool).astype(np.float64)

    trip_env = (np.percentile(trip[tr_idx], 0.5, axis=0),
                np.percentile(trip[tr_idx], 99.5, axis=0))
    rng = np.random.default_rng(SEED)

    results = {}
    for arm in ARMS:
        head_arm = "method_gram" if arm == "governed_gram" else arm
        X = feats[head_arm]
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000)
        clf.fit(X[tr_idx], sy[tr_idx])
        acc = float(roc_auc_score(sy[te_idx],
                                  clf.decision_function(X[te_idx])))
        thr = _threshold(clf.decision_function(X[tr_idx]),
                         conv[tr_idx], y)
        # base recall on held-out failures
        base = 0
        for i in fail_test:
            m = conv == i
            if (clf.decision_function(X[m]) >= thr).any():
                base += 1
        base_recall = base / max(len(fail_test), 1)
        env = trip_env if arm == "governed_gram" else None
        supp16, unc16 = _attack(emb, off, meta, cfg, head_arm, clf, thr,
                                pool, np.random.default_rng(SEED),
                                fail_test, 16, env)
        supp1, _ = _attack(emb, off, meta, cfg, head_arm, clf, thr, pool,
                           np.random.default_rng(SEED), fail_test, 1, None)
        hacked16 = (len(fail_test) - unc16) / max(len(fail_test), 1) \
            if arm == "governed_gram" else \
            (len(fail_test) - supp16) / max(len(fail_test), 1)
        drift1 = (len(fail_test) - supp1) / max(len(fail_test), 1)
        manip = ((base_recall - hacked16) / base_recall
                 if base_recall > 0 else 0.0)
        results[arm] = {
            "accuracy_auc": acc, "threshold": float(thr),
            "base_recall": float(base_recall),
            "recall_n1_drift": float(drift1),
            "recall_n16": float(hacked16),
            "manipulability": float(manip),
            "suppressed_n16": int(supp16),
            "caught_by_tripwire": int(supp16 - unc16)
            if arm == "governed_gram" else None}

    res = {"dataset": dataset, "n_test_failures": len(fail_test),
           "pool_size": int(len(pool)), "arms": results}
    save_result(f"goodhart_matrix_{dataset}", res, dataset=dataset,
                params={"seed": SEED, "ns": NS})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
