"""Round-11 Arm 2: the policy-choice attack with real-label accounting
(ABCD). ABCD's TRACER label is a deterministic function of the action
sequence, so we can compute the real outcome of a GAMED action sequence -
the one corpus where "did gaming the detector actually change the outcome"
is answerable. PREREGISTRATION-R11.md.

Adversary: replace a failing conversation's actions with the best of 16 real
successful-conversation action sequences (chosen to fool the round-10
skeleton detector). Measure detector-fooled (a), real success by TRACER (b),
and slack = fooled-but-not-succeeded. Run for same-subflow and any-subflow
substitution pools.

    python policy_choice_abcd.py
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from lib.campaign import Checkpoint, save_result
from abcd_preprocess import _subflow_map, _ordered_coverage
from skeleton_abcd import _slug_embeddings, State, DECISION, AGENTLIKE

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
RAW = REPO / "data" / "raw" / "abcd"
SEED = 42
N = 16


def _required_by_subflow():
    with gzip.open(RAW / "abcd_v1.1.json.gz", "rt") as f:
        data = json.load(f)
    convs = ([c for s in data.values() for c in s]
             if isinstance(data, dict) else data)
    guidelines = json.loads((RAW / "guidelines.json").read_text())
    by_flow = {}
    for c in convs:
        by_flow.setdefault(c["scenario"]["flow"], set()).add(
            c["scenario"]["subflow"])
    req = _subflow_map(guidelines, by_flow)
    # collapse to subflow -> required actions (flow is recoverable from meta)
    out = {}
    for (flow, sub), acts in req.items():
        out[(flow, sub)] = acts
    return out


def _tracer_success(flow, subflow, actions, required):
    exp = required.get((flow, subflow))
    if not exp:
        return None
    cov = _ordered_coverage(exp, actions)
    return cov is not None and cov >= 0.8


def _skeleton_feat_for_actions(action_slugs, slug_vecs, slug_idx):
    """Skeleton feature (mean action-slug embedding) for a bare action
    list - matches State.skeleton() at the final decision point."""
    vecs = [slug_vecs[slug_idx[a]] for a in action_slugs if a in slug_idx]
    if not vecs:
        return np.zeros(384)
    return np.mean(vecs, axis=0)


def run():
    z = np.load(EMB / "abcd_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "abcd_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    flow, subflow = meta["flow"], meta["subflow"]
    action_names = meta["action_names"]
    slug_vecs, slug_idx = _slug_embeddings(action_names)
    required = _required_by_subflow()

    # per-conversation action slug list (order preserved)
    actions_of = [[a for a in an if a] for an in action_names]

    # rebuild the round-10 skeleton detector features at decision points
    feats, conv = [], []
    for i in range(len(off) - 1):
        E = emb[off[i]:off[i + 1]]
        tt = meta["turn_types"][i]
        an = action_names[i]
        st = State()
        for t in range(len(E)):
            sv = (slug_vecs[slug_idx[an[t]]]
                  if tt[t] == "action" and an[t] in slug_idx else None)
            st.ingest(E[t].astype(np.float64), tt[t], sv)
            if tt[t] == DECISION:
                feats.append(st.feat("B_skeleton"))
                conv.append(i)
    feats, conv = np.array(feats), np.array(conv)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(feats, sy, conv))
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000).fit(feats[tr_idx], sy[tr_idx])
    # threshold at train precision >= 0.5
    tr_scores = clf.decision_function(feats[tr_idx])
    grid = np.percentile(tr_scores, np.arange(50, 100))
    thr = grid[-1]
    for g in grid:
        al = np.unique(conv[tr_idx][tr_scores >= g])
        if len(al) and y[al].mean() >= 0.5:
            thr = g
            break

    train_convs = set(conv[tr_idx].tolist())
    test_convs = set(conv[te_idx].tolist())
    fail_test = [i for i in test_convs if y[i]]

    # success pools of ACTION SEQUENCES from training conversations
    pool_any = [actions_of[i] for i in train_convs
                if not y[i] and actions_of[i]]
    pool_by_sub = {}
    for i in train_convs:
        if not y[i] and actions_of[i]:
            pool_by_sub.setdefault(subflow[i], []).append(actions_of[i])

    rng = np.random.default_rng(SEED)

    def attack(pool_fn):
        fooled = succeeded = fooled_not_succ = evaluable = 0
        for i in fail_test:
            pool = pool_fn(i)
            if not pool:
                continue
            evaluable += 1
            cand = [pool[j] for j in rng.integers(0, len(pool),
                                                  size=min(N, len(pool)))]
            scored = [(clf.decision_function(
                _skeleton_feat_for_actions(a, slug_vecs, slug_idx
                                           ).reshape(1, -1))[0], a)
                      for a in cand]
            _, best = min(scored, key=lambda s: s[0])
            gamed_score = clf.decision_function(
                _skeleton_feat_for_actions(best, slug_vecs, slug_idx
                                           ).reshape(1, -1))[0]
            is_fooled = gamed_score < thr
            real = _tracer_success(flow[i], subflow[i], best, required)
            if is_fooled:
                fooled += 1
                if real:
                    succeeded += 1
                elif real is False:
                    fooled_not_succ += 1
        return {"evaluable": evaluable, "fooled": fooled,
                "fooled_and_succeeded": succeeded,
                "slack_fooled_not_succeeded": fooled_not_succ,
                "fooled_rate": fooled / max(evaluable, 1),
                "slack_rate": fooled_not_succ / max(evaluable, 1)}

    res = {"dataset": "abcd", "n_test_failures": len(fail_test),
           "threshold": float(thr),
           "same_subflow": attack(
               lambda i: pool_by_sub.get(subflow[i], [])),
           "any_subflow": attack(lambda i: pool_any)}
    save_result("policy_choice_abcd", res, dataset="abcd",
                params={"seed": SEED, "n": N})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
