"""Round-18: adversary-strength sweep to the selection ceiling.

The paper's manipulability numbers use a best-of-16 selection adversary
(draw 16 real pool turns per agentlike turn, substitute the argmin of the
head score; seed 42). This stage sweeps that single budget parameter `n`
from the published 16 up to the class ceiling - EXHAUSTIVE argmin over the
ENTIRE fold-train success-agentlike pool - the strongest adversary
"selection over real turns" can ever be. It reuses each round-13 setup
verbatim (imported helpers; no existing script or artifact is modified), so
the n=16 point nests the published number exactly (guarded). Three cells,
matching the paper's robustness surface (Table 7): ToolBench/MiniLM
(primary), ABCD/MiniLM, ToolBench/BGE-large. PREREGISTRATION-R18.md.

Attackable arms only: `flat` (naive head) and `typicality` (conditional-
typicality residual). The invariant arms (skeleton / drop_written) have
manipulability 0 for every n BY CONSTRUCTION - the agentlike block is not in
the feature - so no budget can move them; asserted, never swept.

    python adversary_sweep.py toolbench
    python adversary_sweep.py abcd
    python adversary_sweep.py bge
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

# Budget grid: 1 (drift control), 16 (published anchor), 64, 256, and the
# exhaustive ceiling (argmin over the whole pool, deterministic).
BUDGETS = [1, 16, 64, 256, "exhaustive"]
SWEPT_ARMS = ("flat", "typicality")
GUARD_TOL = 0.02  # n=16 must reproduce the published manip within this
PUBLISHED_N16 = {
    "toolbench": {"flat": 0.5, "typicality": -0.0969529085872576},
    "abcd": {"flat": 0.8, "typicality": 0.0},
    "bge": {"flat": 0.4476439790575916, "typicality": -0.039682539682539736},
}


def _attack_budget(ad, emb, off, meta, arm, clf, thr, pool, fail_convs,
                   ridge, budget):
    """Round-13 attack loop, generalized over candidate budget and corpus.

    `ad` is a per-corpus adapter binding the imported round-13 helpers
    (State / feat / cand_scores) and a non-agentlike ingest closure, so the
    math is the source scripts' own. At budget=16 the rng draw sequence is
    identical to the published attack, so the operating point nests. For
    "exhaustive": evaluate the whole pool (no rng draw), global argmin - the
    selection ceiling. Returns the count of suppressed failure alerts.
    """
    AG, DEC = ad["AGENTLIKE"], ad["DECISION"]
    w, b = clf.coef_[0], float(clf.intercept_[0])
    rng = np.random.default_rng(ad["SEED"])
    suppressed = 0
    for i in fail_convs:
        E = emb[off[i]:off[i + 1]].astype(np.float64)
        tt = meta["turn_types"][i]
        st = ad["make_state"]()
        ingest_nonagent = ad["nonagent"](i)
        alerted = False
        for t in range(len(E)):
            if tt[t] == AG:
                if budget == "exhaustive":
                    cand = pool
                else:
                    cand = pool[rng.integers(0, len(pool), size=budget)]
                best = cand[int(np.argmin(
                    ad["cand_scores"](st, cand, arm, w, b, ridge)))]
                st.ingest(best, AG)
            else:
                ingest_nonagent(st, E[t], tt[t], t)
            if tt[t] == DEC:
                feat = ad["feat"](st.atoms(), arm, ridge)
                if clf.decision_function(feat.reshape(1, -1))[0] >= thr:
                    alerted = True
        if not alerted:
            suppressed += 1
    return suppressed


def _pool(emb, off, meta, y, train_convs, agentlike):
    p = []
    for i in train_convs:
        if y[i]:
            continue
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        q = E[tt == agentlike]
        if len(q):
            p.append(q)
    return np.vstack(p).astype(np.float64)


def _setup_toolbench():
    import skeleton_roles as mod
    from tool_anchored import _service_embeddings

    z = np.load(mod.EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((mod.EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    svc_vecs, svc_idx, per_conv = _service_embeddings(
        json.loads(mod.CONV.read_text()))

    atoms, conv = mod._collect(emb, off, meta, svc_vecs, svc_idx, per_conv)
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        atoms["skeleton"], y[conv], conv))
    ctx_all = np.hstack([atoms[t] for t in mod.CONTEXT] + [atoms["skeleton"]])
    ridge = Ridge(alpha=1.0).fit(ctx_all[tr_idx], atoms["assistant"][tr_idx])
    resid = np.linalg.norm(atoms["assistant"] - ridge.predict(ctx_all), axis=1)

    def build(arm):
        return (np.hstack([atoms[t] for t in mod.TYPES]) if arm == "flat"
                else np.hstack([ctx_all, resid[:, None]]))

    def nonagent(i):
        seq, tc = per_conv[i], [0]
        def ing(st, e, ttype, t):
            sv = None
            if ttype == "tool_call" and tc[0] < len(seq):
                sv = svc_vecs[svc_idx[seq[tc[0]]]]
                tc[0] += 1
            st.ingest(e, ttype, sv)
        return ing

    ad = {"AGENTLIKE": mod.AGENTLIKE, "DECISION": mod.DECISION,
          "SEED": mod.SEED, "make_state": mod.State,
          "feat": mod._feat_from_atoms, "cand_scores": mod._cand_scores,
          "nonagent": nonagent}
    return _pack(mod, emb, off, meta, y, conv, tr_idx, te_idx, build, ridge,
                 _pool(emb, off, meta, y, set(conv[tr_idx].tolist()),
                       mod.AGENTLIKE), ad)


def _setup_abcd():
    import skeleton_roles_abcd as mod
    from skeleton_abcd import _slug_embeddings

    z = np.load(mod.EMB / "abcd_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((mod.EMB / "abcd_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    slug_vecs, slug_idx = _slug_embeddings(meta["action_names"])
    an = meta["action_names"]

    atoms, conv = mod._collect(emb, off, meta, slug_vecs, slug_idx)
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        atoms["skeleton"], y[conv], conv))
    ctx_all = np.hstack([atoms[t] for t in mod.CONTEXT] + [atoms["skeleton"]])
    ridge = Ridge(alpha=1.0).fit(ctx_all[tr_idx], atoms["agent"][tr_idx])
    resid = np.linalg.norm(atoms["agent"] - ridge.predict(ctx_all), axis=1)

    def build(arm):
        return (np.hstack([atoms[t] for t in mod.TYPES]) if arm == "flat"
                else np.hstack([ctx_all, resid[:, None]]))

    def nonagent(i):
        names = an[i]
        def ing(st, e, ttype, t):
            slug = (slug_vecs[slug_idx[names[t]]]
                    if ttype == "action" and names[t] in slug_idx else None)
            st.ingest(e, ttype, slug)
        return ing

    ad = {"AGENTLIKE": mod.AGENTLIKE, "DECISION": mod.DECISION,
          "SEED": mod.SEED, "make_state": mod.State,
          "feat": mod._feat, "cand_scores": mod._cand_scores,
          "nonagent": nonagent}
    return _pack(mod, emb, off, meta, y, conv, tr_idx, te_idx, build, ridge,
                 _pool(emb, off, meta, y, set(conv[tr_idx].tolist()),
                       mod.AGENTLIKE), ad)


def _setup_bge():
    import skeleton_roles_bge as mod

    emb, off, per_conv, svec, sidx = mod._encode()
    dim = emb.shape[1]
    meta = json.loads((mod.EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)

    atoms, conv = mod._collect(emb, off, meta, svec, sidx, per_conv, dim)
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        atoms["skeleton"], y[conv], conv))
    ctx_all = np.hstack([atoms[t] for t in mod.CONTEXT] + [atoms["skeleton"]])
    ridge = Ridge(alpha=1.0).fit(ctx_all[tr_idx], atoms["assistant"][tr_idx])
    resid = np.linalg.norm(atoms["assistant"] - ridge.predict(ctx_all), axis=1)

    def build(arm):
        return (np.hstack([atoms[t] for t in mod.TYPES]) if arm == "flat"
                else np.hstack([ctx_all, resid[:, None]]))

    def nonagent(i):
        seq, tc = per_conv[i], [0]
        def ing(st, e, ttype, t):
            sv = None
            if ttype == "tool_call" and tc[0] < len(seq):
                sv = svec[sidx[seq[tc[0]]]]
                tc[0] += 1
            st.ingest(e, ttype, sv)
        return ing

    ad = {"AGENTLIKE": mod.AGENTLIKE, "DECISION": mod.DECISION,
          "SEED": mod.SEED, "make_state": lambda: mod.State(dim),
          "feat": mod._feat, "cand_scores": mod._cand_scores,
          "nonagent": nonagent}
    return _pack(mod, emb, off, meta, y, conv, tr_idx, te_idx, build, ridge,
                 _pool(emb, off, meta, y, set(conv[tr_idx].tolist()),
                       mod.AGENTLIKE), ad)


def _pack(mod, emb, off, meta, y, conv, tr_idx, te_idx, build, ridge, pool,
          ad):
    return dict(mod=mod, emb=emb, off=off, meta=meta, y=y, conv=conv,
                tr_idx=tr_idx, te_idx=te_idx, build=build, ridge=ridge,
                pool=pool, ad=ad,
                fail_test=[i for i in set(conv[te_idx].tolist()) if y[i]])


SETUP = {"toolbench": _setup_toolbench, "abcd": _setup_abcd, "bge": _setup_bge}


def run(dataset="toolbench"):
    if dataset not in SETUP:
        raise NotImplementedError(f"R18 sweep has no cell for {dataset!r}.")
    s = SETUP[dataset]()
    mod, ad = s["mod"], s["ad"]
    sy = s["y"][s["conv"]]
    pool_size = int(len(s["pool"]))

    arms = {}
    for arm in SWEPT_ARMS:
        X = s["build"](arm)
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[s["tr_idx"]],
                                                    sy[s["tr_idx"]])
        acc = float(roc_auc_score(sy[s["te_idx"]],
                                  clf.decision_function(X[s["te_idx"]])))
        thr = mod._threshold(clf.decision_function(X[s["tr_idx"]]),
                             s["conv"][s["tr_idx"]], s["y"])
        base = sum(1 for i in s["fail_test"]
                   if (clf.decision_function(X[s["conv"] == i]) >= thr).any())
        base_recall = base / max(len(s["fail_test"]), 1)

        curve = {}
        for budget in BUDGETS:
            supp = _attack_budget(ad, s["emb"], s["off"], s["meta"], arm, clf,
                                  thr, s["pool"], s["fail_test"], s["ridge"],
                                  budget)
            recall = (len(s["fail_test"]) - supp) / max(len(s["fail_test"]), 1)
            manip = ((base_recall - recall) / base_recall
                     if base_recall > 0 else 0.0)
            curve[str(budget)] = {"recall": float(recall),
                                  "manipulability": float(manip)}
        arms[arm] = {"accuracy_auc": acc, "base_recall": float(base_recall),
                     "curve": curve}

    # Nesting guard: the n=16 point must reproduce the published number.
    guard = {}
    ref = PUBLISHED_N16.get(dataset, {})
    for arm in SWEPT_ARMS:
        got = arms[arm]["curve"]["16"]["manipulability"]
        exp = ref.get(arm)
        ok = exp is None or abs(got - exp) <= GUARD_TOL
        guard[arm] = {"n16": got, "published": exp, "within_tol": bool(ok)}
        if not ok:
            raise AssertionError(
                f"R18 nesting guard failed for {dataset}/{arm}: n16 manip "
                f"{got:.4f} != published {exp:.4f} (tol {GUARD_TOL}). Setup "
                "diverged; run is void, not a finding.")

    typ_ceiling = arms["typicality"]["curve"]["exhaustive"]["manipulability"]
    typ_verdict = ("kill" if typ_ceiling > 0.10 else
                   "keep" if typ_ceiling <= 0.05 else "grey")

    res = {"dataset": dataset, "n_test_failures": len(s["fail_test"]),
           "pool_size": pool_size, "budgets": [str(b) for b in BUDGETS],
           "arms": arms, "nesting_guard": guard,
           "typicality_ceiling_manipulability": float(typ_ceiling),
           "typicality_verdict": typ_verdict,
           "invariant_arms_note": (
               "skeleton / drop_written arms: manipulability 0 for every "
               "budget by construction (agentlike block absent from "
               "feature); asserted, not swept.")}
    save_result(f"adversary_sweep_{dataset}", res, dataset=dataset,
                params={"seed": mod.SEED,
                        "budgets": [str(b) for b in BUDGETS]})
    print(json.dumps({"dataset": dataset, "pool_size": pool_size,
                      "guard": guard,
                      "flat": {k: round(v["manipulability"], 3)
                               for k, v in arms["flat"]["curve"].items()},
                      "typicality": {
                          k: round(v["manipulability"], 3)
                          for k, v in arms["typicality"]["curve"].items()},
                      "typicality_verdict": typ_verdict}, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0] if args else "toolbench")
    return 0


if __name__ == "__main__":
    sys.exit(main())
