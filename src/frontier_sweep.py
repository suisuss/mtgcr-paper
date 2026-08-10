"""Round-19: the accuracy/gameability frontier of re-admitted narration.

Round 18 found the conditional-typicality residual's anti-gameability erodes
with adversary budget (ToolBench -0.10 -> +0.03 at the selection ceiling). The
mechanism is a near-conservation law: a residual's ceiling manipulability ~=
the independent recall it adds over the fixed-role head, because a strong
selection adversary removes exactly that independent contribution. This round
maps the constructions into the (independent accuracy, ceiling manipulability)
plane and tests whether one-sided rectification flattens/caps the curve (A),
whether task-conditioning improves the exchange rate (B), and reconfirms the
fixed-role zero (C). PREREGISTRATION-R19.md.

Arms: fixed_role, res_2s (guard anchor), res_1s@p50/p75/p90, res_task_2s,
res_task_1s@p50. Reuses the round-13 setups verbatim (imported helpers); no
existing script or artifact is modified. Three cells: ToolBench/MiniLM
(primary), ABCD/MiniLM, ToolBench/BGE-large.

    python frontier_sweep.py toolbench
    python frontier_sweep.py abcd
    python frontier_sweep.py bge
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

BUDGETS = [1, 4, 16, 64, 256, "exhaustive"]
SEED = 42
GUARD_TOL = 0.02
# Published res_2s (== round-13 typicality) manipulability at n=16, per cell.
PUBLISHED_RES2S_N16 = {"toolbench": -0.0969529085872576,
                       "abcd": 0.0,
                       "bge": -0.039682539682539736}


# ---------------------------------------------------------------------------
# Per-corpus setup: each returns a uniform bundle. Task descriptor = the
# opening turn of the querying role (user / customer). Head context is always
# ctx_all; task-conditioning lives only in the residual model (ridge_task).
# ---------------------------------------------------------------------------

def _bundle(mod, emb, off, meta, agentlike, decision, task_role, nonagent,
            make_state):
    y = np.array(meta["has_failure"], dtype=int)
    atoms, conv = _collect_dispatch(mod, emb, off, meta)
    sy = y[conv]
    tr_idx, te_idx = next(GroupKFold(n_splits=5).split(
        atoms["skeleton"], sy, conv))
    ctx_all = np.hstack([atoms[t] for t in mod.CONTEXT] + [atoms["skeleton"]])
    ridge = Ridge(alpha=1.0).fit(ctx_all[tr_idx], atoms[agentlike][tr_idx])
    resid = np.linalg.norm(atoms[agentlike] - ridge.predict(ctx_all), axis=1)

    dim = emb.shape[1]
    task_vec = np.zeros((len(off) - 1, dim), np.float64)
    for i in range(len(off) - 1):
        tt = meta["turn_types"][i]
        E = emb[off[i]:off[i + 1]]
        k = next((j for j in range(len(tt)) if tt[j] == task_role), None)
        task_vec[i] = E[k] if k is not None else E[0]
    ctx_task = np.hstack([ctx_all, task_vec[conv]])
    ridge_task = Ridge(alpha=1.0).fit(ctx_task[tr_idx], atoms[agentlike][tr_idx])
    resid_task = np.linalg.norm(atoms[agentlike] - ridge_task.predict(ctx_task),
                                axis=1)

    train_convs = set(conv[tr_idx].tolist())
    fail_test = [i for i in set(conv[te_idx].tolist()) if y[i]]
    pool = []
    for i in train_convs:
        if y[i]:
            continue
        E = emb[off[i]:off[i + 1]]
        tt = np.array(meta["turn_types"][i])
        p = E[tt == agentlike]
        if len(p):
            pool.append(p)
    pool = np.vstack(pool).astype(np.float64)

    return dict(mod=mod, emb=emb, off=off, meta=meta, y=y, conv=conv, sy=sy,
                tr_idx=tr_idx, te_idx=te_idx, atoms=atoms, ctx_all=ctx_all,
                ridge=ridge, resid=resid, ridge_task=ridge_task,
                resid_task=resid_task, task_vec=task_vec, pool=pool,
                fail_test=fail_test, agentlike=agentlike, decision=decision,
                nonagent=nonagent, make_state=make_state)


def _collect_dispatch(mod, emb, off, meta):
    """Call the module's own _collect with its corpus-specific args."""
    name = mod.__name__
    if name == "skeleton_roles":
        from tool_anchored import _service_embeddings
        svc_vecs, svc_idx, per_conv = _service_embeddings(
            json.loads(mod.CONV.read_text()))
        return mod._collect(emb, off, meta, svc_vecs, svc_idx, per_conv)
    if name == "skeleton_roles_abcd":
        from skeleton_abcd import _slug_embeddings
        slug_vecs, slug_idx = _slug_embeddings(meta["action_names"])
        return mod._collect(emb, off, meta, slug_vecs, slug_idx)
    if name == "skeleton_roles_bge":
        emb2, off2, per_conv, svec, sidx = mod._encode()
        return mod._collect(emb2, off2, meta, svec, sidx, per_conv,
                            emb2.shape[1])
    raise ValueError(name)


def _setup_toolbench():
    import skeleton_roles as mod
    from tool_anchored import _service_embeddings
    z = np.load(mod.EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((mod.EMB / "toolbench_meta.json").read_text())
    svc_vecs, svc_idx, per_conv = _service_embeddings(
        json.loads(mod.CONV.read_text()))

    def nonagent(i):
        seq, tc = per_conv[i], [0]
        def ing(st, e, ttype, t):
            sv = None
            if ttype == "tool_call" and tc[0] < len(seq):
                sv = svc_vecs[svc_idx[seq[tc[0]]]]
                tc[0] += 1
            st.ingest(e, ttype, sv)
        return ing
    return _bundle(mod, emb, off, meta, "assistant", "tool_result", "user",
                   nonagent, mod.State)


def _setup_abcd():
    import skeleton_roles_abcd as mod
    from skeleton_abcd import _slug_embeddings
    z = np.load(mod.EMB / "abcd_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((mod.EMB / "abcd_meta.json").read_text())
    slug_vecs, slug_idx = _slug_embeddings(meta["action_names"])
    an = meta["action_names"]

    def nonagent(i):
        names = an[i]
        def ing(st, e, ttype, t):
            slug = (slug_vecs[slug_idx[names[t]]]
                    if ttype == "action" and names[t] in slug_idx else None)
            st.ingest(e, ttype, slug)
        return ing
    return _bundle(mod, emb, off, meta, "agent", "customer", "customer",
                   nonagent, mod.State)


def _setup_bge():
    import skeleton_roles_bge as mod
    emb, off, per_conv, svec, sidx = mod._encode()
    dim = emb.shape[1]
    meta = json.loads((mod.EMB / "toolbench_meta.json").read_text())

    def nonagent(i):
        seq, tc = per_conv[i], [0]
        def ing(st, e, ttype, t):
            sv = None
            if ttype == "tool_call" and tc[0] < len(seq):
                sv = svec[sidx[seq[tc[0]]]]
                tc[0] += 1
            st.ingest(e, ttype, sv)
        return ing
    return _bundle(mod, emb, off, meta, "assistant", "tool_result", "user",
                   nonagent, lambda: mod.State(dim))


SETUP = {"toolbench": _setup_toolbench, "abcd": _setup_abcd, "bge": _setup_bge}


# ---------------------------------------------------------------------------
# Generic arm building + attack.
# ---------------------------------------------------------------------------

def run(dataset="toolbench"):
    if dataset not in SETUP:
        raise NotImplementedError(f"R19 has no cell for {dataset!r}.")
    s = SETUP[dataset]()
    mod = s["mod"]
    AG, DEC = s["agentlike"], s["decision"]
    ctx_all, resid, resid_task = s["ctx_all"], s["resid"], s["resid_task"]
    sy, tr_idx, te_idx, conv, y = (s["sy"], s["tr_idx"], s["te_idx"],
                                   s["conv"], s["y"])
    fail_test, pool = s["fail_test"], s["pool"]

    def fit(X):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000).fit(X[tr_idx], sy[tr_idx])
        acc = float(roc_auc_score(sy[te_idx],
                                  clf.decision_function(X[te_idx])))
        thr = mod._threshold(clf.decision_function(X[tr_idx]), conv[tr_idx], y)
        return clf, acc, thr

    # Fixed-role head (H_fixed): the drop-written-role reference (manip 0).
    hf, acc_fixed, thr_f = fit(ctx_all)
    s_fixed_all = hf.decision_function(ctx_all)

    r0 = {p: float(np.percentile(resid[tr_idx], p)) for p in (50, 75, 90)}
    r0t = {50: float(np.percentile(resid_task[tr_idx], 50))}

    specs = [dict(name="fixed_role", kind="invariant", clf=hf, thr=thr_f,
                  acc=acc_fixed, uses_task=False, rectify=None, gcoef=None)]

    def add(name, use_task, rectify_p):
        rr = resid_task if use_task else resid
        r0map = r0t if use_task else r0
        if rectify_p is None:
            X = np.hstack([ctx_all, rr[:, None]])
            clf, acc, thr = fit(X)
            specs.append(dict(name=name, kind="joint", clf=clf, thr=thr,
                              acc=acc, uses_task=use_task, rectify=None,
                              gcoef=None))
        else:
            r0v = r0map[rectify_p]
            X = np.column_stack([s_fixed_all, np.maximum(0.0, rr - r0v)])
            clf, acc, thr = fit(X)
            specs.append(dict(name=name, kind="stacked", clf=clf, thr=thr,
                              acc=acc, uses_task=use_task, rectify=r0v,
                              gcoef=float(clf.coef_[0][1])))

    add("res_2s", False, None)
    for p in (50, 75, 90):
        add(f"res_1s@p{p}", False, p)
    add("res_task_2s", True, None)
    add("res_task_1s@p50", True, 50)

    def base_recall(spec):
        clf, thr = spec["clf"], spec["thr"]
        hit = 0
        for i in fail_test:
            m = conv == i
            if spec["kind"] == "invariant":
                sc = clf.decision_function(ctx_all[m])
            elif spec["kind"] == "joint":
                rr = resid_task if spec["uses_task"] else resid
                sc = clf.decision_function(np.hstack([ctx_all[m], rr[m, None]]))
            else:
                rr = resid_task if spec["uses_task"] else resid
                g = np.maximum(0.0, rr[m] - spec["rectify"])
                sc = clf.decision_function(np.column_stack([s_fixed_all[m], g]))
            if (sc >= thr).any():
                hit += 1
        return hit / max(len(fail_test), 1)

    def attack(spec, budget):
        if spec["kind"] == "invariant":
            return 0
        clf, thr = spec["clf"], spec["thr"]
        use_task = spec["uses_task"]
        rmodel = s["ridge_task"] if use_task else s["ridge"]
        rng = np.random.default_rng(SEED)
        suppressed = 0
        for i in fail_test:
            E = s["emb"][s["off"][i]:s["off"][i + 1]].astype(np.float64)
            tt = s["meta"]["turn_types"][i]
            st = s["make_state"]()
            ing = s["nonagent"](i)
            tvec = s["task_vec"][i]
            alerted = False
            for t in range(len(E)):
                if tt[t] == AG:
                    if budget == "exhaustive":
                        C = pool
                    else:
                        C = pool[rng.integers(0, len(pool), size=budget)]
                    n = st.cnts[AG]
                    am_new = (st.sums[AG][None, :] + C) / (n + 1)
                    ctx = st.context()
                    cin = np.concatenate([ctx, tvec]) if use_task else ctx
                    pred = rmodel.predict(cin.reshape(1, -1))[0]
                    res_cand = np.linalg.norm(am_new - pred, axis=1)
                    if spec["kind"] == "joint":
                        sc = _cand_joint(clf, ctx, res_cand)
                    else:
                        s_fixed = hf.decision_function(ctx.reshape(1, -1))[0]
                        g_cand = np.maximum(0.0, res_cand - spec["rectify"])
                        sc = _cand_stacked(clf, s_fixed, g_cand)
                    st.ingest(C[int(np.argmin(sc))], AG)
                else:
                    ing(st, E[t], tt[t], t)
                if tt[t] == DEC:
                    ctx = st.context()
                    cin = np.concatenate([ctx, tvec]) if use_task else ctx
                    res = float(np.linalg.norm(
                        st.m(AG) - rmodel.predict(cin.reshape(1, -1))[0]))
                    if spec["kind"] == "joint":
                        score = _score_joint(clf, ctx, res)
                    else:
                        s_fixed = hf.decision_function(ctx.reshape(1, -1))[0]
                        score = _score_stacked(
                            clf, s_fixed, max(0.0, res - spec["rectify"]))
                    if score >= thr:
                        alerted = True
            if not alerted:
                suppressed += 1
        return suppressed

    arms = {}
    for spec in specs:
        br = base_recall(spec)
        curve = {}
        for budget in BUDGETS:
            if spec["kind"] == "invariant":
                rec = br
            else:
                supp = attack(spec, budget)
                rec = (len(fail_test) - supp) / max(len(fail_test), 1)
            manip = (br - rec) / br if br > 0 else 0.0
            curve[str(budget)] = {"recall": float(rec),
                                  "manipulability": float(manip)}
        ck, n1k = str(BUDGETS[-1]), str(BUDGETS[0])
        arms[spec["name"]] = {
            "kind": spec["kind"], "accuracy_auc": spec["acc"],
            "base_recall": float(br),
            "independent_acc": float(spec["acc"] - acc_fixed),
            "gcoef": spec["gcoef"], "curve": curve,
            "ceiling_manip": float(curve[ck]["manipulability"]),
            "flatness_span": float(curve[ck]["manipulability"]
                                   - curve[n1k]["manipulability"]),
            "independent_recall": float(br - curve[ck]["recall"])}

    got = arms["res_2s"]["curve"]["16"]["manipulability"]
    exp = PUBLISHED_RES2S_N16[dataset]
    if abs(got - exp) > GUARD_TOL:
        raise AssertionError(
            f"R19 guard failed ({dataset}): res_2s n16 {got:.4f} != "
            f"published {exp:.4f}.")

    pts = [(a["independent_acc"], a["ceiling_manip"])
           for n, a in arms.items() if a["kind"] != "invariant"]
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    rho = float(np.corrcoef(rx, ry)[0, 1]) if len(xs) > 1 else float("nan")

    res = {"dataset": dataset, "n_test_failures": len(fail_test),
           "pool_size": int(len(pool)), "acc_fixed_role": float(acc_fixed),
           "budgets": [str(b) for b in BUDGETS], "r0_percentiles": r0,
           "r0_task": r0t, "guard_res2s_n16": {"got": got, "published": exp},
           "conservation_rank_corr": rho,
           "plane": {n: {"independent_acc": a["independent_acc"],
                         "ceiling_manip": a["ceiling_manip"],
                         "flatness_span": a["flatness_span"],
                         "accuracy_auc": a["accuracy_auc"]}
                     for n, a in arms.items()},
           "arms": arms}
    save_result(f"frontier_sweep_{dataset}", res, dataset=dataset,
                params={"seed": SEED, "budgets": [str(b) for b in BUDGETS]})
    print(json.dumps({"dataset": dataset, "guard": res["guard_res2s_n16"],
                      "conservation_rank_corr": round(rho, 3),
                      "plane": {n: {"ind_acc": round(v["independent_acc"], 3),
                                    "ceil_manip": round(v["ceiling_manip"], 3),
                                    "flat_span": round(v["flatness_span"], 3)}
                                for n, v in res["plane"].items()}}, indent=1))
    return res


def _score_joint(clf, ctx, res):
    return clf.decision_function(np.concatenate([ctx, [res]]).reshape(1, -1))[0]


def _score_stacked(clf, s_fixed, g):
    return clf.decision_function(np.array([[s_fixed, g]]))[0]


def _cand_joint(clf, ctx, res_cand):
    m = len(res_cand)
    return clf.decision_function(np.hstack([np.tile(ctx, (m, 1)),
                                            res_cand[:, None]]))


def _cand_stacked(clf, s_fixed, g_cand):
    m = len(g_cand)
    return clf.decision_function(np.hstack([np.full((m, 1), s_fixed),
                                            g_cand[:, None]]))


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0] if args else "toolbench")
    return 0


if __name__ == "__main__":
    sys.exit(main())
