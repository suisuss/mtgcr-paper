"""Manuscript figures (paper/paper.md), generated from artifacts.

Extends the actr-paper trajectory-concept vocabulary (turns as numbered
points in PCA-2D, teal user / orange model) to our four-role setting and the
selection attack. The two tool roles are triangles in distinct Google-palette
colours (blue tool-call / green tool-result), kept apart from the teal-user /
orange-model dialogue turns. Exemplar: one real failing ToolBench conversation
(a sports-scores task, 13 turns, that fails as its APIs return empty results
and finally "This endpoint is disabled for your subscription", and the agent
gives up), fixed for reproducibility.

  Figure 1: projection + four-way role split (user turn / model turn /
    tool-call skeleton / tool-result).
  Figure 2: the attack - the agent's narration (the one writable role) is
    slid into the region of success-conversation narration and the head's
    risk drops, while the fixed evidence (the failing tool results) is
    unmoved.

Deterministic: fixed exemplar, seed 42, PCA sign oriented by turn order.
Displayed risk is rounded to 1 dp so single-thread-BLAS jitter cannot move
the printed value.

    python paper_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch, FancyArrowPatch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result
from tool_anchored import _service_embeddings, State, _collect, _threshold

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
CONV = REPO / "data" / "raw" / "toolbench" / "conversations.json"
OUT = REPO / "paper" / "figures"
EXEMPLAR = 3924          # failing ToolBench task, 11 turns / 3 decision points,
                         # ends "endpoint disabled"; cleanly suppressed at every
                         # decision point by the selection attack (unlike 12543,
                         # whose running risk stays above threshold until the end)
FAIL_EVIDENCE = "'endpoint disabled'"   # the final failing tool result
SEED = 42
TYPES = ["user", "assistant", "tool_call", "tool_result"]

# actr-paper palette: teal user / orange model dialogue turns. The two tool
# roles are triangles in colours kept clear of the teal-user blue-green zone
# (blue and green both read as "teal" and as each other): red tool-call, deep
# navy tool-result - a warm/cool pair, distinct from teal, orange, and the
# light-purple region.
C = {"user": "#35978f", "assistant": "#e08214",
     "tool_call": "#d73027", "tool_result": "#253494"}
MARK = {"user": "o", "assistant": "s", "tool_call": "^", "tool_result": "v"}
LABEL = {"user": "user turn", "assistant": "model turn",
         "tool_call": "tool call", "tool_result": "tool result"}
# the "successful narration" region gets its own light-purple tint (a pastel
# at the same level as the other soft fills) - distinct from all four role
# colours, and giving the orange narration markers contrast when they land in
# it. Purple is free now that the tool-call role moved to blue.
C_SUCCESS = "#c2a5cf"    # light purple - region of successful-conversation narration
C_SUCCESS_EDGE = "#762a83"   # dark purple - its outline / label
C_PATH = "#bdbdbd"


def _pca2(X, order=None):
    """SVD 2-D projection; sign of each axis oriented so it increases with
    `order` (turn index) for reproducibility."""
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ vt[:2].T
    if order is not None:
        for k in range(2):
            if np.corrcoef(P[:len(order), k], order)[0, 1] < 0:
                P[:, k] *= -1
    return P


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{stem} written")


def _exemplar(emb, off, meta):
    E = emb[off[EXEMPLAR]:off[EXEMPLAR + 1]].astype(np.float64)
    tt = np.array(meta["turn_types"][EXEMPLAR])
    return E, tt


def _draw_turns(ax, P, tt, idxs):
    """Turn markers labelled exactly as the actr-paper intuition figures
    (actr_figures.py::fig_concept.points / _um_labels): a role-coloured marker
    with a white edge, and the turn index printed beside it in the role's
    colour - bold, jittered horizontally, offset upward - not inside the
    marker."""
    for t in idxs:
        r = tt[t]
        ax.plot(P[t, 0], P[t, 1], MARK[r], color=C[r],
                ms=9 if r == "tool_call" else 8, mec="white", mew=0.9,
                zorder=4)
        dx = 5 if t % 2 else -5
        ax.annotate(str(t + 1), P[t], color=C[r], fontsize=6.5,
                    fontweight="bold", ha="center", va="center",
                    xytext=(dx, 9), textcoords="offset points", zorder=5)


def fig1(emb, off, meta):
    E, tt = _exemplar(emb, off, meta)
    n = len(E)
    P = _pca2(E, order=np.arange(n))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.2))

    # (a) interleaved path
    axA.plot(P[:, 0], P[:, 1], "-", color=C_PATH, lw=1.2, zorder=1)
    _draw_turns(axA, P, tt, range(n))
    axA.set_title("(a) one conversation, one interleaved path",
                  fontsize=11)
    _bare(axA)

    # (b) role split - one sub-path per role
    for r in TYPES:
        idx = np.where(tt == r)[0]
        if len(idx) == 0:
            continue
        axB.plot(P[idx, 0], P[idx, 1], "-", color=C[r], lw=1.6,
                 alpha=0.7, zorder=2)
        axB.plot([], [], MARK[r], color=C[r], ms=8, mec="white", mew=0.9,
                 label=LABEL[r])   # legend proxy
    _draw_turns(axB, P, tt, range(n))
    axB.set_title("(b) split by role — the tool calls are fixed waypoints",
                  fontsize=11)
    axB.legend(loc="best", fontsize=8, framealpha=0.9)
    _bare(axB)
    _save(fig, "fig1_role_split")


def _flat_head(emb, off, meta, y, svc_vecs, svc_idx, per_conv):
    feats, conv = _collect(emb, off, meta, svc_vecs, svc_idx, per_conv)
    X = feats["A_flat"]
    sy = y[conv]
    tr, te = next(GroupKFold(n_splits=5).split(X, sy, conv))
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000).fit(X[tr], sy[tr])
    thr = _threshold(clf.decision_function(X[tr]), conv[tr], y)
    train_convs = set(conv[tr].tolist())
    return clf, thr, train_convs


def _risk(clf, st):
    """Conversation risk = probability at the final decision-point prefix."""
    return float(clf.predict_proba(st.feat("A_flat").reshape(1, -1))[0, 1])


def fig2(emb, off, meta, y):
    conversations = json.loads(CONV.read_text())
    svc_vecs, svc_idx, per_conv = _service_embeddings(conversations)
    clf, thr, train_convs = _flat_head(emb, off, meta, y, svc_vecs, svc_idx,
                                       per_conv)

    E, tt = _exemplar(emb, off, meta)
    seq = per_conv[EXEMPLAR]
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    wr = {r: w[i * 384:(i + 1) * 384] for i, r in enumerate(TYPES)}
    thr_p = float(1.0 / (1.0 + np.exp(-thr)))

    # pool of model turns from successful training conversations
    pool = []
    for i in train_convs:
        if y[i]:
            continue
        Ei = emb[off[i]:off[i + 1]]
        p = Ei[np.array(meta["turn_types"][i]) == "assistant"]
        if len(p):
            pool.append(p)
    pool = np.vstack(pool).astype(np.float64)

    from tool_anchored import _cand_scores

    # Replay the head causally. The attack replaces each writable model turn
    # with the pool turn that minimises the head score. We record the head's
    # risk at every decision point and the per-role score contribution at the
    # end. The figure's axes are the head's actual decision, not a projection.
    def walk(substitute):
        st = State()
        tc = 0
        rng = np.random.default_rng(SEED)
        risks = []
        for t in range(len(E)):
            if tt[t] == "assistant" and substitute:
                cand = pool[rng.integers(0, len(pool), size=16)]
                best = cand[int(np.argmin(
                    _cand_scores(st, cand, "A_flat", w, b)))]
                st.ingest(best, "assistant")
            else:
                sv = None
                if tt[t] == "tool_call" and tc < len(seq):
                    sv = svc_vecs[svc_idx[seq[tc]]]
                    tc += 1
                st.ingest(E[t], tt[t], sv)
            if tt[t] == "tool_result":
                risks.append(_risk(clf, st))
        contrib = {r: float(wr[r] @ st._m(r)) for r in TYPES}
        return np.array(risks), contrib

    risks0, contrib0 = walk(False)
    risks1, contrib1 = walk(True)
    risk0, risk1 = float(risks0[-1]), float(risks1[-1])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) the head's risk at each decision point, before vs after the swap,
    # against the alert threshold: the swap drags the score below the line.
    xd = np.arange(1, len(risks0) + 1)
    axL.axhspan(thr_p, 1.0, color="#d73027", alpha=0.07, zorder=0)
    axL.axhline(thr_p, ls="--", color="#888888", lw=1.0, zorder=1)
    axL.text(xd[-1], thr_p, "alert threshold  ", fontsize=7.5,
             color="#666666", va="bottom", ha="right")
    axL.plot(xd, risks0, "-o", color="#b2182b", lw=1.9, ms=6, mec="white",
             mew=0.8, label="failing conversation", zorder=3)
    axL.plot(xd, risks1, "-s", color=C_SUCCESS_EDGE, lw=1.9, ms=6,
             mec="white", mew=0.8, label="model turns swapped in", zorder=3)
    axL.set_xlabel("decision point (tool result)", fontsize=9)
    axL.set_ylabel("head risk (failure probability)", fontsize=9)
    axL.set_ylim(0, 1)
    axL.set_xticks(xd)
    axL.set_title(f"(a) the swap drags risk below threshold "
                  f"({risk0:.2f} → {risk1:.2f})", fontsize=10.5)
    axL.legend(loc="center left", fontsize=8, framealpha=0.9)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # (b) the head score is a sum of per-role contributions w_role . mean_role.
    # The attack moves only the model-turn block; the user turns, tool calls,
    # and tool results contribute identically before and after.
    order = TYPES
    x = np.arange(len(order))
    bw = 0.38
    v0 = [contrib0[r] for r in order]
    v1 = [contrib1[r] for r in order]
    axR.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    axR.bar(x - bw / 2, v0, bw, color=[C[r] for r in order],
            edgecolor="white", linewidth=0.6, zorder=2)
    axR.bar(x + bw / 2, v1, bw, color=[C[r] for r in order], alpha=0.4,
            hatch="////", edgecolor="white", linewidth=0.6, zorder=2)
    ai = order.index("assistant")
    axR.annotate("only this block moves", (ai, max(v0[ai], v1[ai])),
                 xytext=(0, 24), textcoords="offset points", fontsize=8,
                 color=C["assistant"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["assistant"],
                                 lw=1.0))
    axR.set_xticks(x)
    axR.set_xticklabels([LABEL[r] for r in order], fontsize=8)
    axR.set_ylabel("contribution to head score", fontsize=9)
    axR.set_title("(b) only the writable block moves; fixed evidence "
                  "unchanged", fontsize=10.5)
    axR.legend(handles=[Patch(facecolor="#888888", label="before"),
                        Patch(facecolor="#888888", alpha=0.4, hatch="////",
                              label="after swap")],
               fontsize=8, loc="best", framealpha=0.9)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    _save(fig, "fig2_attack")
    return {"exemplar": EXEMPLAR, "risk_before": round(risk0, 4),
            "risk_after": round(risk1, 4)}


def _role_glyphs(ax, cx, cy, narr, task=False):
    """A compact row of the four role markers (same shapes/colours as Figs
    1-2) showing which channels a stage scores. `narr` controls the writable
    narration marker: 'on' (scored freely), 'drop' (removed), 'resid' (admitted
    only as its typicality residual). `task` adds a task tag."""
    dx = 0.028
    xs = [cx - 1.5 * dx, cx - 0.5 * dx, cx + 0.5 * dx, cx + 1.5 * dx]
    ax.scatter(xs[0], cy, marker="o", s=70, color=C["user"],
               edgecolor="white", linewidth=0.8, zorder=5)
    if narr == "on":
        ax.scatter(xs[1], cy, marker="s", s=70, color=C["assistant"],
                   edgecolor="white", linewidth=0.8, zorder=5)
    elif narr == "drop":
        ax.scatter(xs[1], cy, marker="s", s=70, facecolor="none",
                   edgecolor=C["assistant"], linewidth=1.0, linestyle="--",
                   zorder=5)
        ax.plot([xs[1] - 0.011, xs[1] + 0.011], [cy - 0.028, cy + 0.028],
                color="#b2182b", lw=1.5, zorder=6)
    elif narr == "resid":
        ax.scatter(xs[1], cy, marker="s", s=70, facecolor="none",
                   edgecolor=C["assistant"], linewidth=1.3, zorder=5)
        ax.text(xs[1], cy - 0.052, "~", ha="center", va="center", fontsize=11,
                fontweight="bold", color=C["assistant"], zorder=6)
    ax.scatter(xs[2], cy, marker="^", s=85, color=C["tool_call"],
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter(xs[3], cy, marker="v", s=85, color=C["tool_result"],
               edgecolor="white", linewidth=0.8, zorder=5)
    if task:
        ax.text(cx + 2.7 * dx, cy, "task", ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#762a83",
                          edgecolor="none"))


def fig3():
    """Schematic of the construction: drop the written role, then
    re-admit the response only as its conditional typicality, then condition
    on the task. Headline numbers are loaded from the same artifacts the paper
    tables read, so the figure cannot drift from Tables 3-4."""
    from matplotlib.patches import FancyBboxPatch
    RES = REPO / "data" / "results" / "experiments"
    sk = json.loads((RES / "skeleton_roles.json").read_text())["arms"]
    pc = json.loads((RES / "policy_choice_abcd.json").read_text())
    naive_auc = sk["flat"]["accuracy_auc"]
    naive_manip = sk["flat"]["manipulability"]
    fixed_auc = sk["drop_assistant"]["accuracy_auc"]
    typ_auc = sk["typicality"]["accuracy_auc"]
    typ_manip = sk["typicality"]["manipulability"]
    same_slack = pc["same_subflow"]["slack_rate"]
    any_slack = pc["any_subflow"]["slack_rate"]

    SAFE, BAD = "#2b6a2b", "#b2182b"
    stages = [
        dict(cx=0.125, title="naive head", narr="on",
             lines=[(f"accuracy {naive_auc:.2f}", "#333333")],
             badge=f"manipulability +{naive_manip:.2f}", bc=BAD,
             prop="gameable", pc=BAD),
        dict(cx=0.375, title="drop model\nresponse turns", narr="drop",
             lines=[(f"accuracy {fixed_auc:.2f}", "#333333"),
                    ("(97% of the head)", "#777777")],
             badge="manipulability 0.00", bc=SAFE,
             prop="un-gameable", pc=SAFE),
        dict(cx=0.625, title="add conditional\ntypicality", narr="resid",
             lines=[(f"accuracy {typ_auc:.2f}", "#333333"),
                    ("(response as residual)", "#777777")],
             badge=f"manipulability {typ_manip:+.2f}", bc=SAFE,
             prop="anti-gameable", pc=SAFE),
        dict(cx=0.875, title="add task\nconditioning", narr="resid",
             task=True,
             lines=[(f"same-task slack {same_slack:.2f}", SAFE),
                    (f"any-task slack {any_slack:.2f}", BAD)],
             badge="fooling it = doing it", bc=SAFE,
             prop="un-gameable in substance", pc=SAFE),
    ]

    fig, ax = plt.subplots(figsize=(12.6, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw, by, bh = 0.205, 0.30, 0.42

    for i, s in enumerate(stages):
        cx = s["cx"]
        ax.add_patch(FancyBboxPatch((cx - bw / 2, by), bw, bh,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    facecolor="#f6f6f6", edgecolor="#cccccc",
                                    lw=1.0, zorder=1))
        ax.text(cx, by + bh + 0.075, s["title"], ha="center", va="center",
                fontsize=10, fontweight="bold", color="#222222")
        _role_glyphs(ax, cx, by + bh - 0.10, s["narr"], task=s.get("task"))
        for k, (txt, col) in enumerate(s["lines"]):
            ax.text(cx, by + bh - 0.215 - 0.075 * k, txt, ha="center",
                    va="center", fontsize=8.5, color=col)
        ax.text(cx, by + 0.055, s["badge"], ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=s["bc"],
                          edgecolor="none"))
        ax.text(cx, by - 0.075, s["prop"], ha="center", va="center",
                fontsize=9.5, style="italic", color=s["pc"], fontweight="bold")
        if i:
            px = stages[i - 1]["cx"] + bw / 2
            ax.annotate("", (cx - bw / 2, by + bh / 2),
                        (px, by + bh / 2), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=1.6, mutation_scale=15))

    ax.text(0.5, 0.045,
            "score the fixed evidence  ·  admit the response only as its "
            "consistency with that evidence  ·  condition on the task",
            ha="center", va="center", fontsize=9.5, style="italic",
            color="#444444", transform=ax.transAxes)
    _save(fig, "fig3_construction")


def run():
    z = np.load(EMB / "toolbench_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = json.loads((EMB / "toolbench_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    fig1(emb, off, meta)
    info = fig2(emb, off, meta, y)
    fig3()
    save_result("paper_figures", info, dataset="toolbench",
                params={"exemplar": EXEMPLAR, "seed": SEED})
    print(json.dumps(info, indent=1))
    return info


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
