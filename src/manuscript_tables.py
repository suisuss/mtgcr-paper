"""Manuscript table generator + draft-vs-artifact gate (paper/paper.md).

Modeled on the actr-paper/gp-paper convention: each builder emits a table's
markdown block from the flat result artifacts, formatted to match paper.md
byte-for-byte (whitespace-insensitive). `--check` locates each emitted table in
paper.md by its header row + first data cell and diffs cell-by-cell, so the
manuscript's numbers are gated against the record rather than hand-authored.
Each generated table carries a `<!-- src: manuscript_tables.py -->` tag.

    python manuscript_tables.py            regenerate data/tables/*.md
    python manuscript_tables.py --check    regenerate + diff against paper.md
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "data" / "results" / "experiments"
TABLES = REPO / "data" / "tables"
DRAFT = REPO / "paper" / "paper.md"
MINUS = "−"   # the typographic minus paper.md uses in signed cells


def _load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def _auc(v):
    return f"{v:.3f}"


def _signed(v, places, zero=None):
    """Signed cell: '+x', '<minus>x', or a bare zero string when |v| rounds
    to 0. `zero` overrides the zero rendering (e.g. '0.00' vs '0.000')."""
    q = round(v, places)
    z = zero if zero is not None else f"{0:.{places}f}"
    if abs(q) < 10 ** (-places) / 2:
        return z
    return (f"+{q:.{places}f}" if q > 0
            else f"{MINUS}{abs(q):.{places}f}")


def _pct(v):
    return f"{round(v * 100):.0f}%"


# --------------------------------------------------------------------- tables

def t3_construction():
    a = _load("skeleton_roles")["arms"]
    def acc(k): return _auc(a[k]["accuracy_auc"])
    def man(k): return _signed(a[k]["manipulability"], 2, zero="0.00")
    lines = [
        "| Representation | Accuracy AUC | Manipulability |",
        "|---|---|---|",
        f"| full head (all roles, incl. model turns) | {acc('flat')} | {man('flat')} |",
        f"| grounding only (tool identity) | {acc('skeleton')} | {man('skeleton')} |",
        f"| **fixed roles (drop the written role)** | **{acc('drop_assistant')}** | **{man('drop_assistant')}** |",
        f"| **+ conditional typicality of the reasoning and response** | **{acc('typicality')}** | **{man('typicality')}** |",
    ]
    return "t3_construction", lines


def t4_grounding():
    r = _load("policy_choice_abcd")
    s, y = r["same_subflow"], r["any_subflow"]
    def succ(d): return f"{round(d['fooled_and_succeeded'] / d['fooled'] * 100):.0f}% of those fooled"
    lines = [
        "| Substituted actions | Detector fooled | Real task succeeded | Slack (fooled, not succeeded) |",
        "|---|---|---|---|",
        f"| same task | {_pct(s['fooled_rate'])} | {succ(s)} | {s['slack_rate']:.2f} |",
        f"| any task | {_pct(y['fooled_rate'])} | {succ(y)} | {y['slack_rate']:.2f} |",
    ]
    return "t4_grounding", lines


def t7_robustness():
    ab = _load("skeleton_roles_abcd")["arms"]
    bg = _load("skeleton_roles_bge")["arms"]
    def row(label, ak, bk):
        return (f"| {label} | {_auc(ab[ak]['accuracy_auc'])} | "
                f"{_signed(ab[ak]['manipulability'], 2, zero='0.00')} | "
                f"{_auc(bg[bk]['accuracy_auc'])} | "
                f"{_signed(bg[bk]['manipulability'], 2, zero='0.00')} |")
    lines = [
        "| Representation | ABCD AUC | ABCD manip | BGE AUC | BGE manip |",
        "|---|---|---|---|---|",
        row("full head", "flat", "flat"),
        row("grounding only", "skeleton", "skeleton"),
        row("fixed roles (drop written role)", "drop_agent", "drop_assistant"),
        row("+ conditional typicality", "typicality", "typicality"),
    ]
    return "t7_robustness", lines


def t8_frontier():
    def ceil(corpus, arm):
        v = _load(f"frontier_sweep_{corpus}")["plane"][arm]["ceiling_manip"]
        return _signed(v, 3, zero="0.000")
    def row(label, arm):
        return (f"| {label} | {ceil('toolbench', arm)} | {ceil('bge', arm)} | "
                f"{ceil('abcd', arm)} |")
    lines = [
        "| Re-admission | ToolBench | BGE-large | ABCD |",
        "|---|---|---|---|",
        row("conditional typicality", "res_2s"),
        row("+ task-conditioned", "res_task_2s"),
    ]
    return "t8_frontier", lines


def t1_corpora():
    EMB = REPO / "data" / "embeddings"
    def stat(meta):
        y = json.loads((EMB / f"{meta}.json").read_text())["has_failure"]
        return f"{len(y):,}", f"{sum(y) / len(y) * 100:.1f}%"
    n_tb, f_tb = stat("toolbench_meta")
    n_ab, f_ab = stat("abcd_meta")
    n_sw, f_sw = stat("spokenwoz_meta")
    n_swe, f_swe = stat("swe_meta")
    lines = [
        "| Corpus | Conversations | Label | Failure rate | Roles |",
        "|---|---|---|---|---|",
        f"| ToolBench | {n_tb} | verified give-answer / give-up | {f_tb} | user, model, tool-call, tool-result |",
        f"| ABCD | {n_ab} | action-completion (deterministic) | {f_ab} | customer, agent, action |",
        f"| SpokenWOZ | {n_sw} | derived goal-completion | {f_sw} | user, system |",
        f"| SWE-agent-traj. | {n_swe} | PR-test-verified | {f_swe} | user, agent, tool-result |",
        "| USS-SGD / MWOZ | 1,000 each | per-turn human ratings | — | user, system |",
    ]
    return "t1_corpora", lines


def t2_earlywarn():
    tb = _load("tb_earlywarn")["auc_by_tau"]
    def ds(c):
        d = _load(f"ds_experiments_{c}")
        return d["head"]["auc_by_tau"], d["baselines"]
    ab_h, ab_b = ds("abcd")
    sw_h, sw_b = ds("spokenwoz")
    swe_h, swe_b = ds("swe")
    def row(name, aucs, base):
        return (f"| {name} | " + " | ".join(_auc(a) for a in aucs)
                + f" | {base} |")
    lines = [
        "| Corpus | Q1 | Q2 | Q3 | Q4 | best process baseline (Q4) |",
        "|---|---|---|---|---|---|",
        row("ToolBench", tb["head"], f"step-count {_auc(tb['step_count'][-1])}"),
        row("ABCD", ab_h, f"goal-dist {_auc(ab_b['goal_dist'][-1])}"),
        row("SpokenWOZ", sw_h, f"step-count {_auc(sw_b['step_count'][-1])}"),
        row("SWE-agent-traj.", swe_h,
            f"step-count {_auc(swe_b['step_count'][-1])}"),
    ]
    return "t2_earlywarn", lines


def t5_governance():
    def gm(c): return _load(f"goodhart_matrix_{c}")["arms"]
    tb, ab, sw = gm("toolbench"), gm("abcd"), gm("spokenwoz")
    def m(v): return f"{v:.2f}"
    def cell(a, arm, places):
        return f"{a[arm]['accuracy_auc']:.{places}f}"
    lines = [
        "| Corpus | Representation | Accuracy AUC | Manipulability |",
        "|---|---|---|---|",
        f"| ToolBench | naive geometric head | {cell(tb,'method_gram',3)} | {m(tb['method_gram']['manipulability'])} |",
        f"| ToolBench | + channel-divergence tripwire | {cell(tb,'governed_gram',3)} | {m(tb['governed_gram']['manipulability'])} |",
        f"| ToolBench | TRACE-style relational suite | {cell(tb,'foil_trace',3)} | (predictor too weak) |",
        f"| ABCD | naive geometric head | {cell(ab,'method_gram',2)} | {m(ab['method_gram']['manipulability'])} |",
        f"| ABCD | + channel-divergence tripwire | {cell(ab,'governed_gram',2)} | {m(ab['governed_gram']['manipulability'])} |",
        f"| SpokenWOZ | naive geometric head | {cell(sw,'method_gram',2)} | {m(sw['method_gram']['manipulability'])} |",
    ]
    return "t5_governance", lines


def t6_schemes():
    ta = _load("tool_anchored")["arms"]
    tr = _load("tool_relational")["arms"]
    ty = _load("skeleton_roles")["arms"]["typicality"]
    def m(v):  # positive unsigned, zero '0.00', negative signed
        if abs(round(v, 2)) < 0.005:
            return "0.00"
        return f"{v:.2f}" if v > 0 else f"{MINUS}{abs(v):.2f}"
    lines = [
        "| Representation | How model content enters | Accuracy AUC | Manipulability |",
        "|---|---|---|---|",
        f"| naive head | free absolute position | {_auc(ta['A_flat']['accuracy_auc'])} | {m(ta['A_flat']['manipulability'])} |",
        f"| additive tether | free position + action grounding | {_auc(ta['C_tethered']['accuracy_auc'])} | {m(ta['C_tethered']['manipulability'])} |",
        f"| relational binding | only as cosine-to-grounding | {_auc(tr['relational']['accuracy_auc'])} | {m(tr['relational']['manipulability'])} |",
        f"| grounding gate | content gated by consistency | {_auc(tr['gated']['accuracy_auc'])} | {m(tr['gated']['manipulability'])} |",
        f"| fixed roles + conditional typicality (§4) | as consistency residual | {_auc(ty['accuracy_auc'])} | {m(ty['manipulability'])} |",
        f"| grounding only | not admitted | {_auc(ta['B_skeleton']['accuracy_auc'])} | {m(ta['B_skeleton']['manipulability'])} |",
    ]
    return "t6_schemes", lines


def t9a_trend():
    def cs(c): return _load(f"shape_reward_{c}")["channel_strength"]
    s, w = cs("sgd"), cs("mwoz")
    def row(label, key): return f"| {label} | {s[key]:.3f} | {w[key]:.3f} |"
    lines = [
        "| Trend-signal strength (rank corr.) | SGD | MWOZ |",
        "|---|---|---|",
        row("model-channel only (manipulation-safe)", "trend_system_only"),
        row("user channel alone (unsafe)", "trend_user_only"),
        row("both roles", "trend_both_roles"),
    ]
    return "t9a_trend", lines


def t9b_shape():
    def delta(c):
        by = _load(f"shape_reward_{c}")["probe"]["trend"]["by_n"]
        ns = sorted(by, key=int)
        a, b = by[ns[0]], by[ns[-1]]
        return (b["reward_own"] - a["reward_own"],
                b["judge_final"] - a["judge_final"])
    d_own_s, d_jf_s = delta("sgd")
    d_own_m, d_jf_m = delta("mwoz")
    def s(v): return _signed(v, 3, zero="0.000")
    lines = [
        "| Shape reward under selection (Δ from random) | Δ own score | Δ human final-turn rating |",
        "|---|---|---|",
        f"| trend reward, SGD | {s(d_own_s)} | {s(d_jf_s)} |",
        f"| trend reward, MWOZ | {s(d_own_m)} | {s(d_jf_m)} |",
    ]
    return "t9b_shape", lines


BUILDERS = [t1_corpora, t2_earlywarn, t3_construction, t4_grounding,
            t5_governance, t6_schemes, t7_robustness, t8_frontier,
            t9a_trend, t9b_shape]


# --------------------------------------------------------------------- check

def _norm_row(line):
    cells = [c.strip() for c in line.split("|")[1:-1]]
    if cells and all(c and set(c) <= set("-:") for c in cells):
        return ["-"] * len(cells)
    return cells


def check_against_draft(emitted):
    draft = DRAFT.read_text().splitlines()
    total_ok = total_cells = 0
    per_table = {}
    for fname, lines in emitted:
        hdr, first = _norm_row(lines[0]), _norm_row(lines[2])
        start = None
        for i, dl in enumerate(draft):
            if _norm_row(dl) == hdr and i + 2 < len(draft) and \
                    _norm_row(draft[i + 2])[:1] == first[:1]:
                start = i
                break
        if start is None:
            n = sum(len(l.split("|")) - 2 for l in lines)
            per_table[fname] = {"ok": 0, "cells": n, "status": "NOT FOUND"}
            total_cells += n
            print(f"  {fname}: draft block NOT FOUND (header mismatch)")
            continue
        block = []
        j = start
        while j < len(draft) and draft[j].startswith("|"):
            block.append(draft[j])
            j += 1
        ok = cells = 0
        for k in range(max(len(block), len(lines))):
            ours = _norm_row(lines[k]) if k < len(lines) else []
            theirs = _norm_row(block[k]) if k < len(block) else []
            for m in range(max(len(ours), len(theirs))):
                cells += 1
                o = ours[m] if m < len(ours) else "<missing>"
                t = theirs[m] if m < len(theirs) else "<missing>"
                if o == t:
                    ok += 1
                else:
                    print(f"    {fname} row {k}: ours={o!r} draft={t!r}")
        total_ok += ok
        total_cells += cells
        per_table[fname] = {"ok": ok, "cells": cells,
                            "status": "OK" if ok == cells else "MISMATCH"}
        print(f"  {fname}: {ok}/{cells} [{per_table[fname]['status']}]")
    print(f"  TOTAL: {total_ok}/{total_cells}")
    return {"per_table": per_table, "total_ok": total_ok,
            "total_cells": total_cells}


def run(check=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    emitted = []
    for b in BUILDERS:
        name, lines = b()
        (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n")
        emitted.append((name, lines))
    summary = {"tables_written": [n for n, _ in emitted]}
    if check:
        summary["check"] = check_against_draft(emitted)
    return summary


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    s = run(check="--check" in args)
    if "--check" in args:
        c = s["check"]
        print(json.dumps({"total": f"{c['total_ok']}/{c['total_cells']}"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
