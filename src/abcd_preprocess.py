"""ABCD -> normalized schema (turns + goal_text + TRACER-style label).

Reads data/raw/abcd/abcd_v1.1.json.gz (+ guidelines.json) at the pinned
commit, writes data/raw/abcd/normalized.json:
    [{id, group_id, goal_text, label, turns: [{t, c, a?, v?}],
      flow, subflow}]

Turn text comes from `original`; canonical action slugs and values come
from the aligned `delexed` stream (targets[2], targets[3]).

Label (PREREGISTRATION-R4.md): success iff >= 80% of the subflow's
guideline-required actions are performed in order. Guideline subflow
titles are mapped to scenario subflow slugs by flow-scoped token overlap
(trailing _N variant suffixes stripped); action buttons slugify directly
("Pull up Account" -> pull-up-account). Match-rate diagnostics reported.

    python abcd_preprocess.py
"""
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "abcd"
OUT = RAW / "normalized.json"

GOAL_TEMPLATE = ("Customer intent: {subflow} (flow: {flow}). "
                 "Customer: {name}, member level {level}.")
STOP = {"due", "to", "the", "a", "an", "of", "for"}


def _slug_actions(sub, kinds=("kb query",)):
    """Required actions = guideline actions of type 'kb query' - the
    faithful reading of TRACER's 'knowledge-base-required actions'.
    (Diagnostic sweep 2026-07-23: all-buttons gives 74.8% failure with
    median coverage 0.60 - implausibly strict; kb-query-only gives 13.4%
    failure with median coverage 1.00.)"""
    names = []
    for a in sub.get("actions", []):
        if isinstance(a, dict):
            if kinds and a.get("type") not in kinds:
                continue
            b = a.get("button")
            if b:
                names.append(str(b).strip().lower().replace(" ", "-"))
    return names


def _tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t
            and t not in STOP}


def _subflow_map(guidelines, scenario_subflows_by_flow):
    """(flow_slug, scenario_subflow) -> ordered required action slugs."""
    mapping = {}
    for flow_title, flow in guidelines.items():
        flow_slug = flow_title.strip().lower().replace(" ", "_")
        gl_subs = {title: _slug_actions(sub)
                   for title, sub in flow.get("subflows", {}).items()}
        for scen_sub in scenario_subflows_by_flow.get(flow_slug, ()):
            base = re.sub(r"_\d+$", "", scen_sub)
            btoks = _tokens(base)
            best, best_score = None, 0
            for title, actions in gl_subs.items():
                score = len(btoks & _tokens(title))
                if score > best_score:
                    best, best_score = title, score
            if best is not None and best_score > 0:
                mapping[(flow_slug, scen_sub)] = gl_subs[best]
    return mapping


def _ordered_coverage(expected, performed):
    if not expected:
        return None
    pos = 0
    hits = 0
    for e in expected:
        for j in range(pos, len(performed)):
            if performed[j] == e:
                hits += 1
                pos = j + 1
                break
    return hits / len(expected)


def run():
    with gzip.open(RAW / "abcd_v1.1.json.gz", "rt") as f:
        data = json.load(f)
    convs = ([c for s in data.values() for c in s]
             if isinstance(data, dict) else data)
    guidelines = json.loads((RAW / "guidelines.json").read_text())

    by_flow = {}
    for c in convs:
        by_flow.setdefault(c["scenario"]["flow"], set()).add(
            c["scenario"]["subflow"])
    required = _subflow_map(guidelines, by_flow)

    out = []
    n_unmapped = 0
    coverages = []
    for c in convs:
        scen = c["scenario"]
        flow, subflow = scen.get("flow", ""), scen.get("subflow", "")
        personal = scen.get("personal", {})
        goal = GOAL_TEMPLATE.format(
            subflow=re.sub(r"_\d+$", "", subflow).replace("_", " "),
            flow=flow.replace("_", " "),
            name=personal.get("customer_name", "unknown"),
            level=personal.get("member_level", "unknown"))

        turns = []
        performed = []
        delexed = c["delexed"]
        for i, (spk, txt) in enumerate(c["original"]):
            if not str(txt).strip():
                continue
            turn = {"t": spk, "c": str(txt).strip()}
            if spk == "action" and i < len(delexed):
                tg = delexed[i].get("targets", [])
                if len(tg) > 3 and tg[2]:
                    turn["a"] = tg[2]
                    turn["v"] = tg[3] or []
                    performed.append(tg[2])
            turns.append(turn)

        expected = required.get((flow, subflow))
        if expected is None:
            n_unmapped += 1
            label = None
        else:
            cov = _ordered_coverage(expected, performed)
            if cov is None:
                n_unmapped += 1
                label = None
            else:
                coverages.append(cov)
                label = cov < 0.8  # has_failure
        out.append({"id": str(c["convo_id"]), "group_id": str(c["convo_id"]),
                    "goal_text": goal, "label": label, "turns": turns,
                    "flow": flow, "subflow": subflow})

    labeled = [c for c in out if c["label"] is not None]
    OUT.write_text(json.dumps(out))
    res = {"n_conversations": len(out),
           "n_labeled": len(labeled),
           "n_unmapped": n_unmapped,
           "failure_rate": (sum(c["label"] for c in labeled)
                            / max(len(labeled), 1)),
           "mean_action_coverage": (sum(coverages) / max(len(coverages), 1)),
           "n_mapped_subflow_pairs": len(required),
           "avg_turns": sum(len(c["turns"]) for c in out) / max(len(out), 1)}
    save_result("abcd_preprocess", res, dataset="abcd",
                params={"success_threshold": 0.8})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
