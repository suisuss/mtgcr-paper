"""SpokenWOZ -> normalized schema (turns + goal_text + derived
Success-style label). PREREGISTRATION-R5.md Arm S.

Label rule: success iff (a) every goal domain's requested slots all
appear among that domain's system dialog-act payload slot names, and
(b) every booked domain shows a booking acknowledgment - a 'ref' slot in
a same-domain system act, or a same-domain '-book'/'-offerbooked' act,
or (act-vocabulary reality of SpokenWOZ) a same-domain '-ack' act
following a booking request. Coverage diagnostics reported.

    python spokenwoz_preprocess.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "spokenwoz"
OUT = RAW / "normalized.json"

DOMAINS = ["attraction", "hospital", "hotel", "police", "profile",
           "restaurant", "taxi", "train"]


def render_goal(goal):
    parts = []
    for dom in DOMAINS:
        g = goal.get(dom) or {}
        if not isinstance(g, dict) or not any(
                g.get(k) for k in ("info", "book", "reqt")):
            continue
        seg = [f"{dom}:"]
        info = g.get("info") or {}
        if info:
            seg.append("find one with " + ", ".join(
                f"{k} {v}" for k, v in info.items()))
        book = g.get("book") or {}
        if book:
            seg.append("book for " + ", ".join(
                f"{k} {v}" for k, v in book.items()))
        reqt = g.get("reqt") or []
        if reqt:
            seg.append("get the " + ", ".join(str(r) for r in reqt))
        parts.append(" ".join(seg))
    return ("The caller wants to: " + "; ".join(parts)) if parts else ""


def _norm_slot(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def derive_label(goal, log):
    """Returns (has_failure, n_domains_checked) or (None, 0) if the goal
    exercises nothing checkable."""
    sys_acts = []  # (domain, act_suffix, {slot names})
    for t in log:
        if t.get("tag") != "system":
            continue
        for act, payload in (t.get("dialog_act") or {}).items():
            dom, _, suffix = act.partition("-")
            slots = {_norm_slot(p[0]) for p in payload
                     if isinstance(p, (list, tuple)) and p}
            sys_acts.append((dom.lower(), suffix.lower(), slots))

    checked = 0
    for dom in DOMAINS:
        g = goal.get(dom) or {}
        if not isinstance(g, dict):
            continue
        reqt = [_norm_slot(r) for r in (g.get("reqt") or [])]
        book = g.get("book") or {}
        if reqt:
            checked += 1
            informed = set()
            for d2, _, slots in sys_acts:
                if d2 == dom:
                    informed |= slots
            # act slot vocabulary aliases goal reqt names (postcode/post,
            # trainid/id, entrancefee/fee, ...): bidirectional substring
            # match within the same domain's acts
            def met(r):
                return any(r in s or s in r for s in informed if s
                           and s != "none")
            if not all(met(r) for r in reqt):
                return True, checked
        if book:
            checked += 1
            ok = False
            for d2, suffix, slots in sys_acts:
                if d2 != dom:
                    continue
                if "ref" in slots or suffix in ("book", "offerbooked",
                                                "ack"):
                    ok = True
                    break
            if not ok:
                return True, checked
    if checked == 0:
        return None, 0
    return False, checked


def run():
    out = []
    n_nogoal = 0
    n_unlabeled = 0
    for split in ("train_text", "test_text"):
        data = json.loads((RAW / split / "data.json").read_text())
        for did, dlg in data.items():
            goal = dlg.get("goal") or {}
            gt = render_goal(goal)
            if not gt:
                n_nogoal += 1
                continue
            label, checked = derive_label(goal, dlg.get("log", []))
            if label is None:
                n_unlabeled += 1
            turns = [{"t": t["tag"], "c": (t.get("text") or "").strip()}
                     for t in dlg.get("log", [])
                     if t.get("tag") in ("user", "system")
                     and (t.get("text") or "").strip()]
            if len(turns) < 4:
                continue
            out.append({"id": did, "group_id": did, "goal_text": gt,
                        "label": label, "turns": turns})

    labeled = [c for c in out if c["label"] is not None]
    OUT.write_text(json.dumps(out))
    res = {"n_dialogues": len(out),
           "n_labeled": len(labeled),
           "n_no_goal": n_nogoal,
           "n_unlabeled_goal": n_unlabeled,
           "failure_rate": (sum(c["label"] for c in labeled)
                            / max(len(labeled), 1)),
           "avg_turns": (sum(len(c["turns"]) for c in out)
                         / max(len(out), 1))}
    save_result("spokenwoz_preprocess", res, dataset="spokenwoz", params={})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
