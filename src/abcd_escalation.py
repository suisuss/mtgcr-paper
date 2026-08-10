"""Round-4 Arm A: ABCD escalation density - the gate for the escalation
early-warning study. Counting rule fixed in PREREGISTRATION-R4.md;
viability bar: >= 300 conversations with a genuine escalation event.

The canonical escalation action discovered at preprocess time is
notify-team with value 'manager'; the pattern sweep and utterance-level
counts are reported alongside, with a seed-42 sample of 50 utterance hits
persisted for the genuineness spot-check.

    python abcd_escalation.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
NORM = REPO / "data" / "raw" / "abcd" / "normalized.json"
PATTERN = re.compile(r"escalat|manager|supervisor|handoff|transfer", re.I)
SEED = 42


def run():
    convs = json.loads(NORM.read_text())
    n = len(convs)

    conv_action_esc = []       # convs with notify-team -> manager
    conv_action_other = []     # notify-team to non-manager teams
    utter_hits = []            # (conv_id, speaker, text)
    subflow_hits = set()
    for c in convs:
        act_esc = False
        act_other = False
        if PATTERN.search(c["subflow"]):
            subflow_hits.add(c["subflow"])
        for t in c["turns"]:
            if t["t"] == "action" and t.get("a") == "notify-team":
                vals = [str(v).lower() for v in t.get("v", [])]
                if any("manager" in v for v in vals):
                    act_esc = True
                else:
                    act_other = True
            if t["t"] in ("customer", "agent") and PATTERN.search(t["c"]):
                utter_hits.append((c["id"], t["t"], t["c"][:160]))
        conv_action_esc.append(act_esc)
        conv_action_other.append(act_other)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(utter_hits),
                            size=min(50, len(utter_hits)), replace=False)
    sample = [utter_hits[i] for i in sorted(sample_idx)]

    n_esc = int(np.sum(conv_action_esc))
    labeled = [c for c in convs if c["label"] is not None]
    esc_in_labeled = sum(1 for c in labeled
                         for t in [None]
                         if any(tt.get("a") == "notify-team"
                                and any("manager" in str(v).lower()
                                        for v in tt.get("v", []))
                                for tt in c["turns"]))

    res = {"n_conversations": n,
           "conv_with_manager_notification": n_esc,
           "conv_with_other_team_notification": int(np.sum(conv_action_other)),
           "manager_notifications_in_labeled_subset": esc_in_labeled,
           "utterance_pattern_hits": len(utter_hits),
           "conv_with_utterance_hit": len({h[0] for h in utter_hits}),
           "subflow_name_matches": sorted(subflow_hits),
           "viability_bar": 300,
           "viable": bool(n_esc >= 300),
           "spot_check_sample": [{"conv": c, "speaker": s, "text": t}
                                 for c, s, t in sample]}
    save_result("abcd_escalation", res, dataset="abcd",
                params={"pattern": PATTERN.pattern, "seed": SEED})
    print(json.dumps({k: v for k, v in res.items()
                      if k != "spot_check_sample"}, indent=1))
    return res


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
