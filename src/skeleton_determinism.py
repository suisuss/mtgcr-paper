"""Round-11 Arm 1: skeleton-to-outcome determinism. If identical action
skeletons yield identical outcomes, the skeleton determines the outcome and
a policy cannot move it toward "looks successful" without moving it toward
"is successful." PREREGISTRATION-R11.md.

Group conversations by their set of distinct action slugs (ABCD) / tool
services (ToolBench); among groups shared by >= 5 conversations, report
outcome purity vs base, coverage, and the both-classes ambiguous fraction.

    python skeleton_determinism.py abcd
    python skeleton_determinism.py toolbench
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lib.campaign import save_result

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "data" / "embeddings"
MIN_GROUP = 5


def _skeletons(dataset):
    meta = json.loads((EMB / f"{dataset}_meta.json").read_text())
    y = np.array(meta["has_failure"], dtype=int)
    if dataset == "abcd":
        an = meta["action_names"]
        keys = [frozenset(a for a in conv if a) for conv in an]
    else:  # toolbench: services from conversations.json
        conv = json.loads((REPO / "data" / "raw" / "toolbench"
                           / "conversations.json").read_text())
        keys = [frozenset(t["metadata"].get("service", "")
                          for t in c["turns"]
                          if t["turn_type"] == "tool_call")
                for c in conv]
    return keys, y


def run(dataset):
    keys, y = _skeletons(dataset)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        if k:
            groups[k].append(i)

    base = float(max(y.mean(), 1 - y.mean()))
    shared = {k: idx for k, idx in groups.items() if len(idx) >= MIN_GROUP}
    n_in_shared = sum(len(v) for v in shared.values())

    purity_num = 0
    ambiguous_convs = 0
    for k, idx in shared.items():
        labels = y[idx]
        maj = float(max(labels.mean(), 1 - labels.mean()))
        purity_num += maj * len(idx)
        if 0 < labels.mean() < 1:
            ambiguous_convs += len(idx)
    purity = purity_num / max(n_in_shared, 1)

    res = {"dataset": dataset, "n": int(len(y)),
           "base_rate": float(y.mean()), "base_purity": base,
           "n_unique_skeletons": len(groups),
           "n_shared_groups": len(shared),
           "coverage_in_shared": n_in_shared / max(len(y), 1),
           "outcome_purity_shared": float(purity),
           "determinism_gap": float(purity - base),
           "ambiguous_fraction_of_shared": ambiguous_convs
           / max(n_in_shared, 1)}
    save_result(f"skeleton_determinism_{dataset}", res, dataset=dataset,
                params={"min_group": MIN_GROUP})
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    run(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
