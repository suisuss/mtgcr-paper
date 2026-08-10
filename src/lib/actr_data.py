"""Loaders for the actr-paper kit's committed artifacts.

All spike stages read data through here. The sibling actr-paper checkout
is treated as pinned external data: its HEAD must match the commit in
KIT_PROVENANCE.json (dataset_pins.actr_paper_commit) or every loader
refuses to run - the spike numbers are only meaningful against the
artifacts the flagship actually published.

USS corpora ship as one npz (embeddings [n_turns_total, 384] L2-normalized
MiniLM + offsets [n_conv+1]) plus a meta json (roles, turn_ratings,
overall, overall_mean, overall_median, n_turns - all per-conversation
lists). Raw turn text is parsed from the USS dumps (tab-separated
ROLE/text/action/ratings lines, blank-line-separated conversations) and
verified against the meta's conversation count and per-conversation turn
counts.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def _pins():
    return json.loads((REPO / "KIT_PROVENANCE.json").read_text())["dataset_pins"]


def actr_root():
    pins = _pins()
    root = (REPO / pins["actr_paper_path"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"actr-paper checkout not found at {root}")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True
                          ).stdout.strip()
    if head != pins["actr_paper_commit"]:
        raise RuntimeError(
            f"actr-paper HEAD {head[:12]} != pinned "
            f"{pins['actr_paper_commit'][:12]}; refusing to run on drifted "
            "data (update the pin deliberately if the drift is intended)")
    return root


def uss_dir():
    """USS-SGD/MWOZ MiniLM embeddings + meta, vendored into this kit so the
    manuscript's Table 9 (shape_reward, endpoint_shape) is self-contained; the
    sibling actr-paper checkout is no longer needed for the paper's numbers.
    PRISM and the raw USS dumps (spike stages only) still load from the sibling
    via actr_root()."""
    return REPO / "data" / "embeddings"


def load_uss_meta(corpus):
    return json.loads((uss_dir() / f"uss_{corpus}_meta.json").read_text())


def load_uss(corpus):
    """corpus in {'sgd', 'mwoz'} -> dict with per-conversation embedding
    arrays, roles, labels."""
    z = np.load(uss_dir() / f"uss_{corpus}_minilm.npz")
    emb, off = z["embeddings"], z["offsets"]
    meta = load_uss_meta(corpus)
    n = len(off) - 1
    convs = []
    for i in range(n):
        e = emb[off[i]:off[i + 1]]
        roles = np.array(meta["roles"][i])
        assert len(e) == len(roles) == meta["n_turns"][i]
        convs.append({
            "emb": e,
            "roles": roles,
            "overall_median": float(meta["overall_median"][i]),
            "overall_mean": float(meta["overall_mean"][i]),
        })
    return convs


def load_uss_text(corpus):
    """Raw turn texts aligned to load_uss ordering: list of list-of-str.

    Alignment is verified against the meta file (conversation count and
    per-conversation turn counts); any mismatch aborts.
    """
    root = actr_root()
    fname = {"sgd": "SGD.txt", "mwoz": "MWOZ.txt"}[corpus]
    raw = (root / "data" / "raw" / "uss_repo" / "dataset" / fname).read_text()
    convs, cur = [], []
    for line in raw.split("\n"):
        if not line.strip():
            if cur:
                convs.append(cur)
                cur = []
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        if parts[1].strip() == "OVERALL":
            # USS dumps end each conversation with a pseudo-turn carrying
            # the conversation-level rating; it is not a real turn.
            continue
        cur.append({"role": parts[0].strip(), "text": parts[1]})
    if cur:
        convs.append(cur)

    meta = json.loads((root / "data" / "embeddings"
                       / f"uss_{corpus}_meta.json").read_text())
    n_meta = len(meta["n_turns"])
    if len(convs) != n_meta:
        raise RuntimeError(f"{fname}: parsed {len(convs)} conversations, "
                           f"meta has {n_meta}")
    for i, c in enumerate(convs):
        if len(c) != meta["n_turns"][i]:
            raise RuntimeError(f"{fname} conv {i}: parsed {len(c)} turns, "
                               f"meta says {meta['n_turns'][i]}")
        roles = [t["role"] for t in c]
        if roles != meta["roles"][i]:
            raise RuntimeError(f"{fname} conv {i}: role sequence mismatch")
    return [[t["text"] for t in c] for c in convs]


def prism_path():
    return actr_root() / "data" / "raw" / "prism" / "conversations.jsonl"


BAD_MAX = 2.5    # overall_median <= BAD_MAX  -> bad band
GOOD_MIN = 3.5   # overall_median >= GOOD_MIN -> good band


def band(conv):
    """1 = bad, 0 = good, None = middle (excluded from band contrasts)."""
    m = conv["overall_median"]
    if m <= BAD_MAX:
        return 1
    if m >= GOOD_MIN:
        return 0
    return None
