"""Experiment-script contract for src/.

Every stage script is an importable module exporting up to three
callables (importing a module never executes work):

    run(**params) -> dict
        The stage's entry point: loads its data, computes, persists
        through save_result() below, and returns the results dict.

    run_sample(max_samples, **params) -> dict
        Optional. Alias for run(max_samples=max_samples, **params),
        built with make_run_sample() below. Sampled in-process runs are
        EPHEMERAL: they must not call save_result and must bypass
        checkpoint caches (a full-N cache entry would otherwise be
        silently reused, and sampled numbers must never enter the
        canonical registry). Only add this if your stages support a
        max_samples-style sandbox mode.

    main(argv=None) -> int
        Thin CLI shim, called from the `if __name__ == "__main__"`
        guard - pipeline specs and run.sh invoke
        `python <script>.py ...` unchanged regardless of what run()
        looks like.
"""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent.parent


def make_run_sample(run):
    """Standard run_sample factory: ephemeral capped invocation."""
    def run_sample(max_samples, **params):
        return run(max_samples=max_samples, **params)
    run_sample.__doc__ = (f"run(max_samples=N, ...) alias; ephemeral - "
                          "no save_result, no checkpoint writes. "
                          "See lib/campaign.py.")
    return run_sample


class Checkpoint:
    """The house checkpoint pattern, in one place.

    One dir per script under data/experiment-caches/<name>/, one npz per
    tag, atomic tmp+rename writes.

    Staleness: a PRODUCER constructs Checkpoint(name, script=__file__).
    save() stamps the producing script's logic digest into
    <dir>/producer.json (idempotently, before writing the npz); when the
    script later changes, load() MISSes any entry stamped under a
    different digest (stale-but-warm entries are never silently replayed)
    and the next save() clears the stale npz files and re-stamps. load()
    is read-only. A dir predating stamping (no producer.json) is adopted:
    reads are allowed and the next save() writes the stamp without
    wiping. READERS of another script's cache construct Checkpoint(name)
    with no script: passive reads that never invalidate, since freshness
    is the producer's job (pipeline depends_on ordering runs the
    producer first).
    """

    def __init__(self, name, root=None, script=None):
        self.dir = Path(root) / name if root else \
            WORK.parent / "data" / "experiment-caches" / name
        self._script = Path(script) if script else None

    def _stamp_file(self):
        return self.dir / "producer.json"

    def _source_digest(self):
        """Digest of the script's LOGIC, used only for cache invalidation:
        a hash of the parsed AST, so comment- and formatting-only edits
        leave it unchanged and do not wipe cached checkpoints. Only a
        change to the code itself (or a docstring) re-invalidates. Falls
        back to a byte hash if the source does not currently parse (e.g.
        mid-edit). This is deliberately separate from the exact file
        sha256 recorded in each artifact's _provenance for external
        verifiability (lib/provenance.py::file_sha256)."""
        import ast
        import hashlib
        src = self._script.read_bytes()
        try:
            payload = ast.dump(ast.parse(src)).encode()
        except SyntaxError:
            payload = src
        return hashlib.sha256(payload).hexdigest()

    def _stamped_sha(self):
        stamp = self._stamp_file()
        if stamp.exists():
            return json.loads(stamp.read_text()).get("source_digest")
        return None

    def _ensure_stamp(self):
        """Producer-only, idempotent: stamp the dir with the current
        script digest, clearing stale npz entries when a different
        digest was stamped. No-op for readers (script=None)."""
        if self._script is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        cur = self._source_digest()
        stamped = self._stamped_sha()
        if stamped == cur:
            return
        if stamped is not None:            # producer logic changed - wipe stale
            for f in self.dir.glob("*.npz"):
                f.unlink()
            print(f"[cache] {self.dir.name}: producer script changed - "
                  "cleared stale checkpoints", flush=True)
        stamp = self._stamp_file()
        tmp = stamp.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"script": self._script.name, "source_digest": cur}, indent=1))
        tmp.rename(stamp)

    def load(self, tag):
        # a producer never replays a cache stamped by a different script
        # version; a missing stamp (legacy dir) is adopted, not refused.
        if self._script is not None:
            stamped = self._stamped_sha()
            if stamped is not None and stamped != self._source_digest():
                return None
        f = self.dir / f"{tag}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return {k: z[k] for k in z.files}
        return None

    def save(self, tag, arrays):
        self._ensure_stamp()
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"{tag}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(f)


def save_result(name, results, *, group="experiments", dataset="default",
                config="default", variant="none", params=None,
                tags=(), n_samples=None, duration=None, also_flat=True,
                script=None):
    """Persist a stage result: flat file (what tables/figures read) +
    provenance-stamped registry copy under
    data/results/<group>/<name>/<dataset>/<config>/<name>_<ts>.json.

    script: repo-relative path of the producing script; auto-detected
    from the caller when omitted.
    """
    import inspect
    import os
    from lib.ioutils import atomic_json_dump
    from lib.results import ResultsRegistry

    if also_flat:
        atomic_json_dump(WORK.parent / "data" / "results" / group
                         / f"{name}.json", results, indent=1)
    if script is None:
        caller = inspect.stack()[1].filename
        try:
            script = os.path.relpath(caller, WORK)
        except ValueError:
            script = caller
    key = ResultsRegistry().save(
        experiment=f"{group}/{name}", config=config, dataset=dataset,
        variant=variant, params=params or {}, results=results,
        tags=list(tags), script=script, n_samples=n_samples,
        duration=duration, filename_stem=name)
    print(f"registered {key}", flush=True)
    return key


def verify_provenance(group="experiments", results_root=None):
    """The reproduction gate as a callable: every canonical artifact must
    come from a clean pinned checkout. Returns a list of offending
    paths (empty = gate passes)."""
    root = Path(results_root) if results_root else \
        WORK.parent / "data" / "results"
    bad = []
    for f in sorted((root / group).glob("*/**/*.json")):
        p = json.loads(f.read_text()).get("_provenance", {})
        if p.get("git_dirty") is not False or not (
                p.get("git_commit") and p.get("script_sha256")):
            bad.append(str(f))
    return bad


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        offenders = verify_provenance()
        for f in offenders:
            print(f"DIRTY/UNSTAMPED: {f}")
        print(f"{'FAIL' if offenders else 'PASS'}: "
              f"{len(offenders)} offending artifact(s)")
        sys.exit(1 if offenders else 0)
    print("usage: python -m lib.campaign verify")
    sys.exit(64)
