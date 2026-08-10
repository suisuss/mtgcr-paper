# Pre-registration: Round 18 (adversary-strength sweep to the selection ceiling)

**Partially superseded by R19.** The R19 frontier (`frontier_sweep.py`)
subsumes the *typicality* arm — its `res_2s` arm is the same residual swept
over the same budget grid. The **naive-head (`flat`) budget curve is this
round's alone**: no frontier arm carries it, so `adversary_sweep.py` and its
artifacts stay in the kit (results in `reproduction_record.md` section 22;
the ToolBench curve is the clean-provenance one, rerun 20260725T114923).

**Date closed:** 2026-07-24 (committed before any label-touching run).

The manipulability numbers throughout the paper are measured against a
best-of-16 selection adversary (draw 16 real pool turns, substitute the one
minimizing the head score; seed 42). The paper frames this as a *lower bound*
- selection over real turns is strictly weaker than a trained policy, so every
manipulability figure understates a determined attacker. A reviewer's fair
objection is the mirror image: the construction's robustness numbers (fixed
roles 0.00, conditional typicality -0.10) are *also* measured against that same
weak adversary, so "un-gameable" might be an artifact of a toothless attack.

R18 answers that objection *within the paper's own adversary class*. It does
not introduce a generative or trained adversary (that would spend the
methodological purity the paper relies on in section 7). It sweeps the one free
parameter of the existing selection attack - the candidate budget `n` - from
the published `n=16` up to the class ceiling: **exhaustive argmin over the
entire fold-train success-assistant pool**, the strongest adversary that
"selection over real turns" can ever be. If the robustness numbers are flat
across this sweep, "un-gameable by selection" is upgraded from "against
best-of-16" to "against the whole selection class up to its supremum."

**Scope disclaimer (frozen, must appear in any write-up):** a flat sweep closes
the *selection* threat completely; it does **not** close the *generative*
threat. An adversary confined to real pooled turns cannot synthesize a turn
that is simultaneously success-shaped and typical given the failing evidence.
The earned claim is "robust against the strongest selection adversary that
exists," not "robust, full stop."

## Frozen setup

Identical to Round 13 (`skeleton_roles.py`) and its ABCD/BGE replicas: MiniLM
(and BGE-large for the encoder arm), GroupKFold fold-0 TRAIN for head + ridge +
threshold, held-out FAILING conversations for every attack metric, candidate
pool = agentlike turns of SUCCESSFUL fold-train conversations, seed 42.
Manipulability = `(base_recall - recall_after_attack) / base_recall`, the
published definition. The R18 stage reconstructs this setup by importing the
Round-13 helpers unchanged; it does not modify any existing script or artifact.

**Budget grid:** `n in {1, 16, 64, 256, EXHAUSTIVE}`.
- `n=1` = random substitution (the drift control), carried through unchanged.
- `n=16` = the published operating point (nesting anchor, see guard below).
- `n=64, 256` = best-of-n over `n` i.i.d. pool draws, seed 42 (same mechanism,
  larger budget).
- `EXHAUSTIVE` = argmin over the *entire* pool at every agentlike turn
  (deterministic, seed-independent). This is the class ceiling.

**Nesting guard (fixed):** at `n=16`, seed 42, the sweep must reproduce the
published manipulability within `|Δ| <= 0.02` for both attackable arms
(`flat` 0.50, `typicality` -0.097). A larger discrepancy means the sweep
diverged from the published attack and the run is void, not a finding.

**Arms.** Only `flat` (naive head) and `typicality` (conditional-typicality
residual) are swept, because they are the only representations whose feature
vector contains an adversary-written channel. The invariant arms (`skeleton`,
`drop_assistant`, `drop_assistant_skel`) have manipulability 0 for **every**
`n` *by construction* - the agentlike block is not in the feature, so no budget
can move it. This is asserted, not measured; spending compute to "confirm" a
structural zero would be theatre.

**Corpora / encoders.** Primary: ToolBench / MiniLM (the headline Table 3
result and the -0.10 at stake). Replication, pre-registered, run if the primary
completes: ABCD / MiniLM (`skeleton_roles_abcd`) and ToolBench / BGE-large
(`skeleton_roles_bge`) - the two other cells of the paper's Table 7 robustness
surface. Same budget grid, same guard against each cell's published `n=16`
number.

## Direction A (confirmatory): the naive head degrades monotonically

`flat` manipulability is non-decreasing in `n` (a larger budget is a superset
of a smaller one under argmin over i.i.d. draws; EXHAUSTIVE dominates all).

**Prediction P1 (expected):** `flat` manipulability rises from ~0.50 (n=16)
toward a higher ceiling at EXHAUSTIVE. This *strengthens* the existing "the
naive head is broken by the weakest adversary" claim - it is confirmatory and
has no kill condition. A *decrease* with increasing `n` would indicate a bug
(the guard should catch gross divergence first), not a result.

## Direction B (the test): does typicality's anti-gameability survive the ceiling?

The published `typicality` manipulability is -0.097: the best-of-16 attack
*raises* the alert rate, because success-shaped narration drawn from the pool
is atypical atop a failing record, so the residual grows and the head fires
more. The question R18 exists to answer: is that an artifact of a weak
adversary, or does it hold - or strengthen - as the adversary approaches the
selection ceiling?

Mechanistically it should hold: minimizing the head score on the typicality arm
means picking the *most typical* real pool turn given the fixed failing
context, which cannot suppress evidence it does not touch. A stronger adversary
searches this harder and may drive manipulability *more* negative.

**Prediction P2 (keep):** `typicality` manipulability stays `<= +0.05` at
EXHAUSTIVE (anti-gameability holds or strengthens). Then the construction's
central re-admission move is robust across the whole selection class, and the
sweep row can be added to the robustness section.

**Kill P2:** `typicality` manipulability at EXHAUSTIVE `> +0.10` (crosses from
anti-gameable to materially gameable). Then the -0.10 was an artifact of
best-of-16, the "conditional typicality is anti-gameable" claim is scoped only
to weak adversaries, and the paper must rescope it - report the sweep as the
finding that breaks it, not bury it.

**Grey zone (`+0.05` to `+0.10`):** report the curve, weaken "anti-gameable" to
"not materially gameable," keep the fixed-role result (unaffected, structural)
as the load-bearing robustness claim.

## Negative-result framing (frozen)

- P1 climbs, P2 holds -> the paper gains a sweep confirming robustness across
  the selection class; add one row/panel, keep all claims, add the generative
  scope disclaimer above.
- P2 killed -> report the break plainly; the fixed-role reward (structural 0)
  survives regardless and becomes the sole un-gameability claim; the typicality
  re-admission is demoted to "recovers accuracy, anti-gameable only under weak
  selection."
- The invariant-arm structural zeros are asserted, never presented as a swept
  measurement.
