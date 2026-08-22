# PENUMBRA — LIMITATIONS

Every limitation here is one we have actually hit, with the measurement that exposed
it. This document exists so that no number PENUMBRA prints gets quoted as more than
it is.

**Updated:** 2026-08-21

---

## 1. The measurement is open-loop action divergence, not task success

The policy's predicted actions never move anything, because the robot is a recording.
On every control step PENUMBRA echoes the **logged** action back through
`set_executed_step_json`, so the observation history stays pinned to the real
trajectory in both the baseline and the perturbed condition.

This is deliberate. Holding the trajectory fixed is what makes the comparison
single-variable: baseline and perturbed differ only in the pixels of
`exterior_image_1_left`. A closed-loop test would confound perception error with
control error and could not attribute a failure to either.

**What it licenses:** "this visual change shifted the policy's action distribution."
**What it does not license:** any claim about task success, closed-loop outcome, or
what would happen on real hardware. Whether open-loop divergence predicts closed-loop
failure is an open question and needs a robot we do not have.

---

## 2. The policy is stochastic, and its noise is larger than most perturbations

This is the single most important empirical fact in the project, and it invalidated
our first experimental design.

Repeated rollouts on the **identical unperturbed** episode differ substantially.
Measured across five independent calibrations of 4–6 runs each:

| calibration | runs | pairs | mean pairwise joint L2 (rad) | sigma |
|---|---|---|---|---|
| 1 | 4 | 6 | 0.0963 | 0.0125 |
| 2 | 4 | 6 | 0.0905 | 0.0059 |
| 3 | 6 | 15 | 0.1032 | 0.0138 |
| 4 | 6 | 15 | 0.1040 | 0.0162 |
| 5 | 6 | 15 | 0.0887 | 0.0095 |

For scale: the policy's mean absolute error against the **logged ground-truth action**
is 0.020–0.046 rad per joint. Its run-to-run variability is larger than its tracking
error.

Unperturbed runs also disagree about the **gripper decision** on up to 1 of 12 control
steps without any perturbation at all.

Consequences, all of them enforced in code:

- A single baseline-vs-perturbed comparison is not evidence. The unit of evidence is a
  **group** of rollouts and a permutation test over group labels
  (`src/penumbra/evaluation/grouptest.py`).
- `sigma` itself is unstable: the five estimates above span 0.0059–0.0162, a factor of
  2.7. Any threshold expressed as "mean + k·sigma" inherits that instability, which is
  why the group test does not depend on sigma at all.
- A gripper flip is only a finding if it exceeds what the unperturbed runs already do.

---

## 3. X2 has no seed — the same fault is a different perturbation each time

`set_seed` does not exist in `xmax/x2`'s OpenAPI schema. This is not a documentation
gap; the command is absent.

We did not infer the consequence, we hit it. `specular_floor` at strength 1.0, the
identical prompt on the identical frames:

| run | geometry drift | structure | temporal ratio | gate |
|---|---|---|---|---|
| 1 | 0.0015 | 0.889 | **1.92** | VALID |
| 2 | 0.0017 | 0.827 | **4.14** | **REJECTED** |

The same fault at the same strength passed the validity gate once and was rejected the
next time, because the model produced visibly more temporal flicker on the second run.

Consequences:

- A reported margin is **the lowest break observed**, not a deterministic threshold.
- Bisection over the strength ladder inherits this: a rung's verdict is a sample.
- `tools/x2_stability.py` exists to measure the spread deliberately rather than
  discovering it by accident.

---

## 4. Strength is an ordinal ladder we constructed, not a model parameter

X2 exposes no numeric strength control. The only channel to intensity is prompt
language, so `FaultSpec` defines an ordered ladder of prompts and calls that the
strength axis.

- The axis is **ordinal by construction**. The distance between rung 0.4 and rung 0.6
  is not physically meaningful, and is not comparable across faults.
- **Monotonicity is a hypothesis, not a property.** It is checked and reported, never
  assumed. It has already been observed to fail: in the `backlight_glare` sweep, RMSE
  rose monotonically across rungs (0.064 → 0.087) while joint divergence did not
  (0.109 → 0.096 → 0.100 → 0.087).
- Because of that, `--mode sweep` is the default and `--mode bisect` records that it
  assumed monotonicity.
- The comparable quantity across faults is the **measured perceptual distance**, which
  is reported alongside every rung.

---

## 5. The validity gate bounds geometry corruption; it does not eliminate it

SEAM v1 uses dense optical flow, gradient-orientation agreement, and a temporal motion
ratio. Its thresholds are measured, not inherited — `tools/calibrate_gate.py`, on the
actual episode:

| transformation | median drift |
|---|---|
| identity | 0.00000 |
| gamma 0.5 | 0.00037 |
| brightness ×1.5 + 40 | 0.00048 |
| X2 `specular_floor` @ 1.0 (appearance only) | 0.00154 |
| 2 px global shift | 0.00272 |
| **threshold** | **0.00550** |
| 4 px global shift | 0.00545 |
| 32 px global shift | 0.04356 |

The research document proposed 0.05. That would have rejected only shifts larger than
about 36 pixels — it would have passed nearly any corruption.

What the gate still cannot see:

- **The aperture problem.** A displaced region with no internal texture is invisible to
  any dense-flow method. Pinned by `test_gate_is_documented_as_blind_to_the_aperture_problem`.
- **Small localised motion.** A patch covering roughly a tenth of the frame can slide
  16 px and pass both the median and p90 statistics. Pinned by
  `test_gate_is_blind_to_a_small_region_moving`.
- **Stationary hallucination.** Content cleanly edited into or out of a *static* region
  produces almost no flow. This matters most for exactly the experiment that most needs
  it — object removal — which is why the PROVENANCE result is framed as a hypothesis
  generator rather than a verdict.
- No depth, no structure-from-motion, no segmentation. Any of them would strengthen it.

Sub-4 px drift is permitted deliberately: the policy consumes downscaled frames, so
4 px on a 640-wide source is on the order of 1 px at its input, below what could
plausibly drive an 8-DoF action change on its own. Every report prints the measured
drift and its pixel equivalent so a reader can apply a stricter bar.

---

## 6. Frame correspondence is reconstructed, not guaranteed

X2 is a streaming generative model: pushing N frames does not return N. Observed 60 in
→ 54 out. PENUMBRA pads with a lead-in and tail, discards the warm-up proportionally,
resamples onto the source timeline, and refines an integer offset by maximising
normalised cross-correlation. The chosen offset and its score are recorded.

This is a reconstruction. It is checked (the gate would reject a badly misaligned
stream) but it is not a guarantee of frame-exact correspondence.

Mitigating measurement: a deliberate 3-frame desynchronisation of the source against
itself produces a median drift of 0.00011 — far below the threshold — because the scene
moves slowly at 15 fps. Small temporal misalignment therefore contributes little to the
geometry statistic, in either direction.

---

## 7. Statistical power is bounded by the group sizes, hard

With N baseline and M perturbed rollouts, the permutation test cannot report a p-value
below the number of distinct group labellings allows. When N = M the statistic is
invariant under swapping the labels, so arrangements come in pairs and the floor is
**2 / C(N+M, N)**, not 1 / C(N+M, N):

| N = M | floor on p |
|---|---|
| 3 | 0.100 — cannot reach alpha = 0.05 at all |
| 4 | 0.029 |
| 5 | 0.008 |
| 6 | 0.002 |

`min_achievable_p` is reported next to every p-value. A sweep also applies a Bonferroni
correction across rungs, because one test per rung is a family of tests.

---

## 7b. The apparatus has sensitivity, but only to very large changes

The positive control (`runs/_probe/positive_control.json`) blacks out the perturbed
camera view entirely and asks whether the policy notices. It does: p = 0.0162,
Cohen's d = +1.32. So nulls measured on this view are informative rather than vacuous.

But the magnitude bounds every null we report:

| condition | RMSE from source | joint p |
|---|---|---|
| generative `target_removal` @ 1.0 | 0.0604 | 0.7167 |
| classical `color_shift` matched | 0.0618 | 0.9196 |
| classical `blur_noise` matched | 0.0606 | 0.1102 (gate REJECTED) |
| generative `target_removal` @ 1.0 (earlier run) | 0.1115 | 0.2575 |
| **positive control: view destroyed** | **0.4521** | **0.0162** |

A null at RMSE 0.06 means "no effect detectable at roughly one seventh of the distance
that does produce a detectable effect." It does **not** mean the policy is robust.

Two further bounds from the same run:

- Even destroying the view produced **no gripper decision change** (gripper p = 1.0000,
  flip rate 0.028 → 0.014). The discrete decision channel appears insensitive to this
  view, so any finding routed through gripper flips on this channel would need
  extraordinary evidence.
- Even destroying the view only reached p = 0.0162, about 7× the design's own floor of
  0.0022. At N = M = 6 the effect is real but not overwhelming.

Where detectability begins between RMSE 0.06 and 0.45 is being measured; until it is,
"how large must a perturbation be before this apparatus could see it?" is unanswered,
and that number is a precondition for interpreting any margin.

---

## 7c. Our own decision rule produced a false positive

The first break criterion combined a significance-tested joint divergence with a bare
1.5× ratio on gripper flip rate. The ratio route declared a **BREAK at p = 0.7167**, on
a flip rate moving 0.000 → 0.028 — about one flip across 432 step-comparisons.

Fixed: the gripper flip rate now has its own permutation test, both statistics are held
to alpha/2, and `test_a_bare_flip_rate_ratio_is_not_enough_to_declare_a_break` feeds the
comparator null data and asserts it finds nothing. Both `BREAK` labels in
`runs/compare-target_removal-20260821-112831` were produced by the old rule and are
superseded by the recorded p-values.

---

## 7d. The classical arm was exempt from the gate it competes against

`blur_noise`, matched on perceptual distance, produced the largest divergence of the
three arms (d = +0.45) — with **9.28× the source's frame-to-frame motion**. Per-frame
independent noise is temporally incoherent in a way the recording never was, and the
gate rejected it. But the classical arm had been coded to never reject, so it would
have counted as a classical "win" against a generative arm being held to that standard.

Fixed: both arms face the same gate; positive controls are explicitly exempt because
their purpose is to exceed any admissible fault.

---

## 8. Sample size

Everything measured so far is **one episode**, a 96-frame window, one policy, one
camera view. This is a demonstration of method. No number produced here is a robustness
benchmark for `cosmos-nano-policy-droid`, and none should be quoted as a property of
that policy in general.

---

## 9. Not yet verified

| Question | Status |
|---|---|
| Session concurrency limit per key | **UNVERIFIED** — would set the search budget |
| SANA-Streaming as an independent corroborator | **UNVERIFIED** — not yet probed |
| Whether findings transfer to closed-loop or real hardware | **OUT OF SCOPE** — needs a robot |
| Whether the policy attends to `exterior_view_1` at all | **OPEN** — see PROVENANCE |

---

## 9b. Experiment length was capped by the execution environment

Long-running background work was terminated at roughly five minutes, repeatedly and
from outside the process. This shaped what could be measured:

- Multi-rung ladders had to be split into single rungs.
- Group sizes were reduced from N = M = 6 to N = M = 5 for later runs, raising the
  attainable p-floor from 0.0022 to 0.0079. Still well under alpha/test = 0.025, so no
  conclusion depends on it, but the design is thinner than intended.
- One ladder was killed mid-rung and lost twelve completed policy rollouts, which
  prompted the incremental-save fix (§10).
- A corrected three-arm comparison could not be re-run: three attempts were each
  terminated short of the ~7 minutes it needs.

Where a re-run was impossible, verdicts were **re-derived in place from the run's own
recorded statistics** rather than repeated. `runs/compare-target_removal-20260821-112831`
carries `superseded`, `status_rederived` on each arm, and `broke_rederived` on the
top-level verdict, each naming the old value, the p-value it was re-derived from, and
the fact that the gripper route was untested in that run. **No measurement was
altered** — only labels that the superseded decision rule had produced.

That record's original top-level `verdict.broke` was `true` while its own
`verdict.detail.broke` was `false`; the two disagreed because the flag came from the
buggy rule and the detail from the sound one. Both now read `false`.

---

## 10. Reactor sessions can wedge, and did

A policy rollout stalled indefinitely after a successful connect, hanging an experiment
that had already spent six baseline rollouts. Root cause: `send_command` and
`publish_track` were awaited without a timeout. Fixed — every Reactor operation now has
a bound (`ReactorSession.DEFAULT_OP_TIMEOUT`), and rollouts retry on a fresh session,
with the retry count carried into the cost record rather than hidden.
