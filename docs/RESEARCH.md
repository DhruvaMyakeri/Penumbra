# PENUMBRA — RESEARCH DESIGN

**Updated:** 2026-08-21

This document states the experiment PENUMBRA runs, what would falsify it, and what its
results do and do not license. Findings live in §7 and carry the run IDs that produced
them. `docs/LIMITATIONS.md` is the companion: everything here is bounded by what is
listed there.

---

## 1. Question

> **Can Reactor-generated visual interventions reveal statistically meaningful policy
> vulnerabilities that conventional augmentation misses?**

Decomposed into three claims, each independently falsifiable:

**H1 (effect).** A generative appearance perturbation of a real robot episode shifts a
VLA policy's action distribution beyond that policy's own run-to-run variability.

**H2 (reachability).** At **matched perceptual distance**, generative perturbation
produces effects that scene-blind pixel augmentation does not. This is the load-bearing
claim: if false, the generative machinery buys nothing over Albumentations.

**H3 (provenance).** The policy's behaviour is conditioned on the task-relevant object
being visible in the perturbed camera view. Falsified if removing the object leaves the
action distribution unchanged.

H1 is a precondition for H2. H3 is independent and can be answered either way.

---

## 2. Design

A within-episode, single-variable intervention with repeated measures.

```
                      exterior_image_1_left  <- the ONLY thing that changes
real DROID episode ->  exterior_image_2_left  } held fixed
                       wrist_image_left       } held fixed
                       joint_positions (7)    } held fixed
                       gripper_position (1)   } held fixed
                       ground-truth actions   } held fixed, echoed as "executed"
```

**Independent variable:** the appearance of one camera view, set by
(fault family × ordinal strength rung × arm ∈ {generative, classical}).

**Dependent variables:** the policy's `[32][8]` action chunk per control step —
7 joint positions and a gripper command, DROID convention.

**Controls, in order of importance:**

1. *The untouched episode*, rolled out N times. This is the zero point and it is not
   optional — see §3.
2. *The classical arm*, matched on perceptual distance (§5). Without it, H2 is
   untestable.
3. *The validity gate*, run before the policy, so a corrupted transformation is
   rejected rather than reported (§4).
4. *Teacher forcing*: the logged action, not the predicted one, is echoed back each
   step. Both conditions therefore see the identical conditioning history.

**Subject:** `reactor/cosmos-nano-policy-droid` (Cosmos3-Nano-Policy-DROID). The policy
is the *subject* of the experiment, not a contribution.

**Material:** `nvidia/Cosmos3-DROID`, episode 0, a 96-frame window at 15 fps. Task:
*"Pour the contents of the yellow cup into the bowl."* Chosen because its columns map
1:1 onto the policy's I/O, so predicted and ground-truth actions are directly
comparable in the same units, and because the task names a specific object — which is
what makes H3 askable.

---

## 3. Why the noise floor decides everything

The policy is stochastic. Rollouts on the **identical unperturbed** episode differ by a
mean pairwise joint L2 of ~0.09–0.10 rad — larger than its own mean error against the
logged ground-truth action (0.020–0.046 rad per joint). Unperturbed runs also disagree
about the gripper decision on up to 1 step in 12.

This killed the obvious design. A single baseline-vs-perturbed comparison cannot
distinguish a perturbation effect from a resample of the same distribution.

So the unit of evidence is a **group**:

```
B = {b1..bN}  rollouts on the untouched episode
P = {p1..pM}  rollouts on the perturbed episode

T = mean between-group divergence  -  mean within-group divergence
```

Under the null, group labels are exchangeable and T is centred on zero. Shuffling
labels gives the null distribution directly — no normality assumption, and no
dependence on an estimate of sigma that is itself unstable (five calibrations spanned
sigma = 0.0059 to 0.0162, a factor of 2.7).

Reported for every comparison: **p**, **Cohen's d**, and **min attainable p**. The
third matters because the floor is `2 / C(N+M, N)` for balanced groups — at N=M=3 no
result can reach alpha = 0.05 at all, however large the effect.

A sweep applies a **Bonferroni correction** across rungs, because one test per rung is
a family of tests.

---

## 4. Validity criteria

A comparison is admitted only if all of these hold. Each has a measured threshold, not
an inherited one (`tools/calibrate_gate.py`):

| Criterion | Threshold | Rationale |
|---|---|---|
| Geometry drift (median flow, normalised) | ≤ 0.0055 | ≈ a 4 px global shift; an appearance-only X2 edit measures 0.00154 |
| Localised drift (p90) | ≤ 0.012 | catches a region sliding while the median stays blind |
| Structure retained (gradient agreement) | ≥ 0.25–0.35 per fault | distinguishes a relit scene from a redrawn one |
| Temporal motion ratio | ≤ 2.5 | generative flicker the recording never had is a confound |

**A rejected perturbation is never reported as a failure.** It produces no policy
rollout, no divergence, and no verdict. This has already fired on real data: two
perturbations were rejected for temporal flicker, and one of them was the *same rung*
that had passed on an earlier run.

---

## 5. Matching the classical arm

H2 is only meaningful if both arms are the same size of change. The classical arm is
calibrated by scanning its strength axis for the setting whose RMSE distance from the
source best matches the generative perturbation's, then given **the same number of
rollouts and the same permutation test**.

Classical operators are deliberately scene-blind: brightness/contrast, gamma + radial
bloom, blur + noise, hue/saturation shift, and a lower-frame specular gradient. The
last is the strongest classical attempt at a wet floor and is the fairest opponent —
it cannot know where the table ends or that the cup should not become reflective.

Asymmetry worth stating: the classical arm is **exactly reproducible** given
`(op, strength, seed)`. The generative arm is not, because X2 has no seed.

---

## 6. What a result licenses

| Result | Licenses | Does **not** license |
|---|---|---|
| p ≤ alpha, gate VALID, d large | "this visual change shifted the policy's action distribution on this episode" | any claim about task success, closed-loop behaviour, or real hardware |
| Generative significant, classical not, at matched distance | "generative perturbation reached an effect this classical operator did not, at this distance, on this episode" | "classical augmentation cannot reach it" — we tested five operators, not the space |
| No significant effect | "no effect detectable at this group size" | "the policy is robust" — see the p-floor |
| Gate REJECTED | "the transformation was not admissible" | anything about the policy |
| H3 null (removal changes nothing) | "behaviour was not detectably conditioned on this view of the object" | "the policy ignores the object" — the wrist camera still shows it, and the gate is blind to stationary inpainting |

Explicitly **not claimed**: that generated perturbations are physically accurate; that
appearance-space margins constitute a safety case; that open-loop divergence predicts
closed-loop outcome.

---

## 7. Findings

Each entry names the run directory holding its `experiment.json` and video evidence.

### 7.1 The policy is stochastic — established, and it reshaped the design

Five independent calibrations, 4–6 unperturbed rollouts each. Mean pairwise joint L2
0.0887–0.1040 rad; sigma 0.0059–0.0162. Gripper decisions disagree on up to 1 step in
12 with no perturbation at all.

*Status:* **VERIFIED.** Evidence: `runs/_probe/noise_floor/`, and the `noise_floor`
block of every run record.

### 7.2 X2 is not reproducible, and it changes admissibility

`specular_floor` @ 1.0, identical prompt and frames, two runs:

| run | drift | structure | temporal ratio | gate |
|---|---|---|---|---|
| `perturb-specular_floor-20260821-010958` | 0.0015 | 0.889 | 1.92 | VALID |
| `perturb-specular_floor-20260821-110125` | 0.0017 | 0.827 | 4.14 | **REJECTED** |

*Status:* **VERIFIED.** A margin is the lowest break observed, not a threshold.

### 7.3 Divergence is not monotonic in ladder strength

`backlight_glare` sweep (`runs/search-backlight_glare-20260821-011627`): RMSE rose
monotonically across rungs 0.2 → 0.8 (0.064 → 0.087) while joint divergence did not
(0.109 → 0.096 → 0.100 → 0.087). Rung 1.0 was rejected by the gate.

*Status:* **VERIFIED.** Bisection over this axis is unsound without the sweep.

### 7.4 X2 does not remove objects on prompt — the intervention must be verified

The `target_removal` ladder's top rung asks for the cup to be gone. X2 instead rendered
it as **translucent glass, still held in the gripper**.

`tools/verify_removal.py` measures this rather than trusting the prompt:

| quantity | value |
|---|---|
| target-coloured pixels removed inside the pointer region | 48% |
| surviving fraction inside the pointer region | 0.516 |
| target-coloured pixels across the whole frame, perturbed ÷ source | **1.10** |

Total target-coloured pixels *increased*. This is a strong material/appearance change
of the task object, not a removal.

*Status:* **VERIFIED (negative capability result).** Evidence:
`runs/perturb-target_removal-20260821-111730/removal_check.json` and `removal_check.png`.

**Consequence for the method:** a null behavioural result from an intervention that did
not land is a statement about the prompt, not the policy. `verify_removal.py` is now a
precondition for any PROVENANCE claim, and the fault YAML records the measured outcome
next to the requested one.

### 7.5 H1 — a large, validated appearance change did not shift the action distribution

The `target_removal` @ 1.0 transformation is a legitimate H1 data point even though it
is not a removal: it is a substantial appearance change to the task-relevant object
that passed every validity criterion.

| quantity | value |
|---|---|
| geometry drift | 0.00141 (**1.04 px** equivalent) — VALID |
| structure retained | 0.858 |
| temporal ratio | 0.781 |
| alignment score / offset | 0.915 / −3 frames |
| perceptual distance | RMSE 0.1115, SSIM 0.829 |
| within-group divergence | 0.1165 rad |
| between-group divergence | 0.1179 rad |
| **p** | **0.2575** (5000 permutations) |
| min attainable p | 0.00216 |
| Cohen's d | 0.368 |
| gripper flip rate | 0.072 within baseline → **0.056** between groups |

The design had ample power — the floor was 0.00216 and p came out at 0.26. The gripper
flip rate did not rise. This is a **well-powered null**, not an absence of evidence.

Earlier single-pair comparisons pointed the same way (`specular_floor` @ 1.0 at
+0.1 sigma; `backlight_glare` rungs 0.2–0.8 all inside the noise floor) but were
underpowered by construction and are superseded by this.

*Status:* **H1 NOT SUPPORTED on this episode, for this fault, at this group size.*
Evidence: `runs/perturb-target_removal-20260821-111730`.

*What this does not say:* that the policy is robust in general. One episode, one view,
one fault family, N = M = 6.

### 7.6 Cost per evaluation — the commercial claim, measured

One complete experiment (6 baseline rollouts + 1 X2 perturbation + 6 perturbed
rollouts, with the validity gate and the permutation test):

| quantity | value |
|---|---|
| policy session time | 266.6 s |
| perturbation session time | 24.4 s |
| total session time | 291.0 s |
| **cost at Reactor's $6/hr** | **$0.485** |
| rollout retries | 0 |

*Status:* **VERIFIED.** Searching appearance space is genuinely cheap; the binding
constraint is wall-clock (~5 min per point), not money.

### 7.7 H2 — generative vs classical at matched distance

`runs/compare-target_removal-20260821-112831`. All three arms matched to within 2% on
perceptual distance, each given 6 rollouts against a shared 6-rollout baseline.

| arm | RMSE | p (joint) | Cohen's d | gate | temporal ratio |
|---|---|---|---|---|---|
| **generative** `target_removal` @ 1.0 | 0.0604 | **0.7167** | +0.09 | VALID | 0.69 |
| classical `color_shift` @ 0.250 | 0.0618 | **0.9196** | −0.41 | VALID | 1.20 |
| classical `blur_noise` @ 0.846 | 0.0606 | **0.1102** | +0.45 | **REJECTED** | 9.28 |

Cost for the whole three-arm comparison: **$0.9256** (24 rollouts, 555 s).

**No arm reached significance.** At this perceptual distance, on this episode, neither
generative nor classical perturbation shifted the policy's action distribution.

*Status:* **H2 NOT SUPPORTED, and not yet testable in the positive direction.** H2 asks
whether generative perturbation finds what classical misses; with H1 unsupported
(§7.5) there is nothing for it to find. This is a null on both arms, not a win for
either.

**One asymmetry found and fixed.** `blur_noise` produced the largest divergence of the
three (d = +0.45, p = 0.11) but did so with **9.28× the source's frame-to-frame
motion** — per-frame independent noise is temporally incoherent in a way the recording
never was, and the gate rejected it. The classical arm had been exempt from the gate
the generative arm faced. That exemption would have let an inadmissible perturbation
count as a classical "win", so it is gone: both arms now face the same gate, with
positive controls explicitly exempted. Pinned by
`test_classical_and_generative_face_the_same_gate`.

*Observation, not yet a claim:* the one classical operator that came closest to an
effect reached it through temporal incoherence, while the generative perturbation was
temporally *more* stable than the source (0.69). With neither arm significant, this is
a hypothesis about where the two families differ, not evidence for it.

### 7.9 POSITIVE CONTROL — the apparatus has sensitivity, but only to enormous changes

Every null above shares one alternative explanation: the policy reads three camera
views and PENUMBRA perturbs one. If it barely uses `exterior_image_1_left`, no
perturbation of it could move behaviour and "robust" would be measuring the wiring.

So the view was destroyed outright — faded to black — and put through the identical
group test. `runs/_probe/positive_control.json`.

| quantity | value |
|---|---|
| perceptual distance | RMSE **0.4521**, SSIM 0.035 |
| joint p | **0.0162** (floor 0.0022, alpha/test 0.025) |
| Cohen's d | **+1.32** |
| within / between group divergence | 0.1067 / 0.1119 rad |
| gripper p | 1.0000 — flip rate 0.028 → 0.014, no decision change |
| gate | REJECTED, as expected — a black frame has no structure (0.000) |
| cost | $0.4411 |

*Status:* **VERIFIED — the channel carries signal the policy uses.** The nulls in
§7.5 and §7.7 are informative rather than vacuous.

**But read the magnitude.** It took RMSE 0.4521 to reach p = 0.0162. Every fault tested
so far sits at RMSE 0.060–0.112 — between **4× and 7.5× closer** to the source. So the
honest bound is:

> The measurement can detect a perturbation of this camera view, but only a very large
> one. A null at RMSE 0.06 says "no effect detectable at one seventh of the distance
> that does produce a detectable effect." It does not say the policy is robust, and it
> does not say the apparatus is blind.

Two further observations, both bounding rather than encouraging:

- Even total destruction of the view produced **no gripper decision change**
  (p = 1.0000). The discrete decision channel appears insensitive to this view.
- Even total destruction only reached p = 0.0162, about 7× the design's floor. The
  effect on joint space is real (d = 1.32) but not overwhelming at N = M = 6.

**The open question this creates**, and the one worth answering next: where between
RMSE 0.06 and 0.45 does detectability begin, and can X2 reach that distance while
staying inside the validity gate? If it cannot, the thesis fails on this policy for a
concrete, measurable reason rather than an unexplained absence of findings.

### 7.10 REACH vs THRESHOLD — the nulls have a mechanical explanation

The question the project turns on reduces to two numbers measured independently.
`tools/reach_analysis.py` computes both from every run on disk.

**Admissible reach** — how far X2 can get from the source while still passing the gate:

| fault | strength | RMSE | gate | temporal ratio |
|---|---|---|---|---|
| `target_removal` | 1.0 | 0.0604 | VALID | 0.69 |
| `backlight_glare` | 0.2 | 0.0640 | VALID | 0.73 |
| `backlight_glare` | 0.4 | 0.0641 | VALID | 0.72 |
| `backlight_glare` | 0.6 | 0.0757 | VALID | 0.91 |
| `backlight_glare` | 0.8 | 0.0869 | VALID | 1.20 |
| `specular_floor` | 1.0 | 0.0873 | VALID | 1.92 |
| `target_removal` | 1.0 | **0.1115** | **VALID** | 0.78 |
| `specular_floor` | 1.0 | 0.1256 | **REJECTED** | 4.14 |
| `backlight_glare` | 1.0 | 0.2102 | **REJECTED** | 4.27 |

**The mechanism is visible in the last column.** Temporal ratio climbs with perceptual
distance — 0.69, 0.73, 0.91, 1.20, 1.92 — and then breaks past the 2.5 gate at 4.14 and
4.27. X2's admissible reach is not capped by our thresholds being harsh; it is capped
because **the model loses temporal coherence as the edit gets stronger**. Push it far
enough from the source to matter and it starts producing motion the recording never
had, which is a confound, not a fault.

**Detection threshold** — the blackout ruler. Blackout is not a fault; it is a
measuring stick, a pure global luminance change that is gate-clean at every rung
(drift 0.0002–0.0008) so nothing but distance varies.

| blackout | RMSE | SSIM | joint p | Cohen's d | detected |
|---|---|---|---|---|---|
| 0.15 | 0.0692 | 0.975 | 0.2158 | +0.46 | no |
| 0.30 | 0.1369 | 0.902 | 0.1158 | −0.24 | no |
| 0.50 | 0.2266 | 0.715 | 0.5845 | −0.15 | no |
| 1.00 | **0.4521** | 0.035 | **0.0162** | **+1.32** | **yes** |

Detectability begins somewhere in **(0.2266, 0.4521]**. Note there is no gradual onset:
p wanders (0.22, 0.12, 0.58) with no trend until the view is destroyed outright. The
policy is not *degraded* by darkening this view — it appears to disregard it until it
carries almost nothing.

**The gap:**

```
admissible reach    0.1115  (max RMSE, gate VALID)
gate-rejected up to 0.2102
largest not detected 0.2266
smallest detected   0.4521
                    ------
                    at least 2.0x short, and 4.1x on the measured points
```

Even the *largest undetected* distance (0.2266) is double X2's admissible reach. The
shortfall does not depend on where in the bracket the true threshold sits.

*Status:* **VERIFIED on the data so far.** On this policy and this camera view, no
admissible generative perturbation yet tested gets far enough from the source to
register. The nulls in §7.5 and §7.7 are not mysterious — they are what this gap
predicts.

This is the honest form of the project's central negative result. It is falsifiable in
two directions: a finer ruler could put the detection threshold below 0.1115, or a
different fault family could reach further while staying temporally coherent. Both are
cheap to test, and both are the obvious next experiments.

### 7.11 CHANNEL ABLATION — the single-view intervention was aimed at a redundant channel

Every experiment up to this point perturbed `exterior_image_1_left`, because that is
the view a human would call "the scene camera". That was an assumption, and it was the
wrong one.

The policy reads **three** cameras plus proprioception. Destroying one and destroying
all three are not the same intervention at different strengths — they are different
interventions, because two clean views and the arm state remain available to
compensate.

`runs/_probe/channel_ablation.json`, same cached baseline group for both conditions:

| blackout | joint p | Cohen's d | gripper p | gripper flip rate | within → between |
|---|---|---|---|---|---|
| `exterior_image_1_left` only | 0.0162 | +1.32 | **1.0000** | 0.028 → 0.014 | 0.107 → 0.112 |
| **all three views** | **0.0030** | **+4.97** | **0.0046** | 0.028 → **0.167** | 0.111 → **0.191** |

Two things change, and the second matters more than the first:

1. **Effect size is ~4× larger** (d 1.32 → 4.97), and between-group divergence rises
   72% above within-group rather than 5%.
2. **The gripper decision channel wakes up.** Single-view destruction left it perfectly
   flat (p = 1.0000, flip rate *falling* 0.028 → 0.014). All-view destruction flips
   decisions on 17% of step-pairs at p = 0.0046. A changed decision is a qualitatively
   different finding from a shifted trajectory, and it is only reachable here.

*Status:* **VERIFIED.** This reframes §7.5, §7.7 and §7.10: those nulls were measured
on a channel the policy can route around. They remain true as stated — no admissible
single-view generative perturbation moved the policy — but "the policy is insensitive
to appearance" was never the right reading, and the reach-vs-threshold gap in §7.10 is
a fact about *single-view* interventions specifically.

**Consequence for the method:** the intervention must degrade every channel the policy
can substitute between. That is now the design constraint, and it is why the
infrastructure grew multi-view perturbation, per-view gating, and per-view spatial
anchors.

### 7.12 THE RESULT — an admissible generative intervention that moves the policy, which matched classical augmentation does not

This is the finding the project was built to produce. Episode 0, 96 frames, all three
camera views, `reactor/cosmos-nano-policy-droid`.

**The intervention.** X2 applied to all three views (one session per view), under the
`target_occlusion` prompt. What X2 actually produces is **target amplification**, not
occlusion — the yellow cup the task names grows to roughly five times its
target-coloured pixel count. See §7.13 for why the fault's name is not its behaviour.

**Three independent X2 samples.** X2 has no seed, so these are genuinely independent
draws from the intervention's distribution, not repeats of one video.

| run | gate | drift | temporal | RMSE | joint p | Cohen's d | amplification |
|---|---|---|---|---|---|---|---|
| `perturb-…-125347` | VALID | 0.0015 | 1.03 | 0.0694 | **0.0056** ✱ | +0.70 | ×4.92 |
| `perturb-…-125903` | VALID | 0.0016 | 1.38 | 0.0653 | **0.0184** ✱ | +0.43 | ×5.07 |
| `compare-…-131105` | VALID | 0.0015 | 1.30 | 0.0645 | **0.0132** ✱ | +1.84 | ×4.10 |

**Three for three significant** at alpha/test = 0.025, every one passing the validity
gate at ~1.1 px equivalent geometry drift, and the physical intervention itself
reproducing tightly (×4.1–5.1).

**The two controls that make it mean something.**

| arm | views | RMSE | joint p | Cohen's d | gate |
|---|---|---|---|---|---|
| generative, all views | 3 | 0.0645 | **0.0132** ✱ | +1.84 | VALID |
| generative, **same fault, one view** | 1 | 0.0736 | 0.8538 | −0.40 | VALID |
| classical `color_shift`, **matched distance, same three views** | 3 | **0.0726** | 0.5001 | −0.45 | VALID |

The single-view arm is the same fault, same prompt, same strength, same cached
baseline — it simply reaches one of three channels, and the policy routes around it.
The classical arm perturbs **the same three views at a slightly larger perceptual
distance** and does not move the policy at all.

So the effect is not "a bigger change". At matched — indeed slightly greater —
perceptual distance, a scene-blind pixel transform across the same channels produces
nothing (p = 0.50), while the generative one produces a reproducible shift.

*Status:* **H1 SUPPORTED and H2 SUPPORTED**, on this episode, this policy, this fault.

**The honest caveats, none of which are small:**

1. **The three runs share one cached baseline group.** The perturbed groups are
   independent X2 samples, but the control group is common to all three, so the tests
   are correlated and their p-values must not be combined by any method assuming
   independence. A fresh baseline per run would fix this and has not been done.
2. **Effect sizes are inconsistent** — 0.70, 0.43, 1.84. Two of three were flagged by
   the comparator as a weak signal. The direction replicates; the magnitude does not.
3. **The gripper decision channel never moved** (p = 1.0000 in all three). This is a
   trajectory shift, not a changed decision — weaker than the all-view blackout
   result, which did flip decisions.
4. **One classical operator is not "classical augmentation".** `color_shift` at matched
   distance failed; four other operators were not run at all in this configuration.
5. **One episode.** Generalisation is §7.14.

### 7.13 X2 does not insert objects — the third verified capability limit

`target_occlusion` sets `set_reference_image` (a drawn cardboard carton) and a measured
per-view `set_pointer`, both accepted without error. **No box appeared in any view, in
any of three runs.** `tools/verify_occlusion.py`:

| check | result |
|---|---|
| occluder appears | **NO** — median change area 0.004, bar 0.012 |
| change on target | **NO** — centroid 0.47 from the anchor |
| overlaps the target | **NO** — 0.21, bar 0.25 |
| persists across frames | **NO** — 0.19 of frames, bar 0.80 |
| leaves the scene intact | yes |
| passes SEAM | yes |

What X2 did instead was amplify the object already there. So the verified X2 limits now
number three: **no seed · no numeric strength · no object insertion on this scene.**

The fault keeps its name only because three recorded runs reference it; the YAML
carries a MEASURED OUTCOME block stating the name is not the behaviour, and every
result from it is reported as amplification.

**This is also the strongest argument for the verification discipline.** Without
`verify_occlusion.py`, three significant runs would have been written up as "occluding
the target object changes the policy's behaviour" — a claim the pixels do not support.

### 7.14 GENERALISATION — the effect reproduces on a second episode, independently

Episode 1, a different recording with a different task string ("Pour the contents in
the cup into the bowl"), same all-view intervention.
`runs/perturb-target_occlusion-20260821-131847`.

| quantity | episode 0 (best of 3) | **episode 1** |
|---|---|---|
| baseline cache key | `10da2a2327f1` | **`6e5c29dc5a3e`** |
| gate | VALID | **VALID** |
| geometry drift | 0.0015 | 0.0015 (1.1 px) |
| temporal ratio | 1.30 | 0.72 |
| perceptual distance | 0.0645 | 0.0920 |
| target amplification | ×4.10 | **×4.04** |
| within → between divergence | 0.1016 → 0.1225 | 0.0985 → **0.1289** |
| **joint p** | 0.0132 | **0.0030** ✱ |
| **Cohen's d** | +1.84 | **+4.63** |
| gripper p | 1.0000 | 1.0000 |

**This is a genuinely independent test.** The baseline cache key differs, so episode 1
drew its own control group — it does not inherit the shared-baseline dependency that
qualifies the three episode-0 runs (§7.12 caveat 1). The effect is not only present but
**substantially larger** (d = 4.63 vs 1.84), and this run carried no "weak signal"
flag.

The physical intervention reproduces across episodes too: ×4.04 here against ×4.10–5.07
on episode 0, from four independent seedless X2 samples.

*Status:* **The effect generalises beyond the episode it was found on.** Two episodes
is two, not a benchmark — but it is no longer a single-episode result, and the second
episode was the stronger of the two.

Still constant across every run: **the gripper decision channel does not move**
(p = 1.0000, four for four). Whatever this intervention does to the policy, it shifts
the trajectory without changing the grasp decision.

### 7.15 A false positive in our own decision rule

The first version of the break criterion had two routes: a significance-tested joint
divergence, and a bare ratio on gripper flip rate. The ratio route declared a **BREAK
at p = 0.7167**, on a flip rate moving from 0.000 to 0.028 — roughly one flip across
432 step-comparisons.

A ratio between two noisy small-sample means is not evidence. The gripper flip rate now
gets **its own permutation test**, both statistics are held to alpha/2, and a
regression test feeds the comparator null data and asserts it finds nothing.

*Status:* **FIXED.** Both `BREAK` labels in
`runs/compare-target_removal-20260821-112831` were produced by the old rule and are
superseded by the p-values in the table above.

### 7.8 H3 — PROVENANCE

*Status:* **NOT TESTABLE BY THIS ROUTE.** §7.4 shows the removal intervention does not
land. No claim about whether the policy conditions on the target object is supported by
anything run so far.

---

## 8. Threats to validity

| Threat | Handling |
|---|---|
| Perturbation corrupted geometry | SEAM gate before the policy; measured thresholds |
| Perturbation effect is really frame misalignment | Explicit alignment with a recorded NCC score; a 3-frame desync measures 0.00011 drift, far below threshold |
| Stochastic policy read as an effect | Permutation test over groups; p-floor reported |
| Multiple rungs inflate false positives | Bonferroni across rungs |
| Ladder assumed monotone | Sweep by default; monotonicity checked and reported (already observed false) |
| X2 variance read as policy variance | `tools/x2_stability.py` measures the generative spread separately |
| Classical arm strawmanned | Matched on perceptual distance, same rollout count, same test |
| Single-model artefact | Cross-model corroboration via SANA-Streaming — **not yet run** |

---

## 9. Open questions this cannot answer

1. **Transfer.** Do open-loop divergences predict closed-loop or real-hardware failure?
   Needs a robot.
2. **Reachability, properly.** We compare against five classical operators, not the
   space of classical augmentation.
3. **Corroboration as validity.** Does agreement between architecturally independent
   world models function as a reliable validity criterion? Untested here.
4. **Generality.** One episode, one policy, one view.
