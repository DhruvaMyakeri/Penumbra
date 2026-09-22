# Methodology

The question PENUMBRA answers for each situation:

> With physics, geometry and motion held fixed, did changing only the scene's appearance move the policy's intended actions further than the policy moves on its own?

Four pieces make that question answerable.

## 1. The noise floor

The policy samples. Run it twice on the identical recording and it gives two different answers. On the episode used here, six unperturbed runs (15 pairs) disagree with each other by:

| | |
|---|---|
| Arm plan, RMS across 7 joints | **0.106 rad (6.1°) ± 0.012 rad (0.7°)** |
| Gripper decision | differs on **2.8%** of steps on average |

Any measured change has to be read against that spread. A single before-and-after comparison cannot separate a real effect from the policy disagreeing with itself.

## 2. The no-op control

The renderer regenerates every pixel whatever it is asked, so comparing a render against the raw recording measures the situation and the re-render together. PENUMBRA instead renders the same episode with a prompt asking for no change, and compares every situation against that.

This matters in practice. Measured across the suites here:

| Cameras perturbed | Suites | No-op re-render vs raw recording |
|---|---|---|
| 3 (including the wrist camera) | 3 | moved the policy every time: d = 1.50 to 2.08, p ≤ 0.006 |
| 2 (exterior only) | 6 | no significant effect: d = -0.18 to 0.67 |

With the wrist camera included, re-rendering alone is enough to shift the policy. Because both sides of every comparison carry the re-render, it cancels out of the result. Two independent no-op renders of the same window are behaviourally indistinguishable from each other (p = 0.32, d = 0.05), which is what makes the control stable enough to measure against.

## 3. The validity gate

A render that shifts, warps or boils the scene would move the policy for reasons that have nothing to do with appearance. The gate rejects those renders before the policy sees them, per camera:

- **Global drift:** median dense optical-flow displacement between original and render.
- **Localised drift:** the 90th percentile of that displacement, which catches one object moving while the rest holds still.
- **Structure:** agreement of gradient orientation, so edges stay where they were.
- **Temporal ratio:** frame-to-frame motion in the render against the source, measured outside the edited region when the edit is local, which catches flicker.

Thresholds were calibrated on the real footage by applying known transforms:

```
brightness change only          0.0005
a real X2 edit                  0.0015
shifted by 2 px                 0.0027
threshold                       0.0055
shifted by 4 px                 0.0055
shifted by 32 px                0.0436
```

A camera that misses a bar by a small margin is redrawn (the renderer has no seed, so a redraw is a fresh sample); a render far past a bar is rejected.

## 4. The group test

For each situation, the policy is rolled out on the render and on the control. Every pair of rollouts is compared with the divergence metrics above, and the test asks whether the between-group disagreement exceeds the within-group disagreement:

```
T = mean divergence between control and situation rollouts
  - mean divergence within the groups
```

If the situation has no effect the group labels are exchangeable, so shuffling them 5,000 times gives the null distribution of T directly, with no assumptions about its shape:

```
p = (count of shuffles with T >= observed + 1) / (5000 + 1)
```

Two statistics are tested, the arm trajectory and the gripper decision, each at α/2. Cohen's d is reported beside every p-value as the measure of size.

**Screening then confirmation.** Every valid render is screened with 3 situation rollouts against 6 control rollouts; anything showing a signal is confirmed at 6 against 6.

**Resolution.** With 6 against 6 there are 924 ways to assign the labels, and because swapping sides gives the same statistic, only 462 are distinct. The smallest p-value the design can express is therefore about 0.0022, and the seeded permutation draw reports it as 0.0030. Many strong results sit at that floor; the effect size is what ranks them.

**Across the suite.** A suite runs about twenty tests against one control, so Benjamini-Hochberg decides which results stand, with Bonferroni reported alongside as the stricter bar.

## Verification of what was rendered

The renderer does not always render what it is asked. So every tested render carries a vision check: an independent model, shown only before and after frames, reports what changed on each camera and whether that matches the request. With two perturbed cameras, it also checks whether both show the same change, since each camera is rendered in a separate pass.

This produces two independent labels on every result:

| Label | Meaning |
|---|---|
| **Policy shift** | the policy's actions separated from the control after suite-wide correction |
| **Verified** | additionally, the gate passed, the vision check says the render matches the request, and the cameras agree |

A policy shift says that the appearance change the policy actually saw moved it. The vision check says what that change was. Only a verified result lets the situation's name stand for its cause.

## Prompt structure

Across 92 renders checked by the vision model, one property of a prompt predicted whether the renderer did what was asked more than any other: how many objects the robot handles it names.

| Objects named | Render matched the request | Renderer added an unrequested object |
|---|---|---|
| 0 | 53% | 40% |
| 1 | 29% | 36% |
| 2 | 18% | 73% |
| 3 or more | 20% | 100% |

Prompts that target part of an object (a rim, an edge, a gripper finger) produced no usable renders in ten attempts. Prompts obeying both rules matched the request 39% of the time against 18% for the rest.

These rules are enforced mechanically. The director's proposals and the render agent's rewrites are checked against them and sent back when they break them. Scene-only categories carry a separate rule: they must name no handled object at all, so they stay a genuine test of whether background changes move the policy.

## Reproducibility

Every suite records the episode and window, the policy and its settings, each prompt exactly as sent, every render attempt, gate metrics per camera, the vision check per camera, every statistic with its test settings, and the cost. The unperturbed baseline is cached and keyed by everything that could change it. Seeds for the permutation test are fixed.

The renderer takes no seed, so a render cannot be reproduced bit for bit. Its run-to-run variation on a no-op prompt is about twenty times smaller than the drift it introduces against the source, and two no-op renders are behaviourally indistinguishable, so results are reproducible at the level of the measurement even though individual pixels are not.
