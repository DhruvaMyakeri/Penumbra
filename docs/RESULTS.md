# Results

All figures come from logged runs. The showcase in [`demo/`](../demo/index.html) presents the same data with every render playable, and `tools/build_demo.py` rebuilds it from the run records.

## Setup

| | |
|---|---|
| Episode | `nvidia/Cosmos3-DROID`, episode 0, frames 120 to 216 (about 4 seconds) |
| Task | "Pour the contents of the yellow cup into the bowl" |
| Policy | `reactor/cosmos-nano-policy-droid` (three cameras and joint state in, 32 x 8 action chunk out) |
| Renderer | `xmax/x2` on Reactor |
| Director and vision check | `gemini-robotics-er-2-preview` |
| Suites | 9 completed, 3 perturbing all three cameras, 6 perturbing the two exterior cameras |
| Rollouts per comparison | 6 control against 3 (screening) or 6 (confirmation) |

## The policy responds to appearance

| | |
|---|---|
| Situations proposed | 170 |
| Rendered, gated and tested | 92 |
| **Shifted the policy** | **51**, all surviving Benjamini-Hochberg correction within their suite |
| Median effect size | d = 1.57 (range 0.26 to 4.10) |
| Median arm-plan shift beyond natural variation | 1.38° (largest 3.23°) |
| Changed the grasp decision | 4 situations |
| Compute | $19.28 in total |

Across 92 tested situations, 51 moved the policy's intended actions further from the control than the policy moves on its own. 36 of the 51 reached the smallest p-value the 6-against-6 design can express, so effect size is the better guide to how strong each one is.

The first suite predates the pipeline's suite-wide correction step. The same correction function was applied to its two results afterwards; both pass Benjamini-Hochberg and Bonferroni.

### The strongest shifts

| Situation | Suite | Arm plan vs control | Natural variation | Grasp flips | d | p | What the policy saw (vision check) |
|---|---|---|---|---|---|---|---|
| `amber_safety_lighting` | S02 | 10.7° | 7.8° | 7% | +4.10 | 0.0030 | intense amber-orange lighting across the scene, plus a glass goblet on a saucer |
| `greasy_wrist_lens` | S03 | 9.2° | 6.6° | 4% | +3.91 | 0.0030 | not adjudicated (suite predates the vision check) |
| `neon_green_slime_coating` | S07 | 9.1° | 6.2° | 1% | +3.86 | 0.0030 | cup and bowl filled and surrounded by neon green slime |
| `liquid_soaked_bowl_and_cup` | S08 | 8.5° | 5.5° | 5% | +3.85 | 0.0030 | a glossy brown syrup puddle on the table, with two orange glass cups |
| `cup_interior_shadow` | S03 | 10.2° | 7.3° | 0% | +3.60 | 0.0030 | not adjudicated |
| `highly_reflective_bowl_interior` | S08 | 7.7° | 5.9° | 2% | +3.23 | 0.0030 | the pink bowl rendered as a larger glossy black bowl |
| `mud_splattered_vessels` | S07 | 8.8° | 6.3° | 1% | +3.04 | 0.0030 | thick brown liquid filling the vessels and running across the table |
| `greasy_wrist_lens` | S02 | 10.5° | 7.3° | 8% | +2.87 | 0.0030 | a glass bowl of red liquid over the pink bowl |
| `tarnished_copper_vessels` | S07 | 7.7° | 6.5° | 4% | +2.81 | 0.0030 | green and brown chalices on the table |
| `rust_corroded_gripper_and_cup` | S08 | 7.7° | 6.1° | 6% | +2.61 | 0.0030 | a dark-brown textured form running from the bowl to the table |

"Arm plan vs control" is the mean RMS disagreement across the seven joints between control and situation rollouts; "natural variation" is the same quantity within the groups. The situation name is the edit that was requested; the last column is what an independent vision model observed in the frames the policy received.

### Grasp decisions

A shifted arm trajectory is a different path. A flipped gripper command is a different decision. In four situations the gripper test was significant on its own:

| Situation | Suite | Gripper flips vs control | Between control runs | p |
|---|---|---|---|---|
| `color_swap_distractor` | S01 | 8.3% of steps | 0% | 0.0030 |
| `high_gloss_glare` | S01 | 8.3% | 0% | 0.0030 |
| `greasy_wrist_lens` | S02 | 8.3% | 0% | 0.0030 |
| `foam_filled_bowl` | S09 | 5.6% | 0% | 0.0204 |

### By category

| Category | Shifts | Median d |
|---|---|---|
| contents and state | 10 | 1.98 |
| surface and spill | 9 | 1.48 |
| object appearance | 9 | 1.30 |
| lighting | 8 | 1.64 |
| sensor-style degradation | 6 | 2.20 |
| confusable objects | 4 | 1.54 |
| environment | 3 | 2.61 |

Across the suites, the renders that moved the policy most were those that changed the objects on the table: their contents, surface, colour, or the appearance of new objects near them. That is consistent with a policy that attends to what it manipulates.

## The control matters

With the wrist camera among the perturbed views, re-rendering the scene with no requested change moved the policy in all 3 suites (d = 1.50 to 2.08, p ≤ 0.006). With only the two exterior cameras perturbed, the re-render alone had no significant effect in 6 suites (d = -0.18 to 0.67).

Every result above is measured against a no-op re-render of the same episode, so that artefact is on both sides of each comparison and cancels. A comparison against the raw recording would have attributed the re-render's effect to every situation in the three-camera suites.

## Verification

Every tested render carries a vision check, camera by camera, and from the latest suites a cross-camera agreement check. One shift is verified end to end: `table_surface_mismatch` (S08), where both cameras show a blue plastic sheet over the workbench, the vision model judged it applied, and the policy separated from the control (d = +0.56, p = 0.0080).

For most shifts the render took a different form from the one requested, as the table above shows: X2 frequently repaints an object differently from the request, or adds a new object near it. Those results establish that the appearance change the policy actually received moved it; the vision check records which change that was.

Measuring the renderer this way also produced a set of rules for prompting it (see [METHODOLOGY.md](METHODOLOGY.md#prompt-structure)). Prompts that name a single handled object and target it whole match the request about twice as often as prompts that do not, and the pipeline now enforces them.

## Generative against classical perturbation

Four classical operators were strength-matched to the median perceptual distance of one suite's generative renders (RMSE 0.111) and run through the same gate and the same test:

| Operator | Matched distance | Gate | d | p (arm) | Shifted the policy |
|---|---|---|---|---|---|
| gamma glare | 0.112 | valid | +0.28 | 0.011 | yes |
| brightness and contrast | 0.112 | valid | +0.15 | 0.030 | no |
| colour shift | 0.115 | valid | -0.56 | 0.283 | no |
| blur and noise | 0.070 (could not reach the target) | rejected, frame-to-frame flicker | +0.89 | 0.003 | not admissible |

At the same perceptual distance, the generative renders in that suite produced a median effect of d = 1.61, against d = 0.28 for the strongest admissible classical operator. Classical augmentation can move this policy, and gamma glare did, but generative appearance changes moved it several times further at the same amount of pixel change. A fifth operator, specular overlay, was interrupted by a network failure and has not been re-run.

## Scope

- The measurement is the policy's **intended actions** in open-loop replay. It was never allowed to act, so these results describe changes in its plan rather than task outcomes.
- All results come from **one episode and one policy**.
- Situation names describe the requested edit. The vision check on each result records what the policy saw.
