# PENUMBRA — HONEST ASSESSMENT

**What this is worth, and to whom.** Written to be read by someone deciding whether to
fund, adopt, or kill this. Every claim here is either backed by a run in `runs/` or
explicitly marked as not established.

**Updated:** 2026-08-23 · session 4

> **Read §4.0 first.** A cross-camera audit run on 2026-08-23 invalidated the
> headline count of every multi-camera suite this project has produced. The
> methodology below survives; the finding counts do not.

---

## 1. What actually exists and runs

A closed loop, end to end, on real infrastructure:

```
real DROID episode  →  LLM director  →  X2 render  →  validity gate
                                                          ↓
   verdict  ←  suite-wide correction  ←  permutation test  ←  real VLA rollouts
```

Nothing in that chain is mocked. The episode is a real robot recording; the director is
`gemini-robotics-er-2-preview` looking at the actual workspace frames; the renderer is
`xmax/x2` over WebRTC; the policy is `reactor/cosmos-nano-policy-droid`, a real
video-plus-proprioception VLA; the statistics are permutation tests with multiplicity
correction. A suite takes about an hour and costs a few dollars.

That is the thing that has been built. Whether it is *useful* is a separate question,
and the rest of this document is about that question.

---

## 2. The methodological result, which is the most transferable thing here

**A generative perturbation must be measured against a no-op re-render, not against the
raw recording.**

X2 regenerates every pixel whatever the prompt asks for. A prompt saying *"the scene is
exactly as it is, no changes"* still shifts this policy's action distribution
(`p = 0.0030`, `d = +2.32`) and moves its gripper decisions — a larger effect than most
of the named faults produced. Anyone running a video model over robot footage and
comparing against the original is, by default, measuring the video model.

The control is itself validated: two independent no-op renders are behaviourally
indistinguishable (`p = 0.32`, `d = 0.05`), so a difference from the control can be
attributed to the situation rather than to generative variance.

This is a small idea and it is load-bearing. **Every earlier result in this project that
used the raw episode as the control has to be read with that confound in mind.** It is
also the part of this work most likely to be true regardless of what happens to the rest
of it.

---

## 3. What has been demonstrated

| Claim | Status | Evidence |
|---|---|---|
| The loop runs end to end and produces reproducible artefacts | **VERIFIED** | any `runs/garage-*` |
| Situations separate a real VLA's actions from a matched control, surviving Bonferroni | **VERIFIED** | `runs/garage-ep0-20260821-142420`: 5 of 7 tested, α_bonf = 0.003571 |
| ...but almost none of those separations can be *attributed* to the situation they are named after | **VERIFIED** | §4.0: 19 of 21 renders incoherent across cameras |
| ...but only **3 of those 5** show the situation they are named after | **VERIFIED** | independent VLM adjudication, §4.3 |
| X2 follows a content prompt about a third of the time | **VERIFIED** | 7 of 21 renders judged applied or partial |
| The finding replicates across an independent suite | **VERIFIED** | `-143928` independently re-found `greasy_wrist_lens` |
| Gate thresholds are measured against known transforms, not inherited | **VERIFIED** | `runs/_probe/gate_calibration*.json` |
| A single-camera perturbation mostly measures channel redundancy | **VERIFIED** | one view `d = 1.32`; all views `d = 4.97`, gripper flips 0.028 → 0.167 |
| X2 corrupts the wrist camera on content prompts | **VERIFIED** | `runs/gatesweep-ep0-20260821-161434`; visible duplicated cups and a hallucinated human hand |
| Restricting to the exterior cameras raises usable renders from 7/21 to 16/21 | **VERIFIED** | `runs/gatesweep-ep0-20260821-162003` |

---

## 4. What has *not* been demonstrated — read this before quoting anything above

### 4.0 The cameras were rendering different scenes, and nothing checked

X2 has exactly one input track. Perturbing two cameras means **two separate sessions**,
and X2 takes no seed, so nothing couples the draws. That limitation was documented from
the start as "a real property of the intervention". Its consequence was never measured.

`tools/coherence_audit.py` re-judged all 21 renders of `garage-ep0-20260821-234747`
camera by camera, then asked an independent model whether the two descriptions describe
one change or two:

```
21 renders:  2 coherent   19 showing DIFFERENT scenes on the two cameras
             14 of the 19 had been reported as vulnerabilities
```

| render | camera 1 | camera 2 |
|---|---|---|
| `mirror_chrome_cup` | the **table** became chrome; cup unchanged | the **robot arm** became chrome; cup replaced |
| `matching_pink_cup_and_bowl` | the cup became pink — correct | a pink **bucket** was added; cup still yellow |
| `color_confused_cup` | bowl filled with reflective liquid | **no visible change at all** |

The policy reads both cameras in one observation. When they disagree it is not shown a
weaker version of the situation — it is shown two different worlds, and **no situation
name describes its input**. A behaviour change measured against that is real and
unattributable.

**Consequence: the reported count of every multi-camera suite in this project is wrong.**
Re-scored under the current taxonomy, `garage-ep0-20260821-234747` goes from **15
vulnerabilities to 1 confirmed and 14 unattributed**. The statistics were never the
problem. The names were.

The fix is structural, not cosmetic. `finding_class` now separates *the policy moved*
from *this situation moved it*, and only a render that passed the gate, that both
cameras agree on, and that an independent judge says depicts its own name can carry the
second claim. Incoherence is a gate, and a retryable one.

**What this does not excuse.** Every earlier headline in this repository, including ones
already shown to people, was produced without this check. Read them as unattributed.


### 4.1 No robot has failed. Not once.

The measurement is **open-loop action divergence**. The policy is replayed over
perturbed pixels while teacher-forced on the logged trajectory, and what is measured is
whether its predicted action distribution moves. It is never allowed to act, so there is
no task success, no dropped cup, no collision, no closed loop.

"The policy's actions changed" is a necessary condition for a real failure. It is
nowhere near sufficient. A policy whose actions shift by 0.1 rad and still completes the
task has not failed at anything.

**This is the single largest gap between what PENUMBRA measures and what a robotics team
cares about.**

### 4.2 One episode, one window

Every headline number comes from episode 0, frames 120–216, one task ("put the cup in
the bowl"), one policy. There is a second-episode replication for one earlier fault, and
nothing at suite scale. Any claim about *this policy* in general is unsupported.

### 4.3 The situation names are often labels, not causes — now measured

X2 does not reliably do what the prompt says, and as of this session that is quantified
rather than asserted. An independent vision model (`gemini-robotics-er-2-preview`),
shown only the before and after frames and never the statistics, adjudicated all 21
renders of `runs/garage-ep0-20260821-142420`:

```
 6  applied              the render shows what was asked for
 1  partially_applied
 7  not_applied          "no visible differences between the original and edited frames"
 7  something_else       the editor did a different thing entirely
```

**X2 produced the requested situation in 7 of 21 attempts.** The "something else" cases
are not subtle: asked for a greasy lens smudge it added a glass bowl of red liquid;
asked for yellow gripper fingers it replaced the gripper with a yellow-and-black robot
toy; asked for dust it added two extra yellow cups.

For the five confirmed vulnerabilities, **3 of 5 have names the render supports**:

| situation | judged | what the render actually shows |
|---|---|---|
| `amber_safety_lighting` | applied | amber-orange lighting across the scene (plus a spurious glass) |
| `pink_juice_spill_near_bowl` | applied | a large glossy pink puddle around the bowl |
| `sunlight_glare_shaft` | applied | a bright diagonal streak across table and bowl |
| `swapped_cup_bowl_colors` | **not applied** | the cup is still yellow and the bowl still pink |
| `greasy_wrist_lens` | **something else** | a glass bowl of red liquid added on top of the pink bowl |

`greasy_wrist_lens` is the one that replicated across both independent suites and the
only one that moved the gripper decision. **The behaviour change is real. The name is
not.** What that run establishes is that *this particular render* moves the policy — not
that a greasy lens does.

The judge is adjudication, not measurement: a vision model's opinion about a render, and
it can be wrong in both directions. It gates nothing, and no scenario is promoted
because the judge approved of it. It exists so a reader can discount a finding whose
render never happened. Runs made from this session onward adjudicate every perturbed
camera on the frames the policy actually received (`scenarios/judge.py`); earlier runs
can be retro-adjudicated from saved media with `tools/judge_run.py`, on one camera and
through mp4 compression.

### 4.3b Where this policy is sensitive — measured, and it reframes everything above

Across 46 tested-and-judged situations, classified by what the perturbation targets:

| perturbation targets | n | median Cohen's d | moved the policy |
|---|---|---|---|
| the manipulated objects (cup, bowl, contents, gripper) | 10 | **+0.53** | 4 / 10 |
| objects **and** scene together | 25 | **+0.60** | 9 / 25 |
| **the scene alone** (tabletop, room lighting, walls, floor) | 9 | **−0.01** | **0 / 9** |

**Changing only the background has never moved this policy.** Nine attempts, median
effect indistinguishable from zero, no discoveries. Every one of the 14 vulnerabilities
found across all suites involved the objects being manipulated.

This was found the hard way. A suite whose director had been told to prefer whole-scene
appearance prompts came back with **zero** vulnerabilities and a maximum effect of
`d = +0.40` — the guidance had steered the search into exactly the region where the
policy is robust. The instruction was mine and the null was its consequence, not a
property of the intervention.

Two consequences, and the second is the one to hold onto:

1. Later suites deliberately aim most situations at object appearance, keeping a couple
   of scene-only ones as a live control so the null keeps being tested. **Any hit rate
   from those suites is conditional on that targeting** and is not a sample of an
   unbiased space. Quoting an improved hit rate without saying so would be
   straightforwardly dishonest.
2. As a characterisation of the policy this is worth more than the individual findings:
   `cosmos-nano-policy-droid` attends to what it manipulates and disregards the backdrop.
   That is the kind of statement a robotics team can act on.

A rival explanation was tested and rejected. Effect size does **not** track hallucinated
objects: median |d| is 0.40 for renders that invented something and 0.51 for those that
did not, and only 4 of 14 vulnerabilities involved an invented object. The earlier
findings were not simply "the policy noticed a toy robot".

### 4.3c X2's usable band is narrow, and it is bounded from both sides

| prompt style | outcome |
|---|---|
| bare scene descriptions, no instruction verbs | **7 of 18 renders showed no visible change at all** — the editor declined |
| vivid, extreme appearance language | **12 of 21 renders rejected** by the validity gate — temporal ratios up to 8.0 against a 2.5 bar, localised drift up to 3.6× its bar |

Both extremes yield about eight testable renders out of twenty-one. Weak prompts fail by
producing nothing; strong ones fail by disrupting the scene faster than they change its
appearance. The rejections at the strong end are not marginal and survive redrawing —
three scenarios were redrawn up to three times and still failed, so this is the model's
behaviour rather than an unlucky sample.

Streaming settings are **not** the lever. A controlled sweep (`tools/tune_x2.py`, same
prompt and window) found every configuration applied the edit at RMSE 0.22–0.26; only
the prompt separated declining from applying. One draw per configuration, and X2 has no
seed, so the differences between configurations sit inside its own draw-to-draw spread.

### 4.4 The central hypothesis is still open

> Can generative appearance perturbations discover physical-AI policy failures that
> classical image augmentation cannot?

The best evidence so far is one comparison: at matched perceptual distance across the
same three cameras, `color_shift` produced nothing (`p = 0.50`) where the generative arm
produced a reproducible shift. **One classical operator is not "classical
augmentation"** — four others were never run in that configuration.
`tools/head_to_head.py` runs all five at matched distance against the same gate and the
same test. Until that has been run at suite scale, **the thesis is supported by a single
data point and should be described that way.**

This matters commercially more than anything else in this document. If a brightness
curve and a blur kernel find the same failures, the generative pipeline is an expensive
way to do something free.

### 4.5 The statistics are at their resolution floor

With N = M = 6 the smallest expressible p-value is 2/C(12,6) = 0.00216. Five confirmed
vulnerabilities all reported `p = 0.0030` — that is the floor, reproduced identically
because the permutation draw is seeded. It means *"no relabelling was more extreme"*,
i.e. maximal separation at this sample size. **It does not mean five equal effects**; the
effect sizes ranged from `+0.95` to `+4.10`. The tooling now flags this explicitly
(`at_resolution_floor`), and the fix is more rollouts, which costs money.

---

## 5. Commercial usefulness, honestly

### Where the value would be

Robotics teams have no equivalent of fuzzing or chaos engineering for perception.
Validation is "run the held-out eval set", which by construction contains the conditions
the policy was trained for. Nobody systematically asks *which visual conditions outside
the training distribution change this policy's behaviour, and by how much*. The shape of
that gap is real.

The parts of this work that would survive contact with a real customer:

1. **The no-op control discipline** (§2). Anyone building generative perturbation
   testing without it will publish the video model's behaviour as the policy's.
2. **The validity gate with measured thresholds.** A generative edit that corrupts the
   scene produces a behaviour change that says nothing about the policy. The gate is
   cheap, it fires on real data, and its thresholds come from measurement. The research
   document's proposed threshold of 0.05 would only have rejected shifts larger than
   about 36 px — it would have passed essentially any corruption.
3. **The camera-specific corruption finding.** Anyone pointing a streaming video editor
   at multi-camera robot data will hit this. It cost two full suites to find because
   only one camera's media was being saved.
4. **The reporting discipline** — resolution floors, suite-wide correction, per-camera
   verdicts, and a rejected render never counted as a finding.

### Where it falls short of a product

| Gap | Why it blocks adoption |
|---|---|
| No closed-loop execution | A customer wants "the robot drops the cup under glare", not "the action distribution moved". Needs a simulator or hardware. |
| No task-success metric | Without one there is no severity, so no way to prioritise which finding to fix. |
| Single-episode coverage | A finding on one 4-second window is an anecdote. Coverage means hundreds of episodes. |
| Generative advantage unproven | §4.4. If classical augmentation matches it, the cost is unjustified. |
| Wrist camera unusable | The camera closest to the manipulation is the one X2 corrupts. That is where a grasp-relevant perturbation would matter most. |
| Cost and latency | ~1 hour and a few dollars per 21-situation suite on one episode. Fleet-scale regression testing needs orders of magnitude better. |

### The honest one-paragraph verdict

**PENUMBRA is a working research instrument and not yet a product.** It demonstrates
that an LLM-directed, generatively-perturbed, statistically-gated robustness search can
be built and run against a real VLA for a few dollars an episode, and it has produced
named situations that measurably move a real policy under a defensible methodology. What
it has not produced is evidence that any of those situations would make a real robot
fail, that they generalise beyond one episode, or that a free classical augmentation
would not have found them. The first two are engineering problems with known solutions
and real cost. The third is the scientific question the project exists to answer, and it
is still open — one data point in favour, four operators untested. Anyone evaluating this
should treat the methodology as the deliverable and the findings as preliminary.

---

## 6. What to do next, in priority order

1. **Run `tools/head_to_head.py` at suite scale.** It settles §4.4, costs a few dollars,
   and determines whether the rest of the project is worth continuing.
2. **Close the loop.** Even a crude simulator that executes the predicted actions and
   reports task success converts every finding from "the distribution moved" to "the
   task failed", which is the difference between a metric and a bug report.
3. **Widen to many episodes.** One suite per episode across ten episodes turns anecdotes
   into a coverage claim.
4. **Verify the intervention per scenario.** Measure that the render did what the prompt
   asked before attributing a behaviour change to the named situation.
5. **Raise the group sizes.** Escape the p-value floor so effect ranking is meaningful.

Do not build fleet infrastructure, dashboards beyond the current one, or a fault
catalogue until (1) has an answer.
