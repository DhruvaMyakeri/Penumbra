# PENUMBRA — IMPLEMENTATION STATUS

**Updated:** 2026-08-21 · session 3

Companions: `README.md` (how to run it), `docs/ASSESSMENT.md` (what it is worth and to
whom), `docs/REACTOR_CAPABILITIES.md` (what Reactor actually does),
`docs/RESEARCH.md` (the experiments and their findings), `docs/LIMITATIONS.md` (what
none of it licenses), `docs/PENUMBRA_RESEARCH.md` (original research context).

---

## THE HEADLINE

The product is **the garage**: a policy is driven through LLM-directed situations and
comes back with a documented list of the ones that measurably changed its behaviour.

```
real DROID episode  →  Gemini director  →  X2 render onto every camera  →  validity gate
                                                                              ↓
      verdict  ←  suite-wide correction  ←  permutation test vs no-op  ←  VLA rollouts
```

Two 21-situation suites have run end to end. The first found **5 vulnerabilities**
surviving Bonferroni; the second independently re-found one of them. A suite costs about
$1.50 and an hour.

The load-bearing design decision, and the most transferable result in the project:

> **Every situation is measured against a no-op re-render, not against the raw
> recording.** X2 redraws every pixel whatever the prompt says, and that redraw *alone*
> shifts this policy (`p = 0.0030`, `d = +2.32`) — more than most named faults. Compare
> against the untouched episode and you are reporting the video model.

---

## WHAT WORKS

| # | Capability | Evidence |
|---|---|---|
| 1 | X2 (`xmax/x2`) drives programmatically; spatial controls are real SDK surface | `runs/_probe/xmax_x2.json` |
| 2 | X2 edits real DROID footage with geometry intact **on the exterior cameras** | median p90 drift 0.0023 vs a 0.012 bar |
| 3 | A real VLA runs on Reactor and tracks the logged trajectory | 0.020–0.046 rad/joint vs ground truth |
| 4 | Real DROID episodes load locally (27 available, 3 views + proprio + GT actions) | `nvidia/Cosmos3-DROID` |
| 5 | Gemini director proposes situations grounded in the task **and the workspace image** | `scenarios/director.py`, 7 mechanism categories |
| 6 | The no-op control, and its validation | null vs null: `p = 0.32`, `d = 0.05` |
| 7 | SEAM gate with **measured** thresholds, firing on real data | `runs/_probe/gate_calibration*.json` |
| 8 | Per-camera gate verdicts, recorded and surfaced | `per_view_gate` in every scenario |
| 9 | Permutation test on **two** statistics, with a resolution floor that is now flagged | `evaluation/grouptest.py` |
| 10 | Suite-wide FDR + Bonferroni before anything is called a discovery | `evaluation/multiple.py` |
| 11 | Two-stage rollout budget: cheap screen, then confirm | 3× the coverage for the same spend |
| 12 | Live dashboard over SSE; every generation, gate verdict and statistic as it happens | `ui/server.py`, `ui/dashboard.html` |
| 13 | REPLAY mode — reopen any finished run with no network and no spend | `tools/view.py` |
| 14 | Classical control arm, matched distance, **same gate, same test** | `perturbation/classical.py`, `tools/head_to_head.py` |
| 15 | 69 unit tests, no network required | `pytest tests/` |

### The numbers, measured

```
GATE (global)     appearance-only X2 edit 0.00154 | 2 px shift 0.00272
                  threshold 0.00550 (= 4 px)      | 32 px shift 0.04356
                  research doc proposed 0.05 = only rejects shifts > ~36 px

GATE (localised)  NEW. The original calibration used only GLOBAL shifts, whose p90
                  equals their median - so the p90 bar had never been measured against
                  the localised displacement it exists to catch. Now it has:
                    recolour 70% of the frame        p90 0.00143   (nothing moved)
                    displace 30% of frame by  8 px   p90 0.01089
                    displace 30% of frame by 16 px   p90 0.02179
                    displace  5% of frame by 16 px   p90 0.00039   <- BLIND SPOT
                  A small region can move a long way and the gate will not see it.

X2 SELF-NOISE     two independent no-op renders of the same window:
                  drift 0.00013 | p90 0.00055 | structure 0.983 | temporal 1.017
                  ~20x below either render's drift against the source. Seedless does
                  not mean unusable as an instrument.

PER-CAMERA        median localised drift, same prompts on all three cameras:
                    exterior_image_1_left  0.00235
                    exterior_image_2_left  0.00224
                    wrist_image_left       0.01416   <- above the 0.012 bar
                  wrist was the SOLE failing camera in every rejection observed.

YIELD             all three cameras   7/21 renders valid
                  exteriors only     16/21 renders valid
                  and the exterior rejections are a different mode: 4 of 5 are
                  marginal temporal flicker (2.57-2.80 vs a 2.50 bar), not geometry

STATISTICS        N = M = 6 -> smallest expressible p = 2/C(12,6) = 0.00216
                  five confirmed vulnerabilities all reported p = 0.0030 = the floor,
                  identical because the permutation draw is seeded. Effect sizes
                  ranged +0.95 to +4.10. Now flagged as `at_resolution_floor`.

COST              $1.18-2.17 per 21-situation suite; 1-2 hours
                  (the slower render settings roughly double the rendering half)

SENSITIVITY MAP   46 tested situations, grouped by what the perturbation targets:
                    manipulated objects (cup/bowl/contents)  median d +0.53   4/10 moved
                    objects and scene together               median d +0.60   9/25 moved
                    scene alone (table, lighting, walls)     median d -0.01   0/9  moved
                  Background appearance has NEVER moved this policy. All 14
                  vulnerabilities found to date involved the manipulated objects.

X2 USABLE BAND    bare scene-description prompts   -> 7/18 renders showed NO change
                  vivid extreme appearance prompts -> 12/21 renders REJECTED by the gate
                  Both ends leave about 8 testable renders in 21. Streaming settings do
                  not move this: a controlled sweep found every configuration applied
                  the edit at rmse 0.22-0.26; only the prompt separated the two failures.

HALLUCINATION     19 of 46 judged renders invented an object nobody asked for.
                  It does NOT predict effect size - median |d| 0.40 with invented
                  objects, 0.51 without. Hypothesis tested and rejected.
```

---

## WHAT FAILED

| Failure | Detail | Resolution |
|---|---|---|
| `reactor/x2` does not exist | 403 | Correct name is **`xmax/x2`** |
| X2 has no seed | Absent from schema | Quantified instead of assumed: run-to-run p90 0.00055 |
| X2 has no strength parameter | Absent from schema | Ordinal prompt ladder, labelled as such |
| **X2 does not remove objects on prompt** | Target-coloured pixels *rose* to 1.10× | H3 not testable by this route |
| **X2 does not insert objects on prompt** | Box never appeared; the cup amplified ×4.04–5.07 | Verified across 3 runs, every camera |
| **X2 corrupts the wrist camera** | Duplicated cups, a hallucinated human hand, ripple artefacts | `--views` selects cameras; the limit is documented, not hidden |
| Gate threshold from the research doc was 30× too lenient | 0.05 only rejects >36 px shifts | Calibrated on real data: 0.0055 |
| **p90 bar was never calibrated for localised corruption** | The old ladder used only global shifts | `tools/calibrate_localised.py`; a blind spot is now documented |
| `min_achievable_p` optimistic by 2× | For N=M the statistic is invariant under label swap | Floor is `2/C(N+M,N)` |
| **Five vulnerabilities all reporting p = 0.0030 looked like a coincidence** | It is the seeded permutation floor | `at_resolution_floor` flagged in the result, the notes, and the UI |
| **False positive: BREAK declared at p = 0.72** | Gripper route was a bare 1.5× ratio, no significance test | Gripper gets its own permutation test; both held to α/2 |
| **Classical arm exempt from the gate it competes against** | `blur_noise` "won" with 9.28× the source's frame-to-frame motion | Both arms face the same gate; positive controls exempt |
| **A Reactor session wedged and hung an experiment** | Unbounded `send_command`/`publish_track` | Every op bounded; rollouts retry; retries counted into cost |
| **A stale dashboard server served a previous run** | `pkill` does not work on Windows; `allow_reuse_address` let a second server bind the same port | `allow_reuse_address = False` and an error naming the cause |
| **The dashboard script was a hard SyntaxError for two sessions** | A raw newline inside a JS string literal. The server answered 200, the markup arrived, and the page rendered nothing | Rebuilt; `node --check` and a headless render check now run in `pytest` |
| Media saved only one camera | Made the wrist corruption invisible for two suites | Contact sheet is one row per camera, source above perturbed |
| **Video was written in a codec no browser decodes** | OpenCV `mp4v` is MPEG-4 Part 2. The server returned 200 and the `<video>` element stayed black | H.264 via PyAV; every old run transcoded; regression test |
| **The judge was fed a contact sheet as if it were one frame** | `judge_run.py` ignored the media grid layout, so verdicts described "a 2x2 grid showing multiple views" - confident nonsense | Splits the grid per camera; regression test |
| **Director guidance steered the search away from the signal** | Told to prefer whole-scene prompts, the director produced a suite that was 5 of 8 scene-only and found nothing | Guidance now carries the measured sensitivity map, keeping scene-only situations as a live control |
| **Renders the editor declined were still tested** | 7 of 18 showed no visible change, consuming rollouts and inflating the multiplicity correction | Anything within 2x X2's own no-op variation is marked `no_change` and skipped |
| **A marginal one-camera gate failure discarded the whole situation** | `granite_benchtop_texture` rejected at p90 0.01203 against a 0.01200 bar while the other camera passed comfortably | The camera is redrawn (cap 2, every attempt recorded); the threshold is untouched |

---

## WHAT WAS VERIFIED

- X2's `set_prompt`, `set_reference_image`, `set_pointer{,_x,_y,_active}`,
  `set_keep_backlog`, `reset` are real SDK commands. **Spatial placement is not
  playground-only.**
- `reactor/cosmos-nano-policy-droid` is a genuine video+proprio→action VLA.
- The policy is **stochastic**; its variability exceeds its own tracking error.
- A **single-camera** perturbation mostly measures channel redundancy: one view
  `d = 1.32`, all views `d = 4.97` with gripper flips 0.028 → 0.167.
- **The no-op re-render alone moves the policy**, which is why it is the control.
- **X2 corrupts the wrist camera on content prompts**, and it alone rejected two thirds
  of every all-camera suite.
- Named situations separate from that control and survive Bonferroni.

---

## WHAT REMAINS UNVERIFIED

| Question | Why it matters | Status |
|---|---|---|
| Does a **matched classical augmentation** find the same situations? | If yes, the generative pipeline is an expensive way to do something free. **This is the project's central hypothesis.** | One operator tested (`color_shift`, p = 0.50). `tools/head_to_head.py` runs all five. **Pending.** |
| Do the findings survive with **only the exterior cameras** perturbed? | The high-yield configuration leaves the policy one clean channel | **Running** |
| Would a real robot fail? | The measurement is open-loop action divergence, never task success | Out of reach without a simulator or hardware |
| Do findings generalise past episode 0, frames 120–216? | One window is an anecdote | One earlier fault replicated on episode 1; nothing at suite scale |
| Does the render do what the prompt says? | `amber_safety_lighting` is a label, not a cause | Verified for a few faults, not for suite scenarios |
| Why does X2 fail on the wrist specifically? | Would say whether any video editor can perturb close-up views | Effect verified; mechanism inferred only |
| SANA-Streaming as an independent corroborator | Cross-model agreement strengthens a finding | Not probed |

---

## WHAT I AM DOING NEXT

1. **`tools/head_to_head.py` at suite scale.** It settles the central hypothesis and
   costs a few dollars. Nothing else should be built before it has an answer.
2. Suite on the exteriors-only configuration, to see whether a 16/21 yield still
   produces findings when the policy keeps a clean wrist channel.
3. Per-scenario intervention verification, so a situation's *name* can be attributed.
4. More episodes, then larger groups to escape the p-value floor.

Not being built: fleet infrastructure, a fault catalogue, automated fixing.

---

## ARCHITECTURE AS BUILT

```
src/penumbra/
  config.py             credentials (fingerprint-only), paths
  episodes/             Episode + ObservationSource (live-camera seam); DROID loader
  reactor/session.py    ReactorSession - connect/ready/events, bounded operations
  policy/               Policy protocol, PolicyTrace; cosmos_droid teacher-forced replay
                        cache.py - baseline traces reused across runs
  perturbation/         FaultSpec ladders; X2Perturbation (one session per view);
                        classical arm
  validation/seam.py    SEAM gate, measured thresholds, perceptual distance
  evaluation/           divergence + noise floor; grouptest (two permutation tests,
                        resolution floor); multiple.py (BH + Bonferroni)
  search/bisect.py      ladder sweep / bisection, Bonferroni across rungs
  scenarios/            director.py (Gemini, 7 categories); garage.py (THE PRODUCT)
  experiments/          runner (gate before policy, retry), record + artifacts
  ui/                   server.py (SSE, refuses a taken port); dashboard.html
tools/
  probe_reactor.py  list_models.py  fetch_droid.py  smoke_x2.py  smoke_policy.py
  calibrate_gate.py         global-shift gate calibration
  calibrate_localised.py    localised-corruption calibration + the blind spot
  gate_sweep.py             render + gate only, per camera, no policy spend
  diagnose_rejection.py     was a rejection real, or a recolour artefact?
  head_to_head.py           generative vs matched classical, same gate, same test
  garage.py                 run a suite with the live dashboard
  view.py                   REPLAY: reopen any finished run
  check_dashboard.js        render the UI headlessly, so a dead page cannot hide
  positive_control.py  reach_analysis.py  channel_ablation.py  verify_removal.py
  verify_occlusion.py  locate_target.py  x2_stability.py  content_test.py
  make_occluder.py  noise_floor.py  probe_account.py
faults/                 specular_floor, backlight_glare, target_removal
tests/                  69 tests, no network
```
