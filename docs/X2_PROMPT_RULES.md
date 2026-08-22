# What X2 actually does with a prompt — measured

Source: 92 judged renders across all `runs/garage-*` suites, adjudicated by an
independent VLM that saw only before/after frames and never the statistics.

Headline: **the render shows the situation it is named after in about 27% of attempts.**
Everything below is an attempt to move that number, and each rule is a measurement
rather than an intuition. Two intuitions I held were tested and rejected; they are
recorded at the bottom so nobody re-derives them.

## Rule 1 — name at most ONE manipulable object (strongest effect)

X2's dominant failure is **additive**: it draws a *new* instance of a noun rather than
restyling the existing one. The odds of that scale with how many manipulable objects
the prompt names.

| distinct object nouns named | n | render usable | invented an object |
|---|---|---|---|
| 0 | 15 | 53.3% | 40.0% |
| 1 | 28 | 28.6% | **35.7%** |
| 2 | 44 | 18.2% | **72.7%** |
| 3+ | 5 | 20.0% | **100%** |

Naming a second object roughly doubles the invention rate. Every prompt that named
three invented something.

## Rule 2 — never target a narrow geometric feature

Rims, edges, gripper fingers, "the chips inside the cup", "the inner wall".

| target | n | render usable |
|---|---|---|
| mentions a narrow feature | 10 | **0.0%** |
| whole object or whole surface | 82 | 30.5% |

Zero for ten. X2 works at the scale of a whole object or a whole surface; below that it
substitutes something instead.

## Rule 3 — one object, not zero, because zero cannot find anything

Object count trades render fidelity against whether the finding is worth having:

| object nouns | n | median \|d\| | moved the policy | render usable | **both** |
|---|---|---|---|---|---|
| 0 | 13 | 0.22 | 3/13 | 5/13 | **1** |
| 1 | 25 | **1.07** | 14/25 | 5/25 | **4** |
| 2 | 37 | 1.25 | 23/37 | 5/37 | **4** |
| 3+ | 3 | 1.03 | 2/3 | 1/3 | 1 |

Naming zero objects renders best and discovers nothing — the policy ignores the
backdrop. Naming two moves the policy just as hard as naming one but lies about what it
shows twice as often. **Exactly one is the operating point.**

## Rejected — tested, not supported

- **"Physical substances render better than optical effects."** Substance prompts were
  34.2% usable (n=38), optics-only 33.3% (n=12). No difference. Do not filter on this.
- **"Indefinite articles cause insertion."** Fewer than 2 of 92 prompts used the
  construction at all, so it explains nothing about the observed 58% invention rate.

## Not established

- Camera-space prompts (lens smudge, frame blur) went 0/3 usable. Suggestive of X2
  editing the scene rather than the lens, but n=3 proves nothing.

---

# The cross-camera coherence problem (measured 2026-08-23)

X2 has exactly one input track. Perturbing two cameras therefore means **two separate
sessions**, and X2 takes no seed, so nothing couples the two draws. This was documented
as "a real property of the intervention". Its consequences were never measured.

`tools/coherence_audit.py` re-judged all 21 renders of
`runs/garage-ep0-20260821-234747` camera by camera, then asked an independent model
whether the two descriptions describe one change or two.

```
audited 21 renders:  2 coherent   19 showing DIFFERENT situations   0 undetermined
14 of the 19 incoherent renders had been reported as vulnerabilities
```

Examples of what "incoherent" means here:

| render | camera 1 | camera 2 |
|---|---|---|
| `color_confused_cup` | bowl filled with reflective liquid | **no visible change at all** |
| `greasy_gripper_and_cup_glare` | a black nozzle emitting a yellow beam was added | cup and bowl replaced by glass vessels |
| `oil_slick_rainbow_sheen` | a green bowl replaced the yellow cup | a pouring stream of liquid |

The policy reads both cameras in the same observation. When they disagree it is not
being shown a weaker version of the situation — it is being shown two different worlds,
and **no situation name describes its input**. A behaviour change measured against that
stimulus is real but unattributable.

## Does prompt discipline fix it?

Partly, and not enough to rely on.

| prompt | coherent |
|---|---|
| 0 rule violations | 2 / 7 |
| 1 violation | 0 / 13 |
| 2+ violations | 0 / 1 |

Both coherent renders obeyed every rule; every rule-breaking render was incoherent. But
29% coherence on clean prompts is not a working yield on its own.

## What follows

Coherence has to be **gated, not just reported**: an incoherent render is rejected and
re-drawn like any other invalid one, and can never become a named vulnerability. Prompt
discipline raises the odds each draw succeeds; the gate is what keeps the failures out
of the headline.

**Every previously reported multi-camera vulnerability count in this project predates
this check** and should be read as unattributed until re-run.
