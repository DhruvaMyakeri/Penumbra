# PENUMBRA

**A garage for robot policies.** Put a trained policy on the lift, drive it through
situations it was never trained for, and get back a documented list of the ones that
changed its behaviour — with the video, the statistics, and the reason each verdict is
trustworthy attached.

```
real robot episode  (DROID, real physics / geometry / camera trajectory)
   ↓
LLM director        proposes situations grounded in this task and this workspace
   ↓
X2 video editor     renders each situation onto the camera feeds
   ↓
validity gate       rejects renders that corrupted the scene, before the policy sees them
   ↓
intent judge        an independent vision model says what the render ACTUALLY shows
   ↓
real VLA policy     rolled out on perturbed pixels, teacher-forced on the real trajectory
   ↓
permutation test    against a no-op re-render, corrected across the whole suite
   ↓
verdict             "this situation moved the policy" — or "it did not"
```

Everything runs on a laptop plus Reactor APIs. **No physical robot is involved**, and no
result in this repository comes from one.

---

## Run it

```bash
pip install -e .

# a full suite: 21 situations across 7 mechanism categories, live dashboard
.venv/Scripts/python tools/garage.py --episode 0 --scenarios 21

# reopen any finished run (this is REPLAY mode - no network, no spend)
.venv/Scripts/python tools/view.py runs/garage-ep0-...
```

The dashboard opens at `http://127.0.0.1:8765` and streams every generation, gate
verdict and statistical result as it happens.

Credentials come from `.env` and never leave the process — the config layer logs only a
length and a hash prefix.

### Useful flags

| flag | what it does |
|---|---|
| `--views` | which cameras to perturb. X2 corrupts the wrist view on most content prompts, so `--views exterior_image_1_left,exterior_image_2_left` trades intervention strength for a much higher share of usable renders |
| `--scenarios` | how many situations the director proposes (spread over 7 categories) |
| `--repeats` / `--screen` | rollouts for confirmation / for cheap screening |
| `--port` | dashboard port. The server refuses a port already in use rather than silently serving an older run |

---

## The one design decision that matters

**Every situation is measured against a no-op re-render, not against the raw
recording.**

X2 regenerates every pixel whatever the prompt says. A prompt asking for *no change at
all* still shifts this policy's action distribution (`p = 0.0030`, `d = +2.32`) and
moves its gripper decisions. Compare a situation to the untouched episode and you are
measuring the situation **plus** the re-render — and the re-render alone is enough to
reach significance.

So the control is the same episode through the same model with a null prompt. That
control is itself validated: two independent no-op renders are behaviourally
indistinguishable (`p = 0.32`, `d = 0.05`), so the measurement is stable enough to
attribute a difference to the situation.

---

## The second design decision that matters

**The situation's name is a hypothesis about the render, not a fact about it.**

X2 carries out a content prompt roughly a third of the time. Measured on one suite by an
independent vision model shown only the before and after frames: 7 of 21 renders applied
(or partly applied) what was asked, 7 showed no visible change at all, and 7 did
something else entirely — asked for a greasy lens smudge it added a glass bowl of red
liquid; asked for yellow gripper fingers it replaced the gripper with a robot toy.

So every render is adjudicated before its result is reported. A finding whose render
does not show the situation is still a real behaviour change — the statistics are
untouched — but the card says plainly that its *stated cause* is not established. The
judge gates nothing and never promotes a finding; it only lets you discount one.

---

## What is verified, and what is not

Read `docs/ASSESSMENT.md` for the honest evaluation of what this is worth, and
`docs/LIMITATIONS.md` before quoting any number from this repository. In short:

**VERIFIED** — X2 drives programmatically and edits real DROID footage with geometry
intact on the exterior cameras; a real VLA (`reactor/cosmos-nano-policy-droid`) runs on
Reactor and tracks the logged trajectory; the validity gate's thresholds are measured
against known transforms rather than inherited from a design document; suites have
produced situations that separate from the control and survive Bonferroni correction.

**VERIFIED, AND LIMITING** — X2 corrupts the **wrist camera** on content prompts
(duplicated objects, a hallucinated human hand, ripple artefacts). It was the sole
failing camera in every rejection observed, and it alone was rejecting two thirds of
every all-camera suite.

**NOT CLAIMED** — that a real robot would fail in any of these situations; that the
cause of a behaviour change is the object named in the prompt; that these findings
generalise past the episodes they were measured on. Each of those needs work this
repository has not done.

---

## Layout

```
src/penumbra/
  reactor/       session + transport for Reactor models
  episodes/      DROID loading, the Episode type, multi-view replacement
  perturbation/  X2 adapter, classical control arm, fault specs
  policy/        the VLA under test, teacher-forced rollout, baseline cache
  validation/    SEAM validity gate
  evaluation/    permutation tests, effect sizes, multiplicity correction
  scenarios/     the LLM director and the garage itself
  ui/            live dashboard (stdlib HTTP + SSE)
tools/           every experiment and probe, each runnable on its own
docs/            capabilities, findings, limitations
```

### Tools worth knowing about

| tool | what it answers |
|---|---|
| `tools/garage.py` | run a suite, live dashboard |
| `tools/view.py` | REPLAY a finished run — no network, no spend |
| `tools/head_to_head.py` | does a matched classical augmentation find the same thing? **the central hypothesis** |
| `tools/gate_sweep.py` | render + gate only, per camera — no policy spend |
| `tools/judge_run.py` | retro-adjudicate what a finished run's renders actually show |
| `tools/calibrate_gate.py`, `tools/calibrate_localised.py` | where the gate's thresholds come from |
| `tools/diagnose_rejection.py` | was a rejection real corruption, or a recolour artefact? |
| `tools/check_dashboard.js` | render the UI headlessly, so a dead page cannot hide |

`pytest tests/` — 71 tests, no network required.
