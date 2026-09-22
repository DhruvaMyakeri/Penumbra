# Architecture

PENUMBRA is a pipeline with one job: take a real robot recording, change only how it looks, and measure whether a policy's intended actions change. Every stage is behind an interface, so the renderer, the policy and the episode source can each be swapped without touching the rest.

```
episode ──► director ──► renderer ──► validity gate ──► vision check
                                                            │
             report ◄── suite correction ◄── group test ◄── policy rollouts
```

## Components

### Episodes (`src/penumbra/episodes/`)

Loads real teleoperated episodes from the `nvidia/Cosmos3-DROID` dataset: three camera streams (two exterior views and a wrist view), the arm's joint state, and the logged human actions. A window of the episode (96 frames by default) is the unit every experiment runs on.

Because the physics, geometry, camera motion and robot trajectory come from a real recording, nothing about the scene has to be simulated. A live camera feed can replace the recording behind the same `Episode` type.

### Director (`src/penumbra/scenarios/director.py`)

A vision-language model (`gemini-robotics-er-2-preview`, with fallbacks) looks at the actual first frame of each camera and proposes situations that could plausibly occur in that workspace and might mislead the policy. Each proposal carries a situation, a reason it might matter, and a prompt for the renderer.

Proposals are spread across weighted categories (object appearance, contents and state, confusable objects, surfaces and spills, lighting, environment, sensor-style degradation) so a suite covers different mechanisms rather than twenty variations of one. Every prompt is checked against measured prompt-structure rules before it is used (see [METHODOLOGY.md](METHODOLOGY.md#prompt-structure)), and a prompt that breaks them is sent back for a rewrite.

### Renderer (`src/penumbra/perturbation/x2.py`)

`xmax/x2` on Reactor, a streaming video-to-video model. Frames are pushed over WebRTC and the repainted frames are aligned back to the source by normalised cross-correlation. Each perturbed camera is its own session. Transport failures (dropped sessions, connect timeouts) are retried on a fresh session; genuine request errors are not.

A classical augmentation arm (`perturbation/classical.py`: brightness and contrast, gamma glare, blur and noise, colour shift, specular overlay) runs through the same gate and the same statistics, for comparison.

### Render agent (`src/penumbra/scenarios/render_agent.py`)

When a render fails the gate, comes back unchanged, or shows something other than what was asked, a language model rewrites the prompt and the render is attempted again, up to three times. It changes the wording, never the situation being tested. Every attempt, its outcome and the agent's reasoning are kept in the run record.

### Validity gate (`src/penumbra/validation/seam.py`)

Rejects any render that moved the scene rather than repainting it, before the policy ever sees it. Checks run per camera: global and localised optical-flow drift, structure retained, and a temporal motion ratio that catches flicker. Thresholds are calibrated on the real footage (see [METHODOLOGY.md](METHODOLOGY.md#the-validity-gate)).

### Vision check (`src/penumbra/scenarios/judge.py`)

An independent vision model is shown the original and repainted frames for each camera and reports what changed, whether it matches the request, and any objects that appeared. With more than one perturbed camera it also checks whether the cameras show the same change. It records; it never decides whether a render is tested.

### Policy (`src/penumbra/policy/cosmos_droid.py`)

`reactor/cosmos-nano-policy-droid`, a vision-language-action model trained on DROID. Inputs: three camera streams, joint state and the task instruction. Output: an action chunk of 32 future steps by 8 channels (7 joint positions and a gripper command).

The policy is run in **teacher-forced open-loop replay**. At each step it sees the frames and predicts an action chunk; the logged human action is then reported back as what was executed. That keeps the policy on the recording's trajectory, so a baseline run and a perturbed run differ only in pixels. Unperturbed baseline rollouts are cached on disk keyed by everything that could change them.

### Statistics (`src/penumbra/evaluation/`)

`divergence.py` measures how far two rollouts disagree: RMS difference across the seven joints in radians, and the rate at which the gripper decision differs. `grouptest.py` runs the permutation test between a group of control rollouts and a group of situation rollouts. `multiple.py` applies Benjamini-Hochberg and Bonferroni across a suite.

### Suite runner (`src/penumbra/scenarios/garage.py`)

Orchestrates a suite: calibrate the noise floor, propose situations, render the no-op control, render and gate each situation, screen every valid render with 3 rollouts, confirm promising ones at 6, correct across the suite, and write the reports. State is written to `state.json` after every step, rendered frames are saved so an interrupted suite can resume, and an insight ledger records what the renderer did with each prompt so later situations in the suite are briefed with it.

### Reports and UI

- `report/run_report.py` writes a report per situation and a run summary (`REPORT.md`).
- `ui/dashboard.html` is the live console. `tools/garage.py` serves it during a run; `tools/view.py` reopens any finished run from disk with no network or spend.
- `demo/` is the static showcase, built from finished runs by `tools/build_demo.py`.

## Run artefacts

Each suite writes to `runs/garage-ep<N>-<timestamp>/`:

```
state.json       every situation: prompt, render attempts, gate, vision check, statistics
insights.json    what the renderer did with each prompt, across the suite
REPORT.md        run summary
reports/         one markdown report per situation
media/           H.264 clips (original views over repainted views) and contact sheets
frames/          rendered frames, for resume and re-analysis
```

`runs/` and `data/` are not versioned.

## Live and replay

Everything that touches Reactor or Gemini is live. Everything downstream of a finished run (the dashboard, the reports, the showcase) replays from disk, so a demonstration does not depend on the network.
