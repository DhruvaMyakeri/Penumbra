# Reactor capability report

Experimentally verified by connecting to live Reactor endpoints with the account's
API key. Every line marked `VERIFIED` was observed on the wire, not read from
marketing or inferred.

- **Probed:** 2026-08-21
- **SDK:** `reactor-sdk` 1.1.0 (PyPI, `py3-none-win_amd64`, ships `reactor_ffi.dll`)
- **Coordinator:** `https://api.reactor.inc`
- **Raw evidence:** `runs/_probe/*.json` (status timelines, capabilities, full OpenAPI schemas)
- **Reproduce:** `python tools/probe_reactor.py xmax/x2 reactor/cosmos-nano-policy-droid`

---

## 1. `xmax/x2` - primary perturbation operator

| Field | Value | Status |
|---|---|---|
| MODEL | `xmax/x2` | VERIFIED |
| VERSION | `v0.0.0`, protocol `0.0.0` | VERIFIED |
| DESCRIPTION (server) | "Streaming video-to-video editing with prompt, reference image, and drag pointer." | VERIFIED |
| HARDWARE | 1x NVIDIA B200 | VERIFIED (`/models`) |
| INPUTS | track `source` - `video`, `sendonly` | VERIFIED |
| OUTPUTS | track `main_video` - `video`, `recvonly` | VERIFIED |
| SDK SUPPORT | Full. `reactor-sdk` 1.1.0 connects, publishes, commands, receives | VERIFIED |
| FRAME CALLBACK | `track.on_frame(fn)` yields `(H,W,3)` uint8 RGB NumPy array | VERIFIED (SDK source + live receipt) |
| FRAME PUSH | `track.push_frame(ndarray)` accepts arbitrary decoded frames | VERIFIED |
| CONNECT LATENCY | 8.7 s and 10.6 s over two runs (`connecting -> waiting -> ready`) | VERIFIED |
| STATEFULNESS | Stateful session; `state_update` webhook carries full state on connect and on every change | VERIFIED |
| REPRODUCIBILITY | **No seed command exists.** Not in the schema, not undocumented-but-present | **VERIFIED ABSENT** |
| SPATIAL CONTROL | **Present in the SDK**, not playground-only | **VERIFIED** |
| VIDEO INPUT | Arbitrary decoded frames via `publish_track("source")` + `push_frame`. **No v4l2 loopback needed** | VERIFIED |

### Commands (from `request_schema()`, OpenAPI paths)

| Command | Payload | Notes |
|---|---|---|
| `set_prompt` | `{prompt: str <=1000}` | Non-empty prompt **required before generation begins**. Applies from next block. |
| `set_reference_image` | `{reference_image: ReactorUploadReference}` | Object/character to insert or swap. Uploaded via `reactor.upload_file()`. **Replacing mid-run restarts the stream.** |
| `set_pointer` | `{x: 0..1, y: 0..1, active: bool}` | Normalized to output frame. **This is "spawn at pointer" plus "follow the drag".** |
| `set_pointer_x` / `set_pointer_y` / `set_pointer_active` | scalar setters | Same handles, individually. |
| `set_keep_backlog` | `{keep_backlog: bool}` | `false` (default) drops backlog for bounded latency. `true` consumes every frame in order. **`true` is what offline replay needs** - completeness over latency. |
| `reset` | `{}` | Clears prompt, reference image, pointer; stops generation. |

### Events (webhooks, arriving as `message` events)

`generation_started` (carries chosen `width`/`height`, fixed for the session),
`generation_stopped` (`reason`), `prompt_accepted`, `reference_image_accepted`,
`pointer_changed`, `state_update`, `command_error`.

### LIMITATIONS (verified)

1. **No seed.** X2 is not reproducible run-to-run. Any strength-search result must be
   reported with across-run variance, not as a deterministic number.

   **Measured, 2026-08-21.** Two independent no-op renders of the same 96-frame window
   were compared against each other: median geometry drift `0.00013`, p90 `0.00055`,
   structure retained `0.983`, temporal ratio `1.017`. That is roughly *twenty times*
   below the drift either render shows against the raw source (p90 `0.0024`), and
   behaviourally the two are indistinguishable (`p = 0.32`, `d = 0.05`). So the absence
   of a seed does **not** make X2 unusable as a measurement instrument: it is not
   bit-reproducible, but its run-to-run variation is small next to the effects being
   measured. Evidence: `runs/perturb-null_edit-20260821-134102` vs `-135946`.
2. **No numeric strength parameter.** Perturbation intensity is only reachable through
   prompt *language*. A continuous `strength` axis in [0,1] must be constructed by
   PENUMBRA and mapped onto prompt text. **Monotonicity of that mapping is an
   assumption, not a fact**, and bisection search inherits that assumption.
3. **Output resolution is chosen once at first generation and fixed for the session.**
4. The pointer is **sampled once per generated block**, not per frame.
5. `set_reference_image` mid-run restarts generation - reference swaps are not free.
6. **X2 corrupts the wrist camera on content prompts. VERIFIED, and it is the single
   biggest constraint on what PENUMBRA can currently test.**

   DROID gives the policy three cameras. Running the same prompt onto all three and
   gating each separately (`tools/gate_sweep.py`, `runs/gatesweep-ep0-20260821-161434`):

   ```
   camera                    median localised drift (p90)   bar
   exterior_image_1_left                          0.00235   0.012
   exterior_image_2_left                          0.00224   0.012
   wrist_image_left                               0.01416   0.012   <- fails
   ```

   The exteriors sit five times *under* the bar; the wrist sits above it, and it was
   the **sole** failing camera in every rejection observed. Because the suite takes the
   strictest verdict across cameras, one camera was rejecting two thirds of every
   suite - and because earlier runs saved only one camera's media, the cause was
   invisible for two full suites.

   This is not a threshold artefact. Inspecting the rejected wrist renders directly
   shows X2 duplicating the cup, hallucinating a **human hand** into the gripper's
   view, and adding concentric ripple artefacts (`runs/gatesweep-ep0-20260821-161434/
   media/high_gloss_table_reflection.jpg`, bottom row). The gate is correct to reject
   them. Plausible mechanism, **INFERRED**: the wrist view is a fast-moving, motion
   blurred close-up, which is the hardest case for a streaming video editor to hold
   structure through - but the mechanism is not verified, only the effect.

   Consequence for the design: `--views` selects which cameras a suite perturbs.
   Restricting to the two exteriors trades intervention strength (the policy keeps one
   clean channel, and this policy is known to route around a single degraded camera)
   for a far higher share of renders that survive the gate.

7. **X2 carries out a content prompt about a third of the time. MEASURED.**

   An independent vision model (`gemini-robotics-er-2-preview`), shown only the before
   and after frames and never told what the experiment found, adjudicated all 21 renders
   of one suite (`tools/judge_run.py` on `runs/garage-ep0-20260821-142420`):

   ```
    6  applied              the render shows what was asked for
    1  partially_applied
    7  not_applied          "no visible differences between the original and edited frames"
    7  something_else       a different change entirely
   ```

   The `something_else` cases are not near-misses. Asked for a greasy lens smudge, X2
   added a glass bowl of red liquid. Asked for yellow gripper fingers, it replaced the
   gripper with a yellow-and-black robot toy. Asked for dust, it added two extra yellow
   cups. **Adding objects nobody asked for is X2's characteristic failure mode**, and it
   is consistent with the separately verified findings that it will not remove an object
   on request (the target's pixel count rose to 1.10x) and will not insert a named one
   (the box never appeared; the cup amplified x4.04-5.07).

   Consequence for the design: a situation's *name* is a hypothesis about the render,
   not a fact about it. Every suite now adjudicates each render before its result is
   reported, and a finding whose render does not show the situation is labelled as a
   real behaviour change with an unsupported cause. The adjudication is an opinion, not
   a measurement, and it gates nothing.


---

## 2. `reactor/cosmos-nano-policy-droid` - the policy under test

**A real vision-language-action policy served over the same API as the renderer, so the
whole loop runs with no local GPU.**

| Field | Value | Status |
|---|---|---|
| MODEL | `reactor/cosmos-nano-policy-droid` | VERIFIED |
| DESCRIPTION (server) | "Cosmos3-Nano-Policy-DROID. Video+proprio-in -> action-out (data channel)." | VERIFIED |
| INPUTS | 3 sendonly video tracks: `wrist_view`, `exterior_view_1`, `exterior_view_2` - the DROID camera rig | VERIFIED |
| INPUTS (control) | `set_task_description` (<=300 chars), `set_proprio_json`, `set_executed_step_json` | VERIFIED |
| OUTPUTS | `action_prediction` webhook: `{step: int, action: [horizon][dof]}` | VERIFIED |
| ACTION SHAPE | **`[32][8]`** - 32-step horizon x (7 joint positions + 1 gripper), DROID joint-position convention | VERIFIED (schema text) |
| CONNECT LATENCY | 9.5 s | VERIFIED |
| RESET | `reset` clears the flow-control step counter | VERIFIED |

### Control protocol (verified from schema)

```
push frames (3 tracks) + set_proprio_json + set_task_description
        -> action_prediction {step, action[32][8]}
        -> set_executed_step_json {"step": <strictly increasing>, "action": [[...]]}
        -> next action_prediction
```

The next prediction is emitted **only** once `step` strictly increases. This is a
hard flow-control handshake, not an optional acknowledgement.

### Why this is the right policy for PENUMBRA

- It consumes **exactly** the DROID observation format, so real DROID episodes are
  a drop-in - physics, geometry, ego-motion and proprio all come from the recording.
- It emits a **continuous 8-DoF action chunk**, so action divergence is well defined
  (normalized L2, gripper-state flips, per-joint attribution) rather than forced.
- Any subset of the three camera streams can be perturbed (`--views`) while proprioception
  stays untouched, so the intervention is visual only.

---

## 3. Account / platform

| Fact | Value | Status |
|---|---|---|
| `GET /me` | roles `account_admin`, `billing_admin`, `developer` | VERIFIED |
| Token exchange | `POST /tokens` with `Reactor-API-Key` returns a JWT. Model-scoped when a model is named | VERIFIED |
| Token lifetime | 3600 s (`exp - iat`) | VERIFIED |
| Model access errors | `403 {"error":"requested model is not available to this API key"}` for models outside the key's grant | VERIFIED |
| Concurrency limit | **UNVERIFIED** - not yet measured. Sets the search budget. | UNVERIFIED |
| Recordings / clips | `request_clip`, `request_recording`, `download_clip` exist in the SDK | INFERRED (present in SDK, not yet exercised) |

---

## 4. Verification status summary

| Claim | Status |
|---|---|
| X2 reachable programmatically | **VERIFIED** |
| X2 accepts arbitrary decoded video frames | **VERIFIED** (API surface); end-to-end pixel round-trip pending |
| X2 spatial placement in SDK (not playground-only) | **VERIFIED** |
| X2 reference-image object insertion in SDK | **VERIFIED** |
| X2 frame callback delivers NumPy arrays | **VERIFIED** |
| X2 seed / determinism | **VERIFIED ABSENT** |
| X2 numeric strength parameter | **VERIFIED ABSENT** - must be prompt-encoded |
| Real VLA policy available on Reactor | **VERIFIED** |
| SANA-Streaming usable for corroboration | **UNVERIFIED** - probe pending |
| Session concurrency limit | **UNVERIFIED** |
| Practical end-to-end throughput / cost | **UNVERIFIED** - must be measured |
