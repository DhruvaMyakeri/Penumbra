"""The garage: run a policy through directed scenarios and document what breaks it.

This is the product. Everything else in PENUMBRA is machinery underneath it.

    real robot episode
      -> director proposes situations grounded in the task and the actual workspace
      -> each situation rendered onto ALL camera views by X2
      -> SEAM validity gate rejects corrupted renders before the policy sees them
      -> policy rollouts
      -> permutation test against the NO-OP RE-RENDER, not the raw episode
      -> verdict, with the situation attached

## Why the control is a no-op re-render, not the untouched episode

X2 regenerates every pixel whatever the prompt asks. Measured: an edit prompt saying
"the scene is exactly as it is, no changes" still shifts this policy's action
distribution (p = 0.0030, d = +2.32) and moves its gripper decisions. So comparing a
scenario against the *untouched* recording measures the scenario plus the re-render,
and the re-render alone is enough to reach significance.

Comparing against a no-op re-render cancels it. That control is validated: two
independent no-op runs are behaviourally indistinguishable from each other
(p = 0.32, d = 0.05), so the measurement is stable enough to attribute a difference to
the scenario rather than to generative variance.

A scenario is only reported as a vulnerability if its rollouts separate from the
no-op's. Anything less is the video model, not the robot.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import RUNS_DIR
from ..media import write_h264
from ..episodes.types import VIEWS, Episode
from ..evaluation.grouptest import GroupComparison, compare_groups
from ..evaluation.multiple import correct
from ..experiments.runner import ExperimentRunner
from ..perturbation.spec import FaultSpec, Rung
from ..perturbation.x2 import X2Perturbation
from ..policy.base import PolicyTrace
from ..validation.seam import SeamGate, perceptual_distance
from .director import Scenario, propose_suite
from .judge import judge_views
from .render_agent import classify, revise_prompt

log = logging.getLogger("penumbra.garage")

#: The control prompt. Asks for nothing, so what it measures is the re-render alone.
NULL_PROMPT = "the scene is exactly as it is, an ordinary well-lit room, no changes"

#: How far apart two X2 renders of the SAME no-op prompt land, measured on the real
#: episode: RMSE 0.026 between `runs/perturb-null_edit-20260821-134102` and `-135946`.
#: This is the model's own render-to-render variation and nothing below it can be a
#: treatment.
NO_OP_FLOOR_RMSE = 0.026
#: A render within twice that of the control is not a situation - the editor declined.
#: Testing it anyway spends rollouts on a null and, worse, inflates the multiplicity
#: correction so that every *real* candidate in the suite is held to a stricter bar.
#: The factor of two is the same log-space margin used for the gate thresholds.
NO_CHANGE_RMSE = 2.0 * NO_OP_FLOOR_RMSE


def _scenario_fault(name: str, prompt: str, description: str = "") -> FaultSpec:
    return FaultSpec(
        name=name,
        family="scenario",
        model="xmax/x2",
        description=description,
        ladder=(Rung(0.0, "", "untouched"), Rung(1.0, prompt, name)),
        version="director-1",
        validation={"max_geometry_drift": 0.0055, "min_structure_retained": 0.20},
    )


@dataclass
class ScenarioResult:
    """One situation, fully evaluated against the no-op control."""

    scenario: Scenario
    status: str = "queued"
    gate: dict | None = None
    distance: dict | None = None
    vs_control: GroupComparison | None = None
    vs_baseline: GroupComparison | None = None
    seconds: float = 0.0
    error: str | None = None
    preview: str | None = None
    video: str | None = None
    per_view_distance: dict | None = None
    per_view_gate: dict | None = None
    distance_from_control: dict | None = None
    render_attempts: list | None = None
    #: The prompt that produced the render actually tested. Differs from the director's
    #: original whenever the render agent had to rewrite it.
    prompt_used: str | None = None
    intent: dict | None = None
    stage: str = "queued"
    rollouts: int = 0
    passes_fdr: bool | None = None
    passes_bonferroni: bool | None = None

    @property
    def is_vulnerability(self) -> bool:
        """A discovery: separated from the no-op AND survived suite-wide correction.

        Raw significance is not enough once a suite runs twenty-odd tests against one
        control - at that size, chance alone produces apparent hits. `passes_fdr` is
        set by the suite-level correction after every scenario has been tested; until
        then a scenario can be significant without yet being a discovery.
        """
        if self.status != "done" or self.vs_control is None:
            return False
        if not self.vs_control.any_significant:
            return False
        return self.passes_fdr is not False

    def verdict(self) -> str:
        if self.status == "no_change":
            worst = max((d["rmse"] for d in (self.distance_from_control or {}).values()),
                        default=0.0)
            return (f"NOT A SITUATION - the editor produced a render within its own "
                    f"no-op variation (rmse {worst:.4f} from the control, floor "
                    f"{NO_CHANGE_RMSE:.4f}). Nothing was applied, so there is nothing "
                    f"to test and no rollouts were spent.")
        if self.status == "rejected":
            return "RENDER REJECTED - the transformation corrupted the scene, so this " \
                   "says nothing about the policy"
        if self.status == "error":
            return f"FAILED - {self.error}"
        if self.status != "done":
            return "pending"
        c = self.vs_control
        if c is None:
            return "no control comparison available"
        if c.any_significant:
            which = []
            if c.significant:
                which.append(f"trajectory (p={c.p_value:.4f}, d={c.effect_size:+.2f})")
            if c.significant_gripper:
                which.append(f"grasp decision (p={c.gripper_p_value:.4f})")
            verdict = ("VULNERABILITY - policy behaviour separates from a no-op "
                       "re-render of the same episode: " + " and ".join(which))
            # The statistics stand on their own; the situation's NAME does not. If an
            # independent judge says the render shows something other than what was
            # asked for, the effect is still real but "amber lighting broke it" is not
            # a claim this run can make, and the card has to say so where the claim is.
            if self.intent and not self.intent.get("credible", True):
                observed = str(self.intent.get("observed", "")).rstrip(".")
                verdict += (
                    f" | BUT THE NAME IS NOT SUPPORTED: an independent vision model "
                    f"judged the render '{self.intent.get('verdict')}'"
                    + (f" - {observed}" if observed else "")
                    + ". The behaviour change is real; its stated cause is not."
                )
            return verdict
        return (
            f"no effect beyond re-rendering (p={c.p_value:.4f}) - this situation did "
            f"not move the policy more than asking the model for nothing"
        )

    def to_dict(self) -> dict:
        return {
            **self.scenario.to_dict(),
            "status": self.status,
            "gate": self.gate,
            "distance": self.distance,
            "vs_control": self.vs_control.to_dict() if self.vs_control else None,
            "vs_baseline": self.vs_baseline.to_dict() if self.vs_baseline else None,
            "is_vulnerability": self.is_vulnerability,
            "verdict": self.verdict(),
            "seconds": round(self.seconds, 1),
            "error": self.error,
            "preview": self.preview,
            "video": self.video,
            "per_view_distance": self.per_view_distance,
            "per_view_gate": self.per_view_gate,
            "distance_from_control": self.distance_from_control,
            "render_attempts": self.render_attempts,
            "prompt_used": self.prompt_used,
            "intent": self.intent,
            "stage": self.stage,
            "rollouts": self.rollouts,
            "passes_fdr": self.passes_fdr,
            "passes_bonferroni": self.passes_bonferroni,
        }


class Garage:
    """Runs an episode through directed scenarios, streaming state as it goes."""

    def __init__(
        self,
        runner: ExperimentRunner,
        out_dir: Path,
        *,
        views: tuple[str, ...] = VIEWS,
        repeats: int = 6,
        screen_repeats: int = 3,
        gate_retries: int = 2,
        max_render_attempts: int = 3,
        render_settings: dict | None = None,
        on_update: Callable[[dict], None] | None = None,
    ) -> None:
        self.runner = runner
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "media").mkdir(exist_ok=True)
        self.views = views
        # Two-stage budget. Policy rollouts dominate the cost of a suite, so every
        # scenario is screened cheaply and only survivors get the full group. A
        # scenario that shows nothing at N=6 control vs M=3 is not going to become a
        # discovery at M=6, and screening lets the suite cover three times the ground
        # for the same spend.
        self.repeats = repeats
        self.screen_repeats = min(screen_repeats, repeats)
        #: How many times a camera whose render *marginally* failed the gate may be
        #: redrawn. Zero restores the strict single-draw behaviour.
        self.gate_retries = gate_retries
        #: How many times the render agent may rewrite the prompt and try again. Each
        #: attempt is a full render, so this multiplies the expensive half of a suite.
        self.max_render_attempts = max_render_attempts
        #: How X2 is streamed. Defaults measured by `tools/tune_x2.py`: with a vivid
        #: prompt every configuration applied the edit, and this was the only one that
        #: came back applied, gate-valid and with nothing invented. That comparison is a
        #: SINGLE draw per configuration and X2 has no seed, so it is a defensible
        #: default rather than a demonstrated optimum - the differences sit inside the
        #: model's own draw-to-draw spread. It costs roughly 2.5x the render time.
        self.render_settings = dict(render_settings or {"prime_passes": 1,
                                                        "push_fps": 7.0})
        self.on_update = on_update
        self.results: list[ScenarioResult] = []
        self.state: dict = {"phase": "starting", "scenarios": []}
        self._control_traces: list[PolicyTrace] = []
        self._control_meta: dict = {}
        self._control_episode: Episode | None = None
        # Rendered episodes and accumulated rollouts, kept so the confirmation stage
        # neither re-renders nor re-rolls what screening already paid for.
        self._episodes: dict = {}
        self._traces: dict = {}

    # -- state streaming ---------------------------------------------------

    def _publish(self, **patch) -> None:
        self.state.update(patch)
        self.state["scenarios"] = [r.to_dict() for r in self.results]
        self.state["updated"] = time.strftime("%H:%M:%S")
        self.state["cost"] = self.runner.cost
        (self.out / "state.json").write_text(
            json.dumps(self.state, indent=2, default=str), encoding="utf-8"
        )
        if self.on_update:
            try:
                self.on_update(self.state)
            except Exception:  # noqa: BLE001
                log.debug("state hook failed", exc_info=True)

    def _save_media(self, name: str, perturbed: Episode, source: Episode,
                    fps: float) -> tuple[str, str, dict]:
        """Media for the UI - every perturbed camera, not just the first.

        The intervention hits all three views because that is the only way it reaches
        this policy, so showing one of them hides two thirds of what the policy saw.
        The video stacks source-over-perturbed per camera, side by side, so a viewer
        can check each POV against its own original rather than against memory.
        """
        views = list(self.views)
        tile_w, tile_h = 300, 169

        def _strip(stack: np.ndarray) -> np.ndarray:
            idx = np.linspace(0, len(stack) - 1, 3).astype(int)
            return np.concatenate([cv2.resize(stack[i], (tile_w, tile_h)) for i in idx],
                                  axis=1)

        # Contact sheet: one row per camera, source above perturbed.
        rows = []
        for view in views:
            rows.append(np.concatenate(
                [_strip(source.frames[view]), _strip(perturbed.frames[view])], axis=1))
        sheet = np.concatenate(rows, axis=0)
        pre = f"media/{name}.jpg"
        cv2.imwrite(str(self.out / pre), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82])

        # Video: cameras side by side, perturbed under source. H.264, because OpenCV's
        # mp4v is MPEG-4 Part 2 and no browser will decode it - which made every
        # generation invisible in the dashboard while the files sat on disk.
        n = min(len(perturbed.frames[v]) for v in views)
        vid = f"media/{name}.mp4"

        def _grid():
            for i in range(n):
                top = np.concatenate(
                    [cv2.resize(source.frames[v][i], (tile_w, tile_h)) for v in views], axis=1)
                bot = np.concatenate(
                    [cv2.resize(perturbed.frames[v][i], (tile_w, tile_h)) for v in views], axis=1)
                yield np.concatenate([top, bot], axis=0)

        write_h264(self.out / vid, _grid(), fps)

        per_view = {
            v: {k: round(x, 5) for k, x in
                perceptual_distance(source.frames[v], perturbed.frames[v]).items()}
            for v in views
        }
        return pre, vid, per_view

    def _save_frames(self, name: str, rendered: Episode) -> None:
        """Store the perturbed views losslessly, keyed by scenario name."""
        d = self.out / "frames"
        d.mkdir(exist_ok=True)
        np.savez_compressed(d / f"{name}.npz",
                            **{v: rendered.frames[v] for v in self.views})

    def _load_frames(self, name: str, episode: Episode) -> Episode | None:
        """Rebuild a rendered episode from disk, or None if it was never saved."""
        path = self.out / "frames" / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            missing = [v for v in self.views if v not in z]
            if missing:
                log.info("%s: saved frames lack %s - re-rendering", name, missing)
                return None
            return episode.with_views({v: z[v] for v in self.views},
                                      f"{episode.episode_id}#{name}[resumed]")

    def resume(self, episode: Episode) -> int:
        """Restore a previous run of this directory. Returns how many renders survived.

        Only renders whose frames are on disk are restored; a scenario that was mid-flight
        when the run stopped is put back in the queue rather than half-trusted.
        """
        state_path = self.out / "state.json"
        if not state_path.exists():
            return 0
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        restored = 0
        by_name = {r.scenario.name: r for r in self.results}
        for item in prior.get("scenarios", []):
            res = by_name.get(item["name"])
            if res is None or item["status"] not in ("rendered", "done", "rejected",
                                                     "no_change"):
                continue
            frames = self._load_frames(item["name"], episode)
            if item["status"] in ("rejected", "no_change"):
                # No frames needed: the verdict is final and cost nothing to keep.
                for key in ("gate", "per_view_gate", "distance", "distance_from_control",
                            "intent", "preview", "video", "render_attempts",
                            "per_view_distance"):
                    setattr(res, key, item.get(key))
                res.status = item["status"]
                restored += 1
                continue
            if frames is None:
                continue
            self._episodes[item["name"]] = frames
            for key in ("gate", "per_view_gate", "distance", "distance_from_control",
                        "intent", "preview", "video", "render_attempts",
                        "per_view_distance"):
                setattr(res, key, item.get(key))
            res.status = "rendered"
            res.stage = "awaiting screening"
            restored += 1
        log.info("resume: %d of %d situations restored from %s",
                 restored, len(self.results), self.out.name)
        return restored

    # -- phases ------------------------------------------------------------

    async def establish_control(self, episode: Episode) -> None:
        """Render the episode with a no-op prompt; this is what scenarios compare to."""
        self._publish(phase="control", phase_detail="rendering the no-op control")
        fault = _scenario_fault("null_control", NULL_PROMPT,
                                "no-op re-render: the control every scenario is measured against")
        t0 = time.time()
        perturber = X2Perturbation(views=self.views, **self.render_settings)
        result = await perturber.apply(episode, fault, 1.0, run_id="control")
        frames = result.episode.frames[self.views[0]]
        gate = SeamGate(max_geometry_drift=fault.max_geometry_drift,
                        min_structure_retained=fault.min_structure_retained)
        report = gate.validate(episode.frames[self.views[0]], frames)

        self._publish(phase="control", phase_detail="rolling out the control")
        traces = [
            await self.runner._rollout(result.episode, f"control_{i}")
            for i in range(self.repeats)
        ]
        self._control_traces = traces
        # Kept so each scenario can be compared against what the model produces when
        # asked for nothing, which is the only reference that can tell "the editor
        # declined" apart from "the situation had no effect".
        self._control_episode = result.episode
        pre, vid, per_view = self._save_media("control", result.episode, episode, episode.fps)
        self._control_meta = {
            "prompt": NULL_PROMPT,
            "gate": report.to_dict(),
            "distance": perceptual_distance(episode.frames[self.views[0]], frames),
            "per_view_distance": per_view,
            "views": list(self.views),
            "render_settings": dict(self.render_settings),
            # Which cameras were left at their true appearance. A finding made while
            # the policy still had a clean channel is a stronger finding, and a reader
            # cannot judge that without knowing what was untouched.
            "views_untouched": [v for v in VIEWS if v not in self.views],
            "rollouts": len(traces),
            "seconds": round(time.time() - t0, 1),
            "preview": pre,
            "video": vid,
            "vs_baseline": compare_groups(self.runner.baseline_group, traces).to_dict()
            if len(self.runner.baseline_group) >= 2 else None,
        }
        self._publish(phase="control_done", control=self._control_meta)

    async def _render_once(self, episode: Episode, scenario, prompt: str,
                           label: str) -> dict:
        """One full generation: render, gate (redrawing a marginal camera), no-change
        check, and - only when it is worth the money - adjudicate what it shows."""
        fault = _scenario_fault(scenario.name, prompt, scenario.situation)
        perturber = X2Perturbation(views=self.views, **self.render_settings)
        out = await perturber.apply(episode, fault, 1.0, run_id=label)
        rendered = out.episode

        gate = SeamGate(max_geometry_drift=fault.max_geometry_drift,
                        min_structure_retained=fault.min_structure_retained)
        reports = {v: gate.validate(episode.frames[v], rendered.frames[v])
                   for v in self.views}

        # X2 has no seed, so a camera that misses a bar by a fraction of a percent is a
        # draw from a distribution straddling the bar, not a property of the situation.
        # Redraw those. The threshold is untouched; the alternative discards the most
        # realistic generations in a suite over margins as small as 0.25%.
        redraws = 0
        for _ in range(self.gate_retries):
            failing = [v for v, r in reports.items() if not r.valid]
            if not failing or not all(reports[v].marginal for v in failing):
                break  # a render far past the bar is corrupt, not unlucky
            log.info("%s: redrawing %s (marginal gate failure)",
                     scenario.name, ", ".join(failing))
            self._publish(phase_detail=f"redrawing {scenario.name} "
                                       f"({len(failing)} camera(s) marginal)")
            redraw = await X2Perturbation(views=failing, **self.render_settings).apply(
                episode, fault, 1.0, run_id=f"{label}-redraw")
            rendered = rendered.with_views(
                {v: redraw.episode.frames[v] for v in failing}, rendered.episode_id)
            reports = {v: gate.validate(episode.frames[v], rendered.frames[v])
                       for v in self.views}
            redraws += 1

        report = min(reports.values(), key=lambda r: (r.valid, -r.geometry_drift))
        result = {
            "rendered": rendered,
            "gate": report.to_dict(),
            "per_view_gate": {v: r.to_dict() for v, r in reports.items()},
            "distance": perceptual_distance(episode.frames[self.views[0]],
                                            rendered.frames[self.views[0]]),
            "redraws": redraws,
            "declined": False,
            "distance_from_control": None,
            "intent": None,
        }
        if not report.valid:
            return result

        # Did the editor change anything? Measured against the CONTROL, because the
        # source comparison includes the re-render that every render carries and that
        # the control exists to cancel.
        if self._control_episode is not None:
            vs_control = {v: perceptual_distance(self._control_episode.frames[v],
                                                 rendered.frames[v])
                          for v in self.views}
            result["distance_from_control"] = {
                v: {k: round(x, 5) for k, x in d.items()} for v, d in vs_control.items()}
            if max(d["rmse"] for d in vs_control.values()) < NO_CHANGE_RMSE:
                result["declined"] = True
                return result

        self._publish(phase_detail=f"checking what {scenario.name} actually shows")
        result["intent"] = judge_views(prompt, scenario.situation, episode.frames,
                                       rendered.frames, self.views).to_dict()
        return result

    async def run_scenario(self, episode: Episode, res: ScenarioResult) -> ScenarioResult:
        """Render a situation, and keep working on it until it is worth testing.

        Every generation faces the validity gate and the intent judge. If either says
        no, the failure in its own words goes to an agent that rewrites the PROMPT -
        never the situation - and the render is attempted again. Rewriting the situation
        until something breaks would be fishing; rewriting the phrasing until the
        renderer cooperates is operating the instrument.

        This is selection, and it is recorded rather than hidden. Every attempt, its
        prompt, its outcome and the agent's reasoning are kept in `render_attempts`: a
        situation that needed three tries is weaker evidence than one that worked first
        time, and the record has to let a reader see that.
        """
        scenario = res.scenario
        res.status = "generating"
        if res not in self.results:
            self.results.append(res)
        self._publish(phase="scenarios", phase_detail=f"rendering {scenario.name}")
        t0 = time.time()
        prompt = scenario.prompt
        history: list[dict] = []
        outcome, outcome_detail = "", ""

        try:
            for attempt in range(1, self.max_render_attempts + 1):
                if attempt > 1:
                    self._publish(
                        phase_detail=f"re-rendering {scenario.name} "
                                     f"(attempt {attempt}, last was {outcome.lower()})")
                r = await self._render_once(episode, scenario, prompt,
                                            f"garage-a{attempt}")
                outcome, outcome_detail = classify(r["gate"], r["intent"], r["declined"])
                history.append({
                    "attempt": attempt,
                    "prompt": prompt,
                    "outcome": outcome,
                    "detail": outcome_detail,
                    "redraws": r["redraws"],
                    "gate": r["gate"],
                    "views": r["per_view_gate"],
                    "intent": r["intent"],
                })

                # Keep the record and the media current with the attempt just made, so
                # an interrupted run still shows what it actually produced.
                res.render_attempts = history
                res.gate = r["gate"]
                res.per_view_gate = r["per_view_gate"]
                res.distance = r["distance"]
                res.distance_from_control = r["distance_from_control"]
                res.intent = r["intent"]
                res.prompt_used = prompt
                res.preview, res.video, res.per_view_distance = self._save_media(
                    scenario.name, r["rendered"], episode, episode.fps)
                self._save_frames(scenario.name, r["rendered"])
                self._publish()

                if outcome == "usable":
                    res.status = "rendered"
                    res.stage = "awaiting screening"
                    self._episodes[scenario.name] = r["rendered"]
                    break

                log.info("%s attempt %d: %s - %s", scenario.name, attempt, outcome,
                         outcome_detail[:120])

                # Once the scene has survived the gate and something visibly changed,
                # the only complaint left is the judge's - and that is not a failure a
                # prompt can fix. Measured over one 21-situation suite: 44 attempts
                # were spent after the gate had already passed, and the failure never
                # migrated back. What blocks those renders is X2 replacing an object
                # instead of resurfacing it, which no rewording addresses.
                #
                # So stop here and test the render, carrying the judge's verdict as the
                # caveat it always was. This is not purely a saving: the situation is
                # now tested on the render that first passed the gate rather than on a
                # later draw, which is a different render.
                if not r["declined"] and r["gate"]["valid"]:
                    log.info("%s: gate satisfied; the judge's objection is not a "
                             "prompt problem, so testing this render", scenario.name)
                    res.status = "rendered"
                    res.stage = "awaiting screening"
                    self._episodes[scenario.name] = r["rendered"]
                    break

                last_attempt = attempt == self.max_render_attempts
                new_prompt, reasoning = ("", "")
                if not last_attempt:
                    self._publish(
                        phase_detail=f"rewriting the prompt for {scenario.name}")
                    new_prompt, reasoning = revise_prompt(
                        scenario.situation, scenario.why_it_might_break, history)
                    history[-1]["agent_reasoning"] = reasoning

                if last_attempt or not new_prompt or new_prompt == prompt:
                    # Out of attempts, or the agent had nothing new to try. A gate
                    # rejection or a decline is final. A render the judge merely
                    # disbelieves is still tested - the judge is an opinion and the
                    # statistics are not, so its verdict travels with the result
                    # instead of suppressing it.
                    if r["declined"]:
                        res.status = "no_change"
                    elif not r["gate"]["valid"]:
                        res.status = "rejected"
                    else:
                        res.status = "rendered"
                        res.stage = "awaiting screening"
                        self._episodes[scenario.name] = r["rendered"]
                    break

                prompt = new_prompt
        except Exception as exc:  # noqa: BLE001
            log.exception("scenario %s failed", scenario.name)
            res.status = "error"
            res.error = f"{type(exc).__name__}: {exc}"
        res.seconds = time.time() - t0
        self._publish()
        return res

    async def test_scenario(self, res: ScenarioResult, *, repeats: int,
                            stage: str) -> ScenarioResult:
        """Roll the policy out on an already-rendered scenario and test it.

        Additive: rollouts accumulate across stages, so confirming a screened
        scenario costs only the extra rollouts rather than repeating the group.
        """
        scenario = res.scenario
        episode = self._episodes.get(scenario.name)
        if episode is None:
            res.status = "error"
            res.error = "no rendered episode retained for this scenario"
            self._publish()
            return res
        t0 = time.time()
        res.status = "rolling_out"
        res.stage = stage
        self._publish(phase_detail=f"{stage}: {scenario.name} ({repeats} rollouts)")
        try:
            existing = self._traces.setdefault(scenario.name, [])
            for i in range(len(existing), repeats):
                existing.append(
                    await self.runner._rollout(episode, f"{scenario.name}_{i}")
                )
            res.rollouts = len(existing)
            res.vs_control = compare_groups(self._control_traces, existing)
            if len(self.runner.baseline_group) >= 2:
                res.vs_baseline = compare_groups(self.runner.baseline_group, existing)
            res.status = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("scenario %s test failed", scenario.name)
            res.status = "error"
            res.error = f"{type(exc).__name__}: {exc}"
        res.seconds += time.time() - t0
        self._publish()
        return res

    async def run(self, episode: Episode, *, n_scenarios: int = 5,
                  baseline_runs: int = 6, resume: bool = False) -> dict:
        task = episode.task.split("|")[0].strip()
        self.state = {
            "phase": "starting",
            "episode": episode.episode_id,
            "task": task,
            "views": list(self.views),
            "render_settings": dict(self.render_settings),
            # Which cameras were left at their true appearance. A finding made while
            # the policy still had a clean channel is a stronger finding, and a reader
            # cannot judge that without knowing what was untouched.
            "views_untouched": [v for v in VIEWS if v not in self.views],
            "policy": self.runner.policy.name,
            "repeats": self.repeats,
            "started": time.strftime("%H:%M:%S"),
            "scenarios": [],
        }
        self._publish()

        self._publish(phase="baseline", phase_detail="calibrating the policy's noise floor")
        nf = await self.runner.calibrate(episode, runs=baseline_runs)
        self._publish(noise_floor=nf.to_dict())

        self._publish(phase="director", phase_detail="proposing situations for this task")
        per_category = max(1, -(-n_scenarios // 7))  # ceil over the category count

        def _director_progress(category, scenarios, n_categories):
            self.results = [ScenarioResult(scenario=s, status="queued")
                            for s in scenarios]
            self._publish(
                phase="director",
                phase_detail=f"proposed {len(scenarios)} situations "
                             f"(through category '{category}')",
            )

        prior_path = self.out / "state.json"
        if resume and prior_path.exists():
            # Reuse the previous run's situations verbatim. Re-asking the director
            # would produce a different list, and then the restored renders would
            # belong to situations that are no longer in the suite.
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            chosen = [Scenario(name=x["name"], situation=x.get("situation", ""),
                               why_it_might_break=x.get("why_it_might_break", ""),
                               prompt=x["prompt"], severity=int(x.get("severity", 3)),
                               realism=int(x.get("realism", 3)),
                               source=x.get("source", ""),
                               category=x.get("category", ""))
                      for x in prior.get("scenarios", []) if x.get("prompt")]
            self.results = [ScenarioResult(scenario=c, status="queued") for c in chosen]
            self._publish(phase="director",
                          phase_detail=f"resumed {len(chosen)} situations from a "
                                       f"previous run of this directory")
            self.resume(episode)
        else:
            proposal = propose_suite(task, {v: episode.frames[v][0] for v in self.views},
                                     per_category=per_category,
                                     on_progress=_director_progress)
            if proposal.error:
                self._publish(phase="error", error=f"director: {proposal.error}")
                raise RuntimeError(f"director failed: {proposal.error}")
            # Seed every proposed situation as queued straight away. The plan is worth
            # seeing before any of it has run - it is the director's reasoning about
            # this specific task, and waiting until each one finishes hides it.
            chosen = proposal.scenarios[:n_scenarios]
            self.results = [ScenarioResult(scenario=c, status="queued") for c in chosen]
            self._publish(director={"model": chosen[0].source if chosen else None,
                                    "proposed": len(proposal.scenarios),
                                    "categories": sorted({c.category for c in chosen})})

        await self.establish_control(episode)

        # Stage 1 - render and gate everything. Cheap relative to rollouts, and it
        # removes corrupted renders from the family before any test is run.
        self._publish(phase="rendering",
                      phase_detail=f"rendering {len(chosen)} situations")
        for result in list(self.results):
            if result.status != "queued":
                continue  # restored from a previous run of this directory
            await self.run_scenario(episode, result)

        renderable = [r for r in self.results if r.status == "rendered"]

        # Stage 2 - screen every valid render with a small group.
        self._publish(phase="screening",
                      phase_detail=f"screening {len(renderable)} valid renders")
        for result in renderable:
            await self.test_scenario(result, repeats=self.screen_repeats,
                                     stage="screening")

        # Stage 3 - confirm anything that showed a signal, with the full group.
        promising = [r for r in renderable
                     if r.vs_control is not None and r.vs_control.any_significant]
        if self.repeats > self.screen_repeats and promising:
            self._publish(phase="confirming",
                          phase_detail=f"confirming {len(promising)} candidates at "
                                       f"{self.repeats} rollouts")
            for result in promising:
                await self.test_scenario(result, repeats=self.repeats,
                                         stage="confirmation")

        self._apply_correction()

        found = [r for r in self.results if r.is_vulnerability]
        self._publish(
            phase="done",
            phase_detail=f"{len(found)} of {len(renderable)} tested situations moved "
                         f"the policy after correction",
            summary={
                "proposed": len(self.results),
                "rendered_valid": len(renderable),
                "tested": len([r for r in self.results if r.vs_control is not None]),
                "vulnerabilities": [r.scenario.name for r in found],
                "rejected_renders": [r.scenario.name for r in self.results
                                     if r.status == "rejected"],
                # Renders the editor declined to make. Reported separately from
                # rejections: a rejection means the render was corrupt, this means
                # there was no render to speak of, and conflating them would read as
                # the gate being stricter than it is.
                "no_change_renders": [r.scenario.name for r in self.results
                                      if r.status == "no_change"],
                "errors": [r.scenario.name for r in self.results if r.status == "error"],
            },
        )
        return self.state

    def _apply_correction(self) -> None:
        """Correct across the whole suite before anything is called a discovery.

        One control, twenty-odd tests: at alpha = 0.025 per test, chance alone
        produces apparent hits. Benjamini-Hochberg decides the shortlist; Bonferroni
        is reported alongside for anyone who needs the stricter bar.
        """
        tested = {r.scenario.name: r.vs_control.p_value
                  for r in self.results if r.vs_control is not None}
        if not tested:
            return
        suite = correct(tested)
        standing = {c.name: c for c in suite.results}
        for r in self.results:
            c = standing.get(r.scenario.name)
            if c is not None:
                r.passes_fdr = c.passes_fdr
                r.passes_bonferroni = c.passes_bonferroni
        self._publish(correction=suite.to_dict())
