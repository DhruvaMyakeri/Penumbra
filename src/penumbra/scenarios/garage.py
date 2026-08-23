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
from ..report.run_report import write_reports
from ..validation.seam import SeamGate, perceptual_distance
from .director import CATEGORIES, Scenario, _category_violation, propose_suite
from .insights import InsightLedger, RenderObservation, prompt_features, rule_violations
from .judge import judge_views
from .render_agent import REPAIRABLE, classify, revise_prompt

#: The render agent's outcome labels, as the ledger records them. Kept as an explicit
#: map rather than a lowercase() so that renaming an outcome breaks loudly here instead
#: of silently splitting the ledger's counts across two spellings.
_LEDGER_OUTCOME = {
    "NOTHING HAPPENED": "declined",
    "SCENE CORRUPTED": "gate_rejected",
    "WRONG THING RENDERED": "wrong_thing",
    "CAMERAS DISAGREED": "incoherent",
    "usable": "usable",
}

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
    def moved_the_policy(self) -> bool:
        """Statistically separated from the no-op AND survived suite-wide correction.

        Raw significance is not enough once a suite runs twenty-odd tests against one
        control - at that size, chance alone produces apparent hits. `passes_fdr` is
        set by the suite-level correction after every scenario has been tested; until
        then a scenario can be significant without yet being a discovery.

        This is a statement about numbers only. It says nothing about *what* moved the
        policy, which is what `finding_class` is for.
        """
        if self.status != "done" or self.vs_control is None:
            return False
        if not self.vs_control.any_significant:
            return False
        return self.passes_fdr is not False

    @property
    def finding_class(self) -> str:
        """What this run is entitled to claim about this situation.

        The distinction this draws is the one the project kept getting wrong. A suite
        previously reported 15 vulnerabilities; an independent judge said 3 of the 15
        renders showed the situation they were named after, and a cross-camera audit
        found 14 of them showed *different scenes on the two cameras*. Every one of
        those numbers was statistically sound. The names were fiction.

        Statistical significance earns the right to say "something in this render moved
        the policy". It does not earn the right to say *what*. Only a render that the
        gate accepted, that both cameras agree on, and that an independent judge says
        depicts the named situation earns that.

            CONFIRMED     the policy moved, and the render shows what it claims
            UNATTRIBUTED  the policy moved, but the render does not show what it claims
                          (wrong thing rendered, or the cameras disagreed). A real
                          effect with an unsupported cause - never quote the name.
            NO EFFECT     tested, did not move the policy beyond re-rendering
            REJECTED      the render corrupted the scene; says nothing about the policy
            DECLINED      the editor produced no visible change; never tested
        """
        if self.status == "no_change":
            return "DECLINED"
        if self.status == "rejected":
            return "REJECTED"
        if self.status == "error":
            return "ERROR"
        if self.status != "done":
            return "PENDING"
        if not self.moved_the_policy:
            return "NO EFFECT"
        intent = self.intent or {}
        # No judge verdict at all is not evidence of a good render. Absent adjudication
        # the name is unsupported in exactly the way it would be if the judge objected,
        # so it lands in the same bucket rather than being waved through.
        if not intent or intent.get("error"):
            return "UNATTRIBUTED"
        if intent.get("coherent") is False:
            return "UNATTRIBUTED"
        # Unknown coherence is not agreement. When more than one camera was perturbed
        # and the check could not run - a judge call failed, a camera went unjudged -
        # nothing establishes that the two views showed the same scene, which is the
        # exact gap this taxonomy exists to close. Caught on the first live run: the
        # sole CONFIRMED finding had one camera judged and `coherent: None`, and was
        # being promoted on a check that never happened.
        if len(self.per_view_gate or {}) > 1 and intent.get("coherent") is not True:
            return "UNATTRIBUTED"
        if not intent.get("credible"):
            return "UNATTRIBUTED"
        return "CONFIRMED"

    @property
    def is_vulnerability(self) -> bool:
        """Retained for the record format's stability; prefer `finding_class`.

        This is `moved_the_policy` - the statistical fact alone. It is deliberately NOT
        the headline any more: counting these as vulnerabilities is what produced a
        suite claiming fifteen findings of which three survived adjudication.
        """
        return self.moved_the_policy

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
            moved = ("policy behaviour separates from a no-op re-render of the same "
                     "episode: " + " and ".join(which))
            intent = self.intent or {}
            klass = self.finding_class

            if klass == "CONFIRMED":
                return (f"CONFIRMED - {moved}. An independent vision model judged the "
                        f"render '{intent.get('verdict')}' and both cameras agree on "
                        f"what changed, so the situation's name is supported by the "
                        f"render the policy actually saw.")

            # The statistics stand on their own; the situation's NAME does not. Say
            # precisely which of the two ways the attribution failed, because they call
            # for different fixes - a rewording, or a redraw.
            # Two distinct failures with two distinct messages. The cameras actively
            # disagreeing is a finding about the render; the check not having run is an
            # absence of evidence. Collapsing them would tell a reader the cameras
            # conflicted when in fact nobody looked.
            if intent.get("coherent") is None and len(self.per_view_gate or {}) > 1:
                why = intent.get("coherence_note") or "the check did not run"
                return (f"UNATTRIBUTED - {moved}. BUT CROSS-CAMERA AGREEMENT WAS NEVER "
                        f"ESTABLISHED: {why}. Two cameras were perturbed in separate "
                        f"unseeded passes, so without that check nothing shows they "
                        f"rendered the same scene. Not evidence of disagreement - "
                        f"evidence of nothing, which cannot carry a name.")
            if intent.get("coherent") is False:
                note = (intent.get("coherence_note")
                        or "the two cameras rendered different scenes")
                return (f"UNATTRIBUTED - {moved}. BUT THE CAMERAS DISAGREE: {note}. The "
                        f"policy read both views in the same observation, so it was "
                        f"shown two different worlds and no single situation describes "
                        f"its input. The effect is real; '{self.scenario.name}' is not "
                        f"what caused it.")
            if not intent or intent.get("error"):
                return (f"UNATTRIBUTED - {moved}. No adjudication of what the render "
                        f"shows was available, so nothing supports the name.")
            observed = str(intent.get("observed", "")).rstrip(".")
            return (f"UNATTRIBUTED - {moved}. BUT THE RENDER SHOWS SOMETHING ELSE: an "
                    f"independent vision model judged it '{intent.get('verdict')}'"
                    + (f" - {observed}" if observed else "")
                    + f". The behaviour change is real; '{self.scenario.name}' is not "
                      f"what caused it.")
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
            "moved_the_policy": self.moved_the_policy,
            "finding_class": self.finding_class,
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
        rebrief_after: int = 6,
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
        #: After this many scenarios have rendered, the prompts of everything still
        #: queued are rewritten with what the run has learned. Six is enough attempts
        #: for a pattern to show and early enough that most of the suite benefits.
        #: Zero disables it.
        self.rebrief_after = rebrief_after
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
        #: What this run has learned about the renderer, shared by every agent in it.
        #: Persisted next to the state so an interrupted run keeps its evidence, and so
        #: a reader can audit what the agents were told at the time they were asked.
        self.insights = InsightLedger(path=self.out / "insights.json")

    # -- state streaming ---------------------------------------------------

    def _publish(self, **patch) -> None:
        self.state.update(patch)
        self.state["scenarios"] = [r.to_dict() for r in self.results]
        self.state["updated"] = time.strftime("%H:%M:%S")
        self.state["cost"] = self.runner.cost
        self.state["insights"] = self.insights.to_dict()["summary"]
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
                    "rule_violations": rule_violations(prompt),
                })

                # Into the shared ledger, so scenario 19 benefits from what scenarios
                # 1-18 discovered about this renderer instead of repeating it.
                intent = r["intent"] or {}
                self.insights.record(RenderObservation(
                    scenario=scenario.name,
                    attempt=attempt,
                    prompt=prompt,
                    features=prompt_features(prompt),
                    violations=rule_violations(prompt),
                    outcome=_LEDGER_OUTCOME.get(outcome, "usable"),
                    judge_verdict=intent.get("verdict", ""),
                    invented=intent.get("added_objects") or [],
                    gate_reasons=(r["gate"] or {}).get("reasons") or [],
                    observed=intent.get("observed", ""),
                ))

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

                # An earlier version stopped here whenever the gate had passed, on the
                # argument that the judge's objection was not a failure a prompt could
                # fix. That argument has since been tested and is wrong. Prompts obeying
                # the measured structural rules render correctly 39% of the time against
                # 18% for the rest (n=92), and of the two renders in the last suite whose
                # cameras agreed with each other, both came from rule-clean prompts while
                # all fourteen rule-breaking ones disagreed. The judge's objection is
                # frequently a prompt problem, so the loop keeps working.
                last_attempt = attempt == self.max_render_attempts
                new_prompt, reasoning = ("", "")
                if not last_attempt and outcome in REPAIRABLE:
                    self._publish(
                        phase_detail=f"rewriting the prompt for {scenario.name}")
                    extra, validate = self._scenario_constraint(scenario)
                    new_prompt, reasoning = revise_prompt(
                        scenario.situation, scenario.why_it_might_break, history,
                        brief=self.insights.brief(), extra_rules=extra,
                        validate=validate)
                    history[-1]["agent_reasoning"] = reasoning

                if last_attempt or not new_prompt or new_prompt == prompt:
                    # Out of attempts, or the agent had nothing new to try. A gate
                    # rejection or a decline is final and nothing is tested. A render
                    # the judge merely disbelieves - or whose cameras disagreed - IS
                    # still tested, because the judge is an opinion and the statistics
                    # are not. What changes is the claim the result is allowed to make:
                    # `finding_class` demotes it to UNATTRIBUTED so the effect is
                    # reported without the situation's name attached to it.
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

    def _scenario_constraint(self, scenario) -> tuple[str, object]:
        """The category contract for this scenario, as prose and as a validator.

        Both the per-scenario retry loop and the mid-run re-brief rewrite prompts, and
        neither knew anything about categories. The general advice they carry pushes
        every prompt toward exactly one manipulable object, which is right for most of
        the suite and exactly wrong for the two null-control categories that must name
        none. Measured consequence: three null controls were repaired to scene-only at
        proposal time and were object prompts again by the time they rendered, so the
        one claim the suite pre-registers went untested for a third run running.
        """
        spec = CATEGORIES.get(getattr(scenario, "category", ""), None)
        if not isinstance(spec, dict) or "max_objects" not in spec:
            return "", rule_violations

        budget = spec["max_objects"]
        text = (
            f"This scenario is a NULL CONTROL for the '{scenario.category}' category. "
            f"Its prompt must name AT MOST {budget} manipulable object(s) - no cup, no "
            f"bowl, no gripper, no container, not even in passing. Change only the "
            f"table, the surroundings, or the light. This overrides the general advice "
            f"to aim at one object: the whole purpose of this scenario is to test "
            f"whether scene-only change moves the policy, and naming an object destroys "
            f"that test while appearing to satisfy every other rule."
        )

        def validate(prompt: str) -> list:
            out = list(rule_violations(prompt))
            breach = _category_violation(prompt, spec)
            if breach:
                out.append(breach)
            return out

        return text, validate

    async def _rebrief(self, episode: Episode) -> None:
        """Rewrite the prompts of situations not yet rendered, using what has been learned.

        The director commits to every prompt before a single frame exists. By the time
        the sixth situation has rendered, the run knows things the director could not
        have known - that this renderer keeps inventing bowls today, that a phrasing
        which worked in a previous suite is being declined in this one - and there is no
        reason the remaining fifteen situations should repeat the mistake.

        The SITUATIONS are untouched. Only the prompts are rewritten, and only for
        scenarios that have not run. Re-choosing what to test after seeing which tests
        are working is fishing; re-choosing how to phrase a fixed test is instrument
        operation. That line is the same one `run_scenario` draws per scenario, applied
        across the suite.

        Best-effort: a failure here leaves the original prompts in place.
        """
        pending = [r for r in self.results if r.status == "queued"]
        if not pending or not self.insights.observations:
            return
        self._publish(phase="rendering",
                      phase_detail=f"re-briefing the director on {len(pending)} "
                                   f"remaining situations")
        brief = self.insights.brief()
        rewritten = 0
        for res in pending:
            history = [{"attempt": 0, "prompt": res.scenario.prompt,
                        "outcome": "NOT YET ATTEMPTED",
                        "detail": "rewrite this prompt using what the run has learned "
                                  "about the renderer, keeping the situation identical"}]
            extra, validate = self._scenario_constraint(res.scenario)
            new_prompt, reasoning = revise_prompt(
                res.scenario.situation, res.scenario.why_it_might_break, history,
                brief=brief, extra_rules=extra, validate=validate)
            # A rewrite that breaks the contract is discarded rather than used. The
            # director's prompt already satisfied it; a "better" prompt that quietly
            # removes a control is worse than no rewrite at all.
            if new_prompt and validate(new_prompt):
                log.info("re-brief rejected for %s: %s", res.scenario.name,
                         "; ".join(validate(new_prompt))[:120])
                new_prompt = ""
            if new_prompt and new_prompt != res.scenario.prompt:
                res.render_attempts = [{
                    "attempt": 0, "prompt": res.scenario.prompt,
                    "outcome": "REWRITTEN BEFORE RENDERING",
                    "detail": f"re-briefed after {len(self.insights.observations)} "
                              f"render attempts elsewhere in this suite",
                    "agent_reasoning": reasoning,
                }]
                res.scenario.prompt = new_prompt
                rewritten += 1
        self.insights.note(
            "director",
            f"re-briefed {rewritten} of {len(pending)} pending situations after "
            f"{len(self.insights.observations)} render attempts")
        log.info("re-brief: rewrote %d of %d pending prompts", rewritten, len(pending))
        self._publish()

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
            # `total`, not a flat per-category count: the categories carry measured
            # weights, and asking for 21 situations should spend them where findings
            # have actually come from rather than three per category regardless.
            proposal = propose_suite(task, {v: episode.frames[v][0] for v in self.views},
                                     total=n_scenarios,
                                     brief=self.insights.brief(),
                                     on_progress=_director_progress)
            if proposal.error:
                self._publish(phase="error", error=f"director: {proposal.error}")
                raise RuntimeError(f"director failed: {proposal.error}")
            # Seed every proposed situation as queued straight away. The plan is worth
            # seeing before any of it has run - it is the director's reasoning about
            # this specific task, and waiting until each one finishes hides it.
            chosen = proposal.scenarios[:n_scenarios]
            self.results = [ScenarioResult(scenario=c, status="queued") for c in chosen]
            self._publish(director={
                "model": chosen[0].source if chosen else None,
                "proposed": len(proposal.scenarios),
                "categories": sorted({c.category for c in chosen}),
                # How often the director's own prompts broke the measured rules and had
                # to be sent back. A rising number here is the signal that the briefing
                # is not landing, which is otherwise invisible until the renders fail.
                "prompt_repairs": proposal.repairs,
                "still_violating": [c.name for c in chosen if c.rule_violations],
            })

        await self.establish_control(episode)

        # Stage 1 - render and gate everything. Cheap relative to rollouts, and it
        # removes corrupted renders from the family before any test is run.
        self._publish(phase="rendering",
                      phase_detail=f"rendering {len(chosen)} situations")
        for i, result in enumerate(list(self.results)):
            if result.status != "queued":
                continue  # restored from a previous run of this directory
            await self.run_scenario(episode, result)
            if i + 1 == self.rebrief_after:
                await self._rebrief(episode)

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

        by_class: dict[str, list] = {}
        for r in self.results:
            by_class.setdefault(r.finding_class, []).append(r.scenario.name)
        confirmed = by_class.get("CONFIRMED", [])
        unattributed = by_class.get("UNATTRIBUTED", [])

        # The headline is CONFIRMED, not "moved the policy". Those two numbers came
        # apart badly once - 15 reported against 3 that survived adjudication - and the
        # count a reader sees first has to be the one the evidence supports.
        detail = (f"{len(confirmed)} confirmed of {len(renderable)} tested situations")
        if unattributed:
            detail += (f"; {len(unattributed)} more moved the policy but the render "
                       f"does not support their name")
        self._publish(
            phase="done",
            phase_detail=detail,
            summary={
                "proposed": len(self.results),
                "rendered_valid": len(renderable),
                "tested": len([r for r in self.results if r.vs_control is not None]),
                # The headline. A situation whose render the gate accepted, whose two
                # cameras agree, and which an independent judge says depicts what it
                # claims - and which then moved the policy.
                "confirmed": confirmed,
                # Real, correction-surviving effects whose CAUSE is not established.
                # Reported in full and never folded into the headline: quoting one of
                # these by name is the specific dishonesty this split exists to prevent.
                "unattributed": unattributed,
                "unattributed_reasons": {
                    r.scenario.name: (
                        "cameras rendered different scenes"
                        if (r.intent or {}).get("coherent") is False
                        else f"render judged '{(r.intent or {}).get('verdict', 'unjudged')}'"
                    )
                    for r in self.results if r.finding_class == "UNATTRIBUTED"
                },
                "no_effect": by_class.get("NO EFFECT", []),
                "rejected_renders": by_class.get("REJECTED", []),
                # Renders the editor declined to make. Reported separately from
                # rejections: a rejection means the render was corrupt, this means
                # there was no render to speak of, and conflating them would read as
                # the gate being stricter than it is.
                "no_change_renders": by_class.get("DECLINED", []),
                "errors": by_class.get("ERROR", []),
                # Retained so older readers of this record keep working, and so the gap
                # between the two counts is visible rather than quietly closed.
                "moved_the_policy": [r.scenario.name for r in self.results
                                     if r.moved_the_policy],
            },
        )

        # Readable output, written last so it reflects the corrected verdicts. Never
        # allowed to sink a finished run: the evidence is already on disk in
        # state.json, and losing a suite to a formatting bug would be absurd.
        try:
            written = write_reports(self.out, self.state)
            self._publish(reports=written)
            log.info("wrote %s and %d situation reports",
                     Path(written["run_report"]).name, len(written["scenario_reports"]))
        except Exception:  # noqa: BLE001
            log.exception("could not write reports")
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
