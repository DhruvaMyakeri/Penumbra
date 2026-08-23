"""The PENUMBRA loop, end to end.

    real episode
      -> perturbation (X2, or the classical control arm)
      -> SEAM validity gate
      -> policy rollout
      -> divergence against a calibrated noise floor
      -> verdict
      -> reproducible artifact

Order matters and is enforced. The gate runs **before** the policy, so a corrupted
transformation costs one rejection rather than one false finding. A rejected
perturbation short-circuits: no rollout, no divergence, no verdict of "broke".
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..episodes.types import PERTURBED_VIEW, Episode
from ..evaluation.divergence import (
    Divergence,
    NoiseFloor,
    Verdict,
    compute_noise_floor,
    divergence,
    judge,
)
from ..evaluation.grouptest import GroupComparison, compare_groups
from ..perturbation.classical import (
    POSITIVE_CONTROLS,
    ClassicalPerturbation,
    ClassicalResult,
)
from ..perturbation.spec import FaultSpec
from ..perturbation.x2 import PerturbationResult, X2Perturbation
from ..policy.base import PolicyTrace
from ..reactor.session import TRANSPORT_ERRORS
from ..policy.cache import baseline_key, load_traces, save_traces
from ..policy.cosmos_droid import CosmosDroidPolicy
from ..validation.seam import SeamGate, ValidationReport, perceptual_distance

log = logging.getLogger("penumbra.experiment")

#: Reactor's published price for a B200 model session, used to cost an evaluation.
#: Both X2 and the policy are billed per second of session wall-clock.
USD_PER_HOUR = 6.0


def session_cost(seconds: float) -> float:
    return seconds / 3600.0 * USD_PER_HOUR


@dataclass
class Evaluation:
    """One (fault, strength) point, fully evaluated."""

    strength: float
    perturbation: PerturbationResult | ClassicalResult
    validation: ValidationReport
    distance: dict
    trace: PolicyTrace | None
    divergence: Divergence | None
    verdict: Verdict | None
    seconds: float
    rejected: bool
    traces: list[PolicyTrace] = field(default_factory=list)
    group: GroupComparison | None = None

    @property
    def broke(self) -> bool:
        """A rejected perturbation is never a break.

        When a group comparison is available it is the evidence, because the policy is
        stochastic and a single pairwise divergence cannot see through that. The
        single-pair verdict is kept for readability but decides nothing once a group
        test exists.
        """
        if self.rejected:
            return False
        if self.group is not None:
            # Both routes are significance-tested. An earlier version compared raw
            # gripper flip rates with a bare 1.5x ratio and declared a BREAK at
            # p = 0.72; a ratio between two noisy small-sample means is not evidence.
            return self.group.any_significant
        return bool(self.verdict and self.verdict.broke)

    def summary(self) -> dict:
        return {
            "strength": self.strength,
            "status": "REJECTED" if self.rejected else ("BREAK" if self.broke else "pass"),
            "validation": self.validation.to_dict(),
            "distance": {k: round(v, 5) for k, v in self.distance.items()},
            "divergence": self.divergence.to_dict() if self.divergence else None,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "group_test": self.group.to_dict() if self.group else None,
            "perturbation": self.perturbation.to_dict(),
            "seconds": round(self.seconds, 2),
        }


class ExperimentRunner:
    """Owns the policy, the gate, and the protocol that connects them."""

    def __init__(
        self,
        *,
        policy: CosmosDroidPolicy | None = None,
        gate: SeamGate | None = None,
        view: str = PERTURBED_VIEW,
    ) -> None:
        self.policy = policy or CosmosDroidPolicy()
        self.gate = gate
        self.view = view
        self._baseline_traces: list[PolicyTrace] = []
        self._noise_floor: NoiseFloor | None = None
        self._policy_seconds = 0.0
        self._perturb_seconds = 0.0
        self._retries = 0
        self._baseline_from_cache = 0
        self._baseline_key: str | None = None

    # -- cost accounting ---------------------------------------------------

    @property
    def cost(self) -> dict:
        total = self._policy_seconds + self._perturb_seconds
        return {
            "policy_session_seconds": round(self._policy_seconds, 1),
            "perturbation_session_seconds": round(self._perturb_seconds, 1),
            "total_session_seconds": round(total, 1),
            "usd_per_hour": USD_PER_HOUR,
            "estimated_usd": round(session_cost(total), 4),
            "rollout_retries": self._retries,
            "baseline_rollouts_from_cache": self._baseline_from_cache,
            "baseline_cache_key": self._baseline_key,
        }

    # -- rollout with retry ------------------------------------------------

    async def _rollout(self, episode: Episode, run_id: str, *, attempts: int = 3) -> PolicyTrace:
        """One policy rollout, retried on transient session failure.

        Reactor sessions occasionally wedge after a successful connect, and sometimes
        fail to connect at all. A single stalled session must not discard an experiment
        that has already spent several minutes of GPU time on its baseline, so a failed
        rollout is retried on a fresh session. Retries are counted into the cost,
        because they really were spent.

        This caught `(TimeoutError, RuntimeError)` until 2026-08-23, which caught none
        of the failures it was written for: every reactor_sdk error derives from
        ReactorError, which derives from Exception and NOT from RuntimeError. The retry
        looked correct in review and in the logs, and silently could not fire. A run
        died on a NETWORK_ERROR during connect, which is exactly the case this exists
        to survive. `RuntimeError` is kept alongside because the policy adapter raises
        its own on a prediction timeout.

        The retry is deliberately *not* silent: `retries` is surfaced so a reader can
        see that a group of rollouts was not all first-attempt.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            t0 = time.time()
            try:
                trace = await self.policy.rollout(episode, run_id=run_id)
                self._policy_seconds += time.time() - t0
                return trace
            except TRANSPORT_ERRORS + (RuntimeError,) as exc:
                self._policy_seconds += time.time() - t0
                self._retries += 1
                last = exc
                log.warning(
                    "rollout %s failed on attempt %d/%d (%s: %s); retrying on a fresh session",
                    run_id, attempt + 1, attempts, type(exc).__name__, exc,
                )
        raise RuntimeError(
            f"rollout {run_id} failed {attempts} times; last error: {last}"
        ) from last

    # -- baseline ----------------------------------------------------------

    async def calibrate(
        self, episode: Episode, *, runs: int = 4, use_cache: bool = True
    ) -> NoiseFloor:
        """Establish the policy's own variability on the unperturbed episode.

        Nothing may be called a failure before this exists. It is the zero point.

        The baseline is cached on disk and keyed by everything that could change it
        (episode, policy, chunk, warm-up, length). Reusing it across conditions is not
        a shortcut: it is literally the same control group, which is what makes two
        conditions comparable to each other rather than each to its own separately
        drawn baseline. `cost.baseline_rollouts_from_cache` records how many rollouts
        came from disk, so a reader can always see it happened.
        """
        if runs < 2:
            raise ValueError("a noise floor needs at least 2 runs")

        key = baseline_key(
            episode_id=episode.episode_id,
            policy=self.policy.name,
            chunk=self.policy.chunk,
            warmup=self.policy.warmup_frames,
            length=len(episode),
        )
        self._baseline_key = key

        traces: list[PolicyTrace] = []
        if use_cache:
            traces = load_traces(key, limit=runs)
            if traces:
                self._baseline_from_cache = len(traces)
                log.info("baseline: %d/%d rollouts reused from cache %s",
                         len(traces), runs, key)

        for i in range(len(traces), runs):
            trace = await self._rollout(episode, f"baseline{i}")
            traces.append(trace)
            log.info("baseline run %d/%d: %d steps", i + 1, runs, len(trace))

        if use_cache and len(traces) > self._baseline_from_cache:
            save_traces(traces, key, {
                "episode_id": episode.episode_id,
                "policy": self.policy.name,
                "chunk": self.policy.chunk,
                "warmup_frames": self.policy.warmup_frames,
                "length": len(episode),
                "task": episode.task,
            })

        self._baseline_traces = traces
        self._noise_floor = compute_noise_floor(traces)
        return self._noise_floor

    @property
    def noise_floor(self) -> NoiseFloor:
        if self._noise_floor is None:
            raise RuntimeError("call calibrate() before evaluating any perturbation")
        return self._noise_floor

    @property
    def baseline(self) -> PolicyTrace:
        """One reference run, for readable single-pair numbers and for plots."""
        if not self._baseline_traces:
            raise RuntimeError("no baseline; call calibrate() first")
        return self._baseline_traces[0]

    @property
    def baseline_group(self) -> list[PolicyTrace]:
        """All baseline runs. This, not `baseline`, is the evidence base."""
        if not self._baseline_traces:
            raise RuntimeError("no baseline; call calibrate() first")
        return self._baseline_traces

    def gate_for(self, fault: FaultSpec) -> SeamGate:
        if self.gate is not None:
            return self.gate
        return SeamGate(
            max_geometry_drift=fault.max_geometry_drift,
            min_structure_retained=fault.min_structure_retained,
        )

    # -- one point ---------------------------------------------------------

    async def evaluate(
        self,
        episode: Episode,
        fault: FaultSpec,
        strength: float,
        *,
        perturber: X2Perturbation | None = None,
        run_id: str = "0",
        k: float = 3.0,
        repeats: int = 4,
    ) -> Evaluation:
        """Perturb, validate, then roll the policy `repeats` times and judge the group.

        `repeats` exists because the policy is stochastic: the measured noise floor is
        a mean pairwise joint divergence of ~0.10 rad, larger than most perturbations
        move it. One rollout cannot be told apart from noise; a group of them can.
        """
        t0 = time.time()
        perturber = perturber or X2Perturbation(view=self.view)
        result = await perturber.apply(episode, fault, strength, run_id=run_id)
        self._perturb_seconds += result.seconds

        source = episode.frames[self.view]
        perturbed_frames = result.episode.frames[self.view]
        report = self.gate_for(fault).validate(source, perturbed_frames)
        distance = perceptual_distance(source, perturbed_frames)

        if not report.valid:
            log.warning("strength %.3f REJECTED: %s", strength, "; ".join(report.reasons))
            return Evaluation(
                strength=strength,
                perturbation=result,
                validation=report,
                distance=distance,
                trace=None,
                divergence=None,
                verdict=None,
                seconds=time.time() - t0,
                rejected=True,
            )

        traces: list[PolicyTrace] = []
        for i in range(max(1, repeats)):
            traces.append(
                await self._rollout(result.episode, f"perturbed@{strength:g}r{run_id}_{i}")
            )

        d = divergence(self.baseline, traces[0])
        v = judge(d, self.noise_floor, k=k)
        group = None
        if len(traces) >= 2 and len(self.baseline_group) >= 2:
            group = compare_groups(self.baseline_group, traces)
        return Evaluation(
            strength=strength,
            perturbation=result,
            validation=report,
            distance=distance,
            trace=traces[0],
            traces=traces,
            group=group,
            divergence=d,
            verdict=v,
            seconds=time.time() - t0,
            rejected=False,
        )

    async def evaluate_classical(
        self,
        episode: Episode,
        op: str,
        strength: float,
        *,
        fault: FaultSpec | None = None,
        run_id: str = "0",
        k: float = 3.0,
        seed: int = 0,
        repeats: int = 4,
        views: Sequence[str] | None = None,
    ) -> Evaluation:
        """The same evaluation, through a deterministic scene-blind transform.

        The transform is deterministic but the *policy* is not, so the classical arm
        gets the same group treatment as the generative one. Comparing a single
        classical rollout against a group-tested generative result would rig the
        comparison in the generative arm's favour.
        """
        t0 = time.time()
        result = ClassicalPerturbation(
            op, view=self.view, views=views, seed=seed
        ).apply(episode, strength, run_id=run_id)

        # The gate runs over every view that was actually touched, and the strictest
        # verdict wins. Validating only the first view would let a multi-view
        # perturbation corrupt the other two unnoticed.
        gate = self.gate_for(fault) if fault else SeamGate()
        reports = {
            v: gate.validate(episode.frames[v], result.episode.frames[v])
            for v in result.views
        }
        report = min(reports.values(), key=lambda r: (r.valid, -r.geometry_drift))
        for v, r in reports.items():
            if not r.valid:
                log.info("classical %s: view %s rejected: %s", op, v, "; ".join(r.reasons))

        traces: list[PolicyTrace] = []
        for i in range(max(1, repeats)):
            traces.append(
                await self._rollout(result.episode, f"classical-{op}@{strength:g}r{run_id}_{i}")
            )

        d = divergence(self.baseline, traces[0])
        v = judge(d, self.noise_floor, k=k)
        group = None
        if len(traces) >= 2 and len(self.baseline_group) >= 2:
            group = compare_groups(self.baseline_group, traces)
        return Evaluation(
            strength=strength,
            perturbation=result,
            validation=report,
            distance=result.distance,
            trace=traces[0],
            traces=traces,
            group=group,
            divergence=d,
            verdict=v,
            seconds=time.time() - t0,
            # The classical arm faces the SAME gate as the generative one. Exempting
            # it looked harmless and was not: at matched perceptual distance,
            # `blur_noise` reached its largest divergence only by producing a stream
            # with 9.28x the source's frame-to-frame motion — per-frame independent
            # noise is temporally incoherent in a way the recording never was. Letting
            # an inadmissible perturbation count as a classical "win" while rejecting
            # the generative arm for the same defect would rig the comparison.
            #
            # Positive controls are exempt, because their whole purpose is to be a
            # larger change than any admissible fault.
            rejected=(not report.valid) and op not in POSITIVE_CONTROLS,
        )
