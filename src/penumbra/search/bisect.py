"""Boundary search: the minimum perturbation strength that breaks the policy.

The headline number PENUMBRA produces is a **margin** — the weakest visual change
that moves behaviour beyond the policy's own noise. Higher is better, and it is
comparable across policies, versions and fault families.

## Why this is a ladder search and not a real bisection

X2 has no numeric strength parameter (verified absent). Strength is an ordered ladder
of prompts that PENUMBRA constructs, so the search space is a small ordered discrete
set, not a continuous interval. Bisecting a 6-rung ladder is 3 evaluations, which is
also all the budget a seedless model deserves.

Two honesty constraints are built in rather than assumed away:

- **Monotonicity is checked, not assumed.** A pure bisection would silently produce a
  wrong answer on a non-monotone ladder. `SearchMode.SWEEP` evaluates every rung and
  reports the full curve, which is the only way to see non-monotonicity at all.
  `SearchMode.BISECT` is offered for when budget is tight, and it records that it
  assumed monotonicity.
- **Rejected rungs are not answers.** A rung whose perturbation failed the SEAM gate
  carries no information about the policy. It is recorded, skipped, and never becomes
  the reported margin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from ..episodes.types import Episode
from ..experiments.runner import Evaluation, ExperimentRunner
from ..perturbation.spec import FaultSpec

log = logging.getLogger("penumbra.search")


class SearchMode(str, Enum):
    SWEEP = "sweep"
    BISECT = "bisect"


@dataclass
class SearchResult:
    """Where the boundary is, and everything that was tried to find it."""

    fault: str
    mode: str
    evaluations: list[Evaluation] = field(default_factory=list)
    margin: float | None = None
    last_passing: float | None = None
    rejected_strengths: list[float] = field(default_factory=list)
    monotonic: bool | None = None
    bonferroni_alpha: float | None = None
    significant_after_correction: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def found_break(self) -> bool:
        return self.margin is not None

    def to_dict(self) -> dict:
        return {
            "fault": self.fault,
            "mode": self.mode,
            "margin": self.margin,
            "last_passing": self.last_passing,
            "found_break": self.found_break,
            "rejected_strengths": self.rejected_strengths,
            "monotonic_in_divergence": self.monotonic,
            "bonferroni_alpha": self.bonferroni_alpha,
            "significant_after_correction": self.significant_after_correction,
            "curve": self.curve(),
            "evaluations": [e.summary() for e in self.evaluations],
            "notes": self.notes,
        }

    def curve(self) -> list[dict]:
        """strength -> divergence, for plotting and for judging monotonicity."""
        out = []
        for e in self.evaluations:
            if e.rejected or e.divergence is None:
                continue
            out.append(
                {
                    "strength": e.strength,
                    "joint_l2": e.divergence.joint_l2,
                    "gripper_flip_rate": e.divergence.gripper_flip_rate,
                    "rmse": e.distance.get("rmse"),
                    "broke": e.broke,
                    # The group test is the evidence; the single-pair numbers above
                    # are only there to make the row readable.
                    "p_value": e.group.p_value if e.group else None,
                    "effect_size": e.group.effect_size if e.group else None,
                    "between_group_divergence": e.group.between_mean if e.group else None,
                    "within_group_divergence": e.group.within_mean if e.group else None,
                }
            )
        return sorted(out, key=lambda r: r["strength"])


def _check_monotonic(curve: list[dict]) -> bool | None:
    if len(curve) < 3:
        return None
    values = [c["joint_l2"] for c in curve]
    return all(b >= a - 1e-6 for a, b in zip(values, values[1:]))


async def search_boundary(
    runner: ExperimentRunner,
    episode: Episode,
    fault: FaultSpec,
    *,
    mode: SearchMode = SearchMode.SWEEP,
    k: float = 3.0,
    run_id: str = "0",
    skip_zero: bool = True,
    repeats: int = 4,
) -> SearchResult:
    """Find the minimum strength on `fault`'s ladder at which the policy breaks."""
    result = SearchResult(fault=fault.name, mode=mode.value)
    rungs = [r.strength for r in fault.ladder]
    if skip_zero:
        # Strength 0.0 is the untouched control; the baseline already covers it.
        rungs = [s for s in rungs if not fault.is_noop(s)]

    async def eval_at(strength: float) -> Evaluation:
        log.info("[%s] evaluating strength %.3f", fault.name, strength)
        ev = await runner.evaluate(episode, fault, strength, run_id=run_id, k=k,
                                   repeats=repeats)
        result.evaluations.append(ev)
        if ev.rejected:
            result.rejected_strengths.append(strength)
        return ev

    if mode is SearchMode.SWEEP:
        for strength in rungs:
            await eval_at(strength)
        breaks = [e.strength for e in result.evaluations if e.broke]
        passes = [e.strength for e in result.evaluations
                  if not e.rejected and e.verdict is not None and not e.verdict.broke]
        result.margin = min(breaks) if breaks else None
        result.last_passing = max([s for s in passes if result.margin is None
                                   or s < result.margin], default=None)
    else:
        lo, hi = 0, len(rungs) - 1
        margin: float | None = None
        last_pass: float | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            ev = await eval_at(rungs[mid])
            if ev.rejected:
                result.notes.append(
                    f"strength {rungs[mid]:g} rejected by the validity gate; "
                    f"treating as uninformative and searching upward"
                )
                lo = mid + 1
                continue
            if ev.broke:
                margin = rungs[mid]
                hi = mid - 1
            else:
                last_pass = rungs[mid]
                lo = mid + 1
        result.margin = margin
        result.last_passing = last_pass
        result.notes.append(
            "bisection assumes the divergence response is monotonic in ladder index; "
            "run --mode sweep to test that assumption"
        )

    curve = result.curve()
    tested = [c for c in curve if c.get("p_value") is not None]
    if mode is SearchMode.SWEEP and len(tested) > 1:
        # One permutation test per rung means the familywise error rate is not the
        # per-test alpha. Bonferroni is conservative and that is the right direction
        # to be wrong in when the headline is "we found a vulnerability".
        alpha = 0.05
        corrected = alpha / len(tested)
        survivors = [c["strength"] for c in tested if c["p_value"] <= corrected]
        result.notes.append(
            f"{len(tested)} rungs were each tested at alpha={alpha}; Bonferroni-corrected "
            f"threshold is {corrected:.4f}. Rungs significant after correction: "
            f"{survivors if survivors else 'none'}"
        )
        result.bonferroni_alpha = corrected
        result.significant_after_correction = survivors
    result.monotonic = _check_monotonic(curve)
    if result.monotonic is False:
        result.notes.append(
            "divergence is NOT monotonic in ladder strength — the margin is the "
            "lowest observed break, not a proven boundary"
        )
    if result.margin is None and not result.rejected_strengths:
        result.notes.append(
            "no rung broke the policy; the reported outcome is ROBUST over this ladder, "
            "not evidence that no perturbation could break it"
        )
    return result
