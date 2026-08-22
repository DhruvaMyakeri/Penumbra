"""Behaviour divergence, and the noise floor it must be measured against.

The policy emits a `[32][8]` action chunk per control step: 7 joint positions plus a
gripper command, in DROID joint-position convention. So the metrics here are chosen
to fit that interface rather than forced onto it:

- **joint_l2** — RMS difference over the 7 joint channels, in radians. Physically
  interpretable: it is how far apart the two commanded arm configurations are.
- **gripper_l1** — mean absolute difference on the gripper channel, which is
  normalised roughly to [0,1] and behaves more like a decision than a coordinate.
- **gripper_flips** — how many control steps disagree about whether the gripper is
  commanded closed. This is the discrete behavioural signal: a flip is a different
  *decision*, not a different number.
- **timing_shift** — cross-correlation lag between the two gripper traces. A policy
  that does the right thing 400 ms late has not made a small error.

### The noise floor is the whole ballgame

The policy is not deterministic across sessions, and the perturbation model has no
seed. So "the actions differed" means nothing on its own. `NoiseFloor` re-runs the
**unperturbed** episode N times and measures the same divergence between those runs.
Everything downstream is expressed in units of that spread.

A break is declared only when divergence exceeds `mean + k·std` of the noise floor.
With small N the standard deviation is itself uncertain, so `k` is deliberately
conservative and the sample size is carried into every report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..policy.base import PolicyTrace

#: DROID action layout: indices 0-6 are joint positions, 7 is the gripper.
JOINT_SLICE = slice(0, 7)
GRIPPER_INDEX = 7
#: Above this commanded value the gripper is treated as "closing".
GRIPPER_CLOSED_THRESHOLD = 0.5


@dataclass
class Divergence:
    """How differently two runs behaved."""

    joint_l2: float
    gripper_l1: float
    gripper_flips: int
    gripper_flip_rate: float
    timing_shift_steps: float
    first_step_joint_l2: float
    steps_compared: int
    per_step_joint_l2: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "joint_l2": round(self.joint_l2, 5),
            "gripper_l1": round(self.gripper_l1, 5),
            "gripper_flips": self.gripper_flips,
            "gripper_flip_rate": round(self.gripper_flip_rate, 4),
            "timing_shift_steps": round(self.timing_shift_steps, 3),
            "first_step_joint_l2": round(self.first_step_joint_l2, 5),
            "steps_compared": self.steps_compared,
        }


def _common(a: PolicyTrace, b: PolicyTrace) -> tuple[np.ndarray, np.ndarray]:
    """Trim two traces to their shared steps and horizon."""
    steps = min(len(a), len(b))
    horizon = min(a.horizon, b.horizon)
    if steps == 0:
        raise ValueError("cannot compare traces with no steps")
    if not np.array_equal(a.step_indices[:steps], b.step_indices[:steps]):
        raise ValueError(
            "traces are not aligned on the same frames; refusing to compare "
            f"{a.step_indices[:steps]} vs {b.step_indices[:steps]}"
        )
    return a.actions[:steps, :horizon], b.actions[:steps, :horizon]


def _timing_shift(a: np.ndarray, b: np.ndarray, max_lag: int = 6) -> float:
    """Lag in control steps that best aligns two gripper traces (positive = b is late)."""
    x = a - a.mean()
    y = b - b.mean()
    if np.allclose(x, 0) or np.allclose(y, 0):
        return 0.0
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xa, yb = x[: len(x) - lag], y[lag:]
        else:
            xa, yb = x[-lag:], y[: len(y) + lag]
        if len(xa) < 3:
            continue
        denom = np.linalg.norm(xa) * np.linalg.norm(yb)
        if denom < 1e-9:
            continue
        score = float(np.dot(xa, yb) / denom)
        if score > best:
            best, best_lag = score, lag
    return float(best_lag)


def divergence(a: PolicyTrace, b: PolicyTrace) -> Divergence:
    """Behavioural difference between two policy runs on the same episode timeline."""
    aa, bb = _common(a, b)
    joints_a, joints_b = aa[:, :, JOINT_SLICE], bb[:, :, JOINT_SLICE]
    per_step = np.sqrt(((joints_a - joints_b) ** 2).mean(axis=(1, 2)))

    grip_a, grip_b = aa[:, :, GRIPPER_INDEX], bb[:, :, GRIPPER_INDEX]
    closed_a = grip_a[:, 0] > GRIPPER_CLOSED_THRESHOLD
    closed_b = grip_b[:, 0] > GRIPPER_CLOSED_THRESHOLD
    flips = int(np.sum(closed_a != closed_b))

    first_a, first_b = aa[:, 0, JOINT_SLICE], bb[:, 0, JOINT_SLICE]
    return Divergence(
        joint_l2=float(per_step.mean()),
        gripper_l1=float(np.abs(grip_a - grip_b).mean()),
        gripper_flips=flips,
        gripper_flip_rate=float(flips / len(closed_a)),
        timing_shift_steps=_timing_shift(grip_a[:, 0], grip_b[:, 0]),
        first_step_joint_l2=float(np.sqrt(((first_a - first_b) ** 2).mean())),
        steps_compared=int(len(aa)),
        per_step_joint_l2=[round(float(v), 5) for v in per_step],
    )


@dataclass
class NoiseFloor:
    """The policy's own run-to-run variability on the *unperturbed* episode.

    `n_runs` repeat rollouts give `n_runs*(n_runs-1)/2` pairwise comparisons. Both the
    mean and the spread are reported, and both are carried into every failure verdict
    so a reader can see how thin the evidence is.
    """

    n_runs: int
    joint_l2_mean: float
    joint_l2_std: float
    joint_l2_max: float
    gripper_l1_mean: float
    gripper_flip_rate_mean: float
    gripper_flip_rate_max: float
    pairs: int
    samples: list[dict] = field(default_factory=list)

    def threshold(self, k: float = 3.0) -> float:
        """Divergence above which a joint-space change is not plausibly noise."""
        return self.joint_l2_mean + k * self.joint_l2_std

    def sigma_units(self, value: float) -> float:
        """`value` expressed in standard deviations above the noise-floor mean."""
        if self.joint_l2_std < 1e-9:
            return float("inf") if value > self.joint_l2_mean else 0.0
        return (value - self.joint_l2_mean) / self.joint_l2_std

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "pairs": self.pairs,
            "joint_l2_mean": round(self.joint_l2_mean, 5),
            "joint_l2_std": round(self.joint_l2_std, 5),
            "joint_l2_max": round(self.joint_l2_max, 5),
            "gripper_l1_mean": round(self.gripper_l1_mean, 5),
            "gripper_flip_rate_mean": round(self.gripper_flip_rate_mean, 4),
            "gripper_flip_rate_max": round(self.gripper_flip_rate_max, 4),
            "threshold_k3": round(self.threshold(3.0), 5),
        }


def compute_noise_floor(traces: list[PolicyTrace]) -> NoiseFloor:
    if len(traces) < 2:
        raise ValueError("a noise floor needs at least 2 repeat runs")
    joint, grip, flip = [], [], []
    samples = []
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            d = divergence(traces[i], traces[j])
            joint.append(d.joint_l2)
            grip.append(d.gripper_l1)
            flip.append(d.gripper_flip_rate)
            samples.append({"pair": [traces[i].run_id, traces[j].run_id], **d.to_dict()})
    return NoiseFloor(
        n_runs=len(traces),
        joint_l2_mean=float(np.mean(joint)),
        joint_l2_std=float(np.std(joint, ddof=1)) if len(joint) > 1 else 0.0,
        joint_l2_max=float(np.max(joint)),
        gripper_l1_mean=float(np.mean(grip)),
        gripper_flip_rate_mean=float(np.mean(flip)),
        gripper_flip_rate_max=float(np.max(flip)),
        pairs=len(joint),
        samples=samples,
    )


@dataclass
class Verdict:
    """Whether an observed divergence counts as a break, and why."""

    broke: bool
    reason: str
    metric: str
    observed: float
    threshold: float
    sigma: float
    k: float
    gripper_flip_rate: float
    noise_flip_rate_max: float

    def to_dict(self) -> dict:
        return {
            "broke": self.broke,
            "reason": self.reason,
            "metric": self.metric,
            "observed": round(self.observed, 5),
            "threshold": round(self.threshold, 5),
            "sigma_above_noise": round(self.sigma, 2),
            "k": self.k,
            "gripper_flip_rate": round(self.gripper_flip_rate, 4),
            "noise_flip_rate_max": round(self.noise_flip_rate_max, 4),
        }


def judge(d: Divergence, floor: NoiseFloor, *, k: float = 3.0) -> Verdict:
    """Declare a break only when the evidence clears the policy's own noise.

    Two independent routes to a break, because the two signals mean different things:
    a joint-space excursion is a different *trajectory*; a gripper flip is a different
    *decision*, and a decision the unperturbed runs never disagreed about is the
    stronger finding of the two.
    """
    threshold = floor.threshold(k)
    sigma = floor.sigma_units(d.joint_l2)
    joint_break = d.joint_l2 > threshold
    grip_break = d.gripper_flip_rate > max(floor.gripper_flip_rate_max, 1e-9)

    if joint_break and grip_break:
        reason = (
            f"joint divergence {d.joint_l2:.4f} rad exceeds noise floor +{k}sigma "
            f"({threshold:.4f}), and gripper decisions flipped on "
            f"{d.gripper_flip_rate:.0%} of steps versus a noise-floor maximum of "
            f"{floor.gripper_flip_rate_max:.0%}"
        )
    elif joint_break:
        reason = (
            f"joint divergence {d.joint_l2:.4f} rad exceeds noise floor +{k}sigma "
            f"({threshold:.4f}); gripper decisions unchanged"
        )
    elif grip_break:
        reason = (
            f"gripper decisions flipped on {d.gripper_flip_rate:.0%} of steps versus a "
            f"noise-floor maximum of {floor.gripper_flip_rate_max:.0%}, while joint "
            f"divergence {d.joint_l2:.4f} stayed within noise"
        )
    else:
        reason = (
            f"joint divergence {d.joint_l2:.4f} is within the noise floor "
            f"({threshold:.4f} at k={k}) and gripper decisions did not flip beyond "
            f"noise"
        )
    return Verdict(
        broke=bool(joint_break or grip_break),
        reason=reason,
        metric="joint_l2+gripper_flip_rate",
        observed=d.joint_l2,
        threshold=threshold,
        sigma=sigma,
        k=k,
        gripper_flip_rate=d.gripper_flip_rate,
        noise_flip_rate_max=floor.gripper_flip_rate_max,
    )
