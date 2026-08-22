"""Group comparison with a permutation test.

The first version of PENUMBRA compared one baseline rollout against one perturbed
rollout. That was wrong, and the data said so: the measured noise floor is a mean
pairwise joint divergence of ~0.09 rad with a standard deviation of ~0.01. The policy
is **stochastic** — repeated rollouts on identical unperturbed input differ by more
than most perturbations move them. A single pairwise comparison cannot see through
that.

So the unit of evidence is a **group of runs**, not a run.

    baseline group   B = {b1 .. bN}   rollouts on the untouched episode
    perturbed group  P = {p1 .. pM}   rollouts on the perturbed episode

    statistic  T = mean between-group divergence - mean within-group divergence

If the perturbation does nothing, group labels are exchangeable and T is centred on
zero. Shuffling the labels many times gives the null distribution of T directly, with
no assumption of normality and no reliance on an estimate of sigma from six pairs.

    p = P(T_shuffled >= T_observed)

Reported alongside a standardised effect size (Cohen's d over the pairwise divergence
populations), because with N=M=4 a small p-value is possible from a tiny effect and a
large effect can miss significance. Both numbers, always.

**Two statistics, two tests.** Joint divergence says the arm moved differently; the
gripper flip rate says a *decision* changed, which is the more interesting finding and
does not have to show up in joint space. So the gripper flip rate gets its own
permutation test rather than a bare ratio comparison.

That correction came from a live false positive: an early version declared a BREAK at
p = 0.72 because between-group flip rate happened to exceed within-group flip rate by
more than half, with no significance attached. A ratio between two noisy small-sample
means is not evidence. Both statistics are now tested, and because two tests are run,
each is compared against alpha / 2.

### What this does and does not license
It licenses: "the perturbation shifted the policy's action distribution, and here is
how unlikely that is under the null." It does not license any statement about task
success, closed-loop outcome, or the real robot. See docs/RESEARCH.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..policy.base import PolicyTrace
from .divergence import divergence


@dataclass
class GroupComparison:
    """Whether two groups of rollouts came from the same behaviour distribution."""

    n_baseline: int
    n_perturbed: int
    within_mean: float
    between_mean: float
    statistic: float
    p_value: float
    permutations: int
    effect_size: float
    gripper_flip_rate_baseline: float
    gripper_flip_rate_between: float
    gripper_p_value: float
    gripper_statistic: float
    significant: bool
    significant_gripper: bool
    alpha: float
    alpha_per_test: float
    min_achievable_p: float
    notes: list[str] = field(default_factory=list)

    @property
    def at_resolution_floor(self) -> bool:
        """Is this p-value the smallest the design can express, rather than a measurement?

        At the floor, no relabelling of the groups was more extreme than the observed
        one. Two conditions can both sit here and differ by a factor of four in effect
        size, so the p-value stops discriminating and `effect_size` has to.
        """
        return bool(self.p_value <= self.min_achievable_p * 2.0)

    @property
    def any_significant(self) -> bool:
        """Either statistic clearing its corrected threshold counts as a finding."""
        return self.significant or self.significant_gripper

    def to_dict(self) -> dict:
        return {
            "n_baseline": self.n_baseline,
            "n_perturbed": self.n_perturbed,
            "within_group_divergence": round(self.within_mean, 5),
            "between_group_divergence": round(self.between_mean, 5),
            "statistic": round(self.statistic, 5),
            "p_value": round(self.p_value, 5),
            "min_achievable_p": round(self.min_achievable_p, 5),
            "at_resolution_floor": self.at_resolution_floor,
            "permutations": self.permutations,
            "effect_size_cohens_d": round(self.effect_size, 3),
            "significant_at_alpha": self.significant,
            "alpha": self.alpha,
            "alpha_per_test": round(self.alpha_per_test, 5),
            "gripper_flip_rate_within_baseline": round(self.gripper_flip_rate_baseline, 4),
            "gripper_flip_rate_between_groups": round(self.gripper_flip_rate_between, 4),
            "gripper_p_value": round(self.gripper_p_value, 5),
            "gripper_statistic": round(self.gripper_statistic, 5),
            "significant_gripper": self.significant_gripper,
            "any_significant": self.any_significant,
            "notes": self.notes,
        }


def _divergence_matrices(traces: list[PolicyTrace]) -> tuple[np.ndarray, np.ndarray]:
    """All pairwise divergences, computed once.

    The permutation loop only reshuffles *labels*; the divergence between any two
    given rollouts never changes. Computing the full symmetric matrix up front turns
    each of the thousands of permutations into array indexing instead of thousands of
    trace comparisons.
    """
    n = len(traces)
    joint = np.zeros((n, n), dtype=np.float64)
    flips = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        d = divergence(traces[i], traces[j])
        joint[i, j] = joint[j, i] = d.joint_l2
        flips[i, j] = flips[j, i] = d.gripper_flip_rate
    return joint, flips


def _select(matrix: np.ndarray, idx_a: list[int], idx_b: list[int] | None = None) -> np.ndarray:
    """Divergences within one index set, or across two, read out of a matrix."""
    if idx_b is None:
        if len(idx_a) < 2:
            return np.empty(0)
        pairs = np.array(list(combinations(idx_a, 2)))
        return matrix[pairs[:, 0], pairs[:, 1]]
    if not idx_a or not idx_b:
        return np.empty(0)
    return matrix[np.ix_(idx_a, idx_b)].ravel()


def _statistic(joint: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    a = np.flatnonzero(labels == 0).tolist()
    b = np.flatnonzero(labels == 1).tolist()
    within_parts = [_select(joint, a), _select(joint, b)]
    within_all = np.concatenate([w for w in within_parts if w.size]) if any(
        w.size for w in within_parts
    ) else np.array([0.0])
    between = _select(joint, a, b)
    between_mean = float(between.mean()) if between.size else 0.0
    within_mean = float(within_all.mean())
    return between_mean - within_mean, within_mean, between_mean


def compare_groups(
    baseline: list[PolicyTrace],
    perturbed: list[PolicyTrace],
    *,
    permutations: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> GroupComparison:
    """Permutation test on whether a perturbation moved the policy's behaviour."""
    if len(baseline) < 2 or len(perturbed) < 2:
        raise ValueError(
            "a group comparison needs at least 2 runs per group; "
            f"got {len(baseline)} baseline and {len(perturbed)} perturbed"
        )
    traces = baseline + perturbed
    n, m = len(baseline), len(perturbed)
    labels = np.array([0] * n + [1] * m)

    joint, flips = _divergence_matrices(traces)
    observed, within, between = _statistic(joint, labels)

    base_idx, pert_idx = list(range(n)), list(range(n, n + m))
    flip_within = _select(flips, base_idx)
    flip_between = _select(flips, base_idx, pert_idx)

    observed_grip, _, _ = _statistic(flips, labels)

    rng = np.random.default_rng(seed)
    count = 0
    count_grip = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        if _statistic(joint, shuffled)[0] >= observed:
            count += 1
        if _statistic(flips, shuffled)[0] >= observed_grip:
            count_grip += 1
    # +1 in both places: the observed labelling is itself one of the arrangements, and
    # omitting it lets a test report p = 0, which no permutation test can support.
    p_value = (count + 1) / (permutations + 1)
    gripper_p = (count_grip + 1) / (permutations + 1)

    within_all = _select(joint, base_idx)
    between_all = _select(joint, base_idx, pert_idx)
    pooled = np.sqrt(
        (within_all.var(ddof=1) * (len(within_all) - 1)
         + between_all.var(ddof=1) * (len(between_all) - 1))
        / max(1, len(within_all) + len(between_all) - 2)
    ) if len(within_all) > 1 and len(between_all) > 1 else 0.0
    effect = float((between_all.mean() - within_all.mean()) / pooled) if pooled > 1e-9 else 0.0

    # With small groups the smallest attainable p-value is bounded by the number of
    # distinct label assignments, not by how many permutations we draw. Reporting it
    # stops "p = 0.03, n = 3" being read as stronger than the design can support.
    #
    # The factor of two in the balanced case is easy to get wrong and matters. The
    # statistic depends only on the *partition* into two groups, not on which side is
    # labelled "baseline": when n == m, a labelling and its complement give an
    # identical T, so the arrangements come in pairs and no observed result can be
    # rarer than 2 / C(n+m, n).
    from math import comb

    arrangements = comb(n + m, n)
    min_p = (2.0 if n == m else 1.0) / arrangements

    # Two statistics are tested, so each is held to alpha / 2. Bonferroni is
    # conservative, which is the right direction to err when the headline would be
    # "we found a vulnerability".
    alpha_per_test = alpha / 2.0
    significant_joint = bool(p_value <= alpha_per_test)
    significant_grip = bool(gripper_p <= alpha_per_test)

    notes: list[str] = []
    if significant_grip and not significant_joint:
        notes.append(
            "the gripper decision distribution shifted while joint divergence did not; "
            "a changed decision is the stronger finding of the two"
        )
    if arrangements < 40:
        notes.append(
            f"only {arrangements} distinct group labellings exist for N={n}, M={m}; "
            f"the smallest attainable p-value is {min_p:.3f}"
        )
    # A run at the floor is "as extreme as this design can report", not a measurement
    # of how extreme. Several conditions in one suite will land on the *same* p-value
    # when each is maximally separated, because the permutation draw is seeded and the
    # floor is a property of the design rather than of the data. Reading five identical
    # p-values as five identical effects is then exactly the wrong conclusion, and the
    # effect size is what distinguishes them.
    at_floor = bool(p_value <= min_p * 2.0)
    if at_floor:
        notes.append(
            f"p = {p_value:.4f} is at the resolution floor for N={n}, M={m} "
            f"(smallest attainable {min_p:.5f}): no relabelling was more extreme than "
            f"the observed one. Read this as maximal separation at this sample size, "
            f"not as a magnitude - the effect size (d = {effect:+.2f}) carries that. "
            f"Two conditions can report the identical p-value here and differ greatly "
            f"in effect."
        )
    if significant_joint and effect < 0.8:
        notes.append(
            "statistically separable but the effect size is small; treat as a weak signal"
        )

    return GroupComparison(
        n_baseline=n,
        n_perturbed=m,
        within_mean=within,
        between_mean=between,
        statistic=observed,
        p_value=p_value,
        permutations=permutations,
        effect_size=effect,
        gripper_flip_rate_baseline=float(flip_within.mean()) if len(flip_within) else 0.0,
        gripper_flip_rate_between=float(flip_between.mean()) if len(flip_between) else 0.0,
        gripper_p_value=gripper_p,
        gripper_statistic=observed_grip,
        significant=significant_joint,
        significant_gripper=significant_grip,
        alpha=alpha,
        alpha_per_test=alpha_per_test,
        min_achievable_p=min_p,
        notes=notes,
    )


def mean_trace(traces: list[PolicyTrace], run_id: str = "mean") -> PolicyTrace:
    """The average action trace over a group — useful for plots, not for the test.

    Averaging a stochastic policy's samples is a summary, not evidence: it hides the
    spread that the permutation test is built to account for.
    """
    steps = min(len(t) for t in traces)
    horizon = min(t.horizon for t in traces)
    stacked = np.stack([t.actions[:steps, :horizon] for t in traces])
    return PolicyTrace(
        policy=traces[0].policy,
        actions=stacked.mean(axis=0),
        step_indices=traces[0].step_indices[:steps],
        latencies=np.concatenate([t.latencies for t in traces]),
        run_id=run_id,
        meta={"averaged_over": [t.run_id for t in traces]},
    )
