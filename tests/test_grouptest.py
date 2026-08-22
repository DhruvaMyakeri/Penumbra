"""Tests for the permutation test.

This module decides what counts as a discovered vulnerability, so its failure modes
matter more than its success modes. The tests below are mostly about it *refusing* to
find things: a null effect must not come back significant, and a design too small to
support a claim must say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.evaluation.grouptest import compare_groups, mean_trace  # noqa: E402
from penumbra.policy.base import PolicyTrace  # noqa: E402


def trace(seed: int, *, shift: float = 0.0, scale: float = 0.02, steps: int = 12,
          gripper: np.ndarray | None = None, run_id: str = "r") -> PolicyTrace:
    """A stochastic policy rollout: same mean, independent noise, optional shift."""
    g = np.random.default_rng(seed)
    actions = (g.normal(scale=scale, size=(steps, 32, 8)) + shift).astype(np.float32)
    if gripper is not None:
        actions[:, :, 7] = gripper[:, None]
    return PolicyTrace(
        policy="test",
        actions=actions,
        step_indices=np.arange(steps, dtype=np.int32) * 8,
        latencies=np.full(steps, 0.1, dtype=np.float32),
        run_id=f"{run_id}{seed}",
    )


def test_null_effect_is_not_significant():
    """The single most important guard in the project.

    Two groups drawn from the same distribution must not produce a discovery. If this
    ever fails, every reported vulnerability is suspect.
    """
    baseline = [trace(i, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, run_id="p") for i in range(4)]
    result = compare_groups(baseline, perturbed, permutations=2000, seed=1)
    assert not result.significant
    assert result.p_value > 0.05


def test_null_effect_stays_insignificant_across_many_seeds():
    """False-positive rate should be near alpha, not above it."""
    positives = 0
    trials = 20
    for t in range(trials):
        baseline = [trace(1000 * t + i, run_id="b") for i in range(4)]
        perturbed = [trace(1000 * t + 50 + i, run_id="p") for i in range(4)]
        if compare_groups(baseline, perturbed, permutations=500, seed=t).significant:
            positives += 1
    assert positives <= 3, f"{positives}/{trials} false positives — test is too permissive"


def test_a_real_shift_is_detected():
    baseline = [trace(i, run_id="b") for i in range(5)]
    perturbed = [trace(100 + i, shift=0.25, run_id="p") for i in range(5)]
    result = compare_groups(baseline, perturbed, permutations=2000, seed=1)
    assert result.significant
    assert result.between_mean > result.within_mean
    assert result.effect_size > 0.8


def test_p_value_can_never_be_zero():
    """A permutation test cannot support p = 0; the +1 correction enforces that."""
    baseline = [trace(i, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, shift=50.0, run_id="p") for i in range(4)]
    result = compare_groups(baseline, perturbed, permutations=500, seed=1)
    assert result.p_value > 0


def test_a_maximally_separated_result_lands_near_the_design_floor():
    """Even an enormous effect cannot beat what the group sizes allow.

    Sampling 500 permutations from only C(8,4)=70 distinct labellings re-draws the
    extreme ones repeatedly, so p settles near `min_achievable_p` rather than near
    1/(permutations+1). Asserting the latter would be asserting a precision the
    design does not have.
    """
    baseline = [trace(i, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, shift=50.0, run_id="p") for i in range(4)]
    result = compare_groups(baseline, perturbed, permutations=500, seed=1)
    assert result.min_achievable_p == pytest.approx(2 / 70)
    assert result.p_value == pytest.approx(result.min_achievable_p, rel=0.6)


def test_balanced_groups_have_twice_the_p_floor_of_the_naive_count():
    """When N == M a labelling and its complement give the same statistic.

    So the arrangements come in pairs and the floor is 2/C(n+m,n). Getting this wrong
    understates the smallest supportable p-value by a factor of two, which is exactly
    the kind of quiet overclaim the project exists to avoid.
    """
    from math import comb

    balanced = compare_groups([trace(i, run_id="b") for i in range(4)],
                              [trace(50 + i, run_id="p") for i in range(4)],
                              permutations=200, seed=1)
    assert balanced.min_achievable_p == pytest.approx(2 / comb(8, 4))

    unbalanced = compare_groups([trace(i, run_id="b") for i in range(5)],
                                [trace(50 + i, run_id="p") for i in range(3)],
                                permutations=200, seed=1)
    assert unbalanced.min_achievable_p == pytest.approx(1 / comb(8, 5))


def test_min_achievable_p_is_reported_when_the_design_is_too_small():
    """With N=M=3 there are 20 labellings and a floor of 0.1: p can never beat 0.05."""
    baseline = [trace(i, run_id="b") for i in range(3)]
    perturbed = [trace(100 + i, shift=50.0, run_id="p") for i in range(3)]
    result = compare_groups(baseline, perturbed, permutations=5000, seed=1)
    assert result.min_achievable_p == pytest.approx(2 / 20)
    assert result.min_achievable_p > 0.05, "N=M=3 cannot support a discovery at alpha=0.05"
    assert not result.significant
    assert any("smallest attainable p-value" in n for n in result.notes)


def test_larger_groups_can_reach_smaller_p():
    baseline = [trace(i, run_id="b") for i in range(6)]
    perturbed = [trace(100 + i, run_id="p") for i in range(6)]
    result = compare_groups(baseline, perturbed, permutations=1000, seed=1)
    assert result.min_achievable_p < 0.005  # 2 / C(12,6) = 2/924


def test_groups_of_one_are_refused():
    """A single rollout per condition cannot see through a stochastic policy."""
    with pytest.raises(ValueError, match="at least 2 runs"):
        compare_groups([trace(0)], [trace(1)])


def test_small_effect_is_flagged_even_when_significant():
    baseline = [trace(i, scale=0.02, run_id="b") for i in range(6)]
    perturbed = [trace(100 + i, scale=0.02, shift=0.012, run_id="p") for i in range(6)]
    result = compare_groups(baseline, perturbed, permutations=2000, seed=3)
    if result.significant and result.effect_size < 0.8:
        assert any("effect size is small" in n for n in result.notes)


def test_gripper_flip_rates_are_carried_through():
    same = np.zeros(12)
    flipped = np.array([0.0] * 6 + [1.0] * 6)
    baseline = [trace(i, gripper=same, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, gripper=flipped, run_id="p") for i in range(4)]
    result = compare_groups(baseline, perturbed, permutations=500, seed=1)
    assert result.gripper_flip_rate_baseline == 0.0
    assert result.gripper_flip_rate_between > 0.4


def test_result_is_deterministic_for_a_fixed_seed():
    baseline = [trace(i, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, shift=0.1, run_id="p") for i in range(4)]
    a = compare_groups(baseline, perturbed, permutations=800, seed=7)
    b = compare_groups(baseline, perturbed, permutations=800, seed=7)
    assert a.p_value == b.p_value and a.statistic == b.statistic


def test_mean_trace_averages_without_pretending_to_be_evidence():
    traces = [trace(i, shift=float(i), run_id="b") for i in range(4)]
    m = mean_trace(traces)
    assert m.actions.shape == traces[0].actions.shape
    assert m.meta["averaged_over"] == [t.run_id for t in traces]
    assert float(m.actions.mean()) == pytest.approx(1.5, abs=0.05)


# ---------------------------------------------------------------- gripper test


def test_a_bare_flip_rate_ratio_is_not_enough_to_declare_a_break():
    """Regression for a live false positive: BREAK was declared at p = 0.72.

    The old rule compared raw gripper flip rates with a 1.5x ratio and no significance
    test. Random small-sample noise clears that bar routinely. Both statistics must now
    be tested.
    """
    rng = np.random.default_rng(11)
    # Gripper patterns drawn from the SAME distribution for both groups.
    baseline = [trace(i, gripper=(rng.random(12) > 0.5).astype(float), run_id="b")
                for i in range(6)]
    perturbed = [trace(100 + i, gripper=(rng.random(12) > 0.5).astype(float), run_id="p")
                 for i in range(6)]
    result = compare_groups(baseline, perturbed, permutations=2000, seed=5)
    assert not result.any_significant, (
        f"null data declared significant: joint p={result.p_value}, "
        f"gripper p={result.gripper_p_value}"
    )


def test_a_real_decision_shift_is_detected_by_the_gripper_test():
    """A changed decision must be findable even when joint space barely moves."""
    closed_late = np.array([0.0] * 8 + [1.0] * 4)
    closed_early = np.array([0.0] * 3 + [1.0] * 9)
    baseline = [trace(i, gripper=closed_late, scale=0.001, run_id="b") for i in range(6)]
    perturbed = [trace(100 + i, gripper=closed_early, scale=0.001, run_id="p") for i in range(6)]
    result = compare_groups(baseline, perturbed, permutations=2000, seed=5)
    assert result.significant_gripper
    assert result.any_significant
    assert result.gripper_flip_rate_between > result.gripper_flip_rate_baseline


def test_two_statistics_are_each_held_to_half_alpha():
    baseline = [trace(i, run_id="b") for i in range(5)]
    perturbed = [trace(100 + i, run_id="p") for i in range(5)]
    result = compare_groups(baseline, perturbed, permutations=500, seed=1, alpha=0.05)
    assert result.alpha_per_test == pytest.approx(0.025)
    assert result.significant == (result.p_value <= 0.025)
    assert result.significant_gripper == (result.gripper_p_value <= 0.025)


def test_gripper_p_value_can_never_be_zero():
    same = np.zeros(12)
    flipped = np.ones(12)
    baseline = [trace(i, gripper=same, run_id="b") for i in range(4)]
    perturbed = [trace(100 + i, gripper=flipped, run_id="p") for i in range(4)]
    result = compare_groups(baseline, perturbed, permutations=500, seed=1)
    assert result.gripper_p_value > 0


# ---------------------------------------------------------------- suite correction


def test_a_suite_of_nulls_yields_no_discoveries():
    """The guard that stops a garage inventing vulnerabilities.

    Twenty null situations tested against one control will throw up apparent hits at
    alpha = 0.025 by chance. Correction is what keeps a shortlist honest.
    """
    from penumbra.evaluation.multiple import correct

    rng = np.random.default_rng(4)
    nulls = {f"null{i}": float(rng.uniform(0.03, 1.0)) for i in range(20)}
    assert correct(nulls).discoveries == []


def test_real_effects_survive_correction():
    from penumbra.evaluation.multiple import correct

    rng = np.random.default_rng(5)
    tests = {f"null{i}": float(rng.uniform(0.06, 1.0)) for i in range(18)}
    tests["real_a"] = 0.0002
    tests["real_b"] = 0.0007
    c = correct(tests)
    assert set(c.discoveries) >= {"real_a", "real_b"}


def test_bonferroni_is_never_more_permissive_than_fdr():
    from penumbra.evaluation.multiple import correct

    rng = np.random.default_rng(6)
    tests = {f"t{i}": float(rng.uniform(0.0001, 0.4)) for i in range(15)}
    c = correct(tests)
    assert set(c.strong_discoveries) <= set(c.discoveries)


def test_correction_handles_an_empty_suite():
    from penumbra.evaluation.multiple import correct

    c = correct({})
    assert c.n_tests == 0 and c.discoveries == []


def test_single_test_suite_needs_no_correction():
    from penumbra.evaluation.multiple import correct

    c = correct({"only": 0.02}, alpha=0.025)
    assert c.discoveries == ["only"] and c.bonferroni_alpha == 0.025


def test_identical_p_values_at_the_floor_are_flagged_not_conflated():
    """Two very differently-sized effects can report the same p-value. Say so.

    The permutation draw is seeded and the floor is a property of the design, so every
    maximally-separated condition in a suite lands on the *same* number. Five identical
    p-values then look like five identical effects, which is the opposite of the truth.
    """
    base_small = [trace(s, scale=0.002) for s in range(6)]
    pert_small = [trace(s, shift=0.2, scale=0.002) for s in range(100, 106)]
    base_huge = [trace(s, scale=0.002) for s in range(200, 206)]
    pert_huge = [trace(s, shift=9.0, scale=0.002) for s in range(300, 306)]

    small = compare_groups(base_small, pert_small)
    huge = compare_groups(base_huge, pert_huge)

    assert small.p_value == huge.p_value, "the floor should be identical across effects"
    assert small.at_resolution_floor and huge.at_resolution_floor
    assert huge.effect_size > small.effect_size * 5, "effects must differ where p does not"
    assert any("resolution floor" in n for n in small.notes)
    assert small.to_dict()["at_resolution_floor"] is True
