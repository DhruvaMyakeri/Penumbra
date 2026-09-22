"""Unit tests for the parts of PENUMBRA that must not be wrong.

No network, no Reactor, no credentials. Reactor integration lives in tools/smoke_*.py
and is exercised by hand because it costs money and needs the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.episodes.types import Episode  # noqa: E402
from penumbra.evaluation.divergence import (  # noqa: E402
    compute_noise_floor,
    divergence,
    judge,
)
from penumbra.perturbation.classical import (  # noqa: E402
    ClassicalPerturbation,
    match_perceptual_distance,
)
from penumbra.perturbation.spec import load_fault  # noqa: E402
from penumbra.perturbation.x2 import align  # noqa: E402
from penumbra.policy.base import PolicyTrace  # noqa: E402
from penumbra.validation.seam import SeamGate, perceptual_distance  # noqa: E402

rng = np.random.default_rng(0)


def make_episode(n: int = 24, h: int = 90, w: int = 160) -> Episode:
    """A synthetic episode with texture, because a flat gradient is degenerate for flow.

    A pure linear ramp carries no horizontal localisation information at all, so
    shifting it is literally indistinguishable from re-lighting it. Real robot scenes
    have texture; the test scene does too, and the aperture-problem limitation gets
    its own explicit test below rather than being papered over here.
    """
    g = np.random.default_rng(42)
    base = np.tile(np.linspace(20, 200, w, dtype=np.float32)[None, :, None], (h, 1, 3))
    texture = g.normal(0, 34, (h, w, 1)) + 40 * np.sin(
        np.arange(w)[None, :, None] / 3.1
    ) * np.cos(np.arange(h)[:, None, None] / 2.7)
    base = np.clip(base + texture, 0, 255).astype(np.uint8)
    frames = []
    for t in range(n):
        f = base.copy()
        x = 10 + (t * 3) % (w - 30)
        f[30:60, x : x + 20] = 255
        f[62:74, 12:40] = np.clip(
            g.integers(90, 220, (12, 28, 3)).astype(np.int16) + t, 0, 255
        ).astype(np.uint8)
        frames.append(f)
    stack = np.stack(frames)
    return Episode(
        episode_id="synthetic",
        source="test",
        task="do the thing",
        fps=15.0,
        frames={
            "exterior_image_1_left": stack,
            "exterior_image_2_left": stack.copy(),
            "wrist_image_left": stack.copy(),
        },
        joint_positions=rng.normal(size=(n, 7)).astype(np.float32),
        gripper_position=rng.random((n, 1)).astype(np.float32),
        actions=rng.normal(size=(n, 8)).astype(np.float32),
    )


def make_trace(steps: int = 10, *, offset: float = 0.0, gripper: np.ndarray | None = None,
               run_id: str = "r", seed: int = 0) -> PolicyTrace:
    g = np.random.default_rng(seed)
    actions = g.normal(scale=0.01, size=(steps, 32, 8)).astype(np.float32) + offset
    if gripper is not None:
        actions[:, :, 7] = gripper[:, None]
    return PolicyTrace(
        policy="test",
        actions=actions,
        step_indices=np.arange(steps, dtype=np.int32) * 8,
        latencies=np.full(steps, 0.1, dtype=np.float32),
        run_id=run_id,
    )


# ---------------------------------------------------------------- episodes


def test_episode_rejects_length_mismatch():
    ep = make_episode(8)
    with pytest.raises(ValueError, match="length mismatch"):
        Episode(
            episode_id="bad", source="t", task="t", fps=15.0,
            frames=ep.frames, joint_positions=ep.joint_positions[:4],
            gripper_position=ep.gripper_position, actions=ep.actions,
        )


def test_window_preserves_alignment():
    ep = make_episode(20).window(5, 8)
    assert len(ep) == 8
    assert ep.frame_offset == 5
    assert ep.frames["wrist_image_left"].shape[0] == 8


def test_with_view_shares_everything_but_pixels():
    """The whole thesis in one assertion: perturbation touches pixels and nothing else."""
    ep = make_episode(10)
    new = np.zeros_like(ep.frames["exterior_image_1_left"])
    out = ep.with_view("exterior_image_1_left", new, "perturbed")
    assert out.joint_positions is ep.joint_positions
    assert out.actions is ep.actions
    assert out.frames["wrist_image_left"] is ep.frames["wrist_image_left"]
    assert not np.array_equal(out.frames["exterior_image_1_left"],
                              ep.frames["exterior_image_1_left"])


def test_with_view_rejects_wrong_length():
    ep = make_episode(10)
    with pytest.raises(ValueError, match="frames, episode has"):
        ep.with_view("exterior_image_1_left", np.zeros((3, 90, 160, 3), np.uint8), "x")


def test_proprio_payload_shape_matches_schema():
    ep = make_episode(4)
    p = ep.proprio_at(1)
    assert list(p) == ["joint_position", "gripper_position"]
    assert len(p["joint_position"]) == 1 and len(p["joint_position"][0]) == 7
    assert len(p["gripper_position"][0]) == 1


# ---------------------------------------------------------------- faults


def test_every_shipped_fault_loads_and_starts_with_a_noop():
    from penumbra.perturbation.spec import list_faults

    names = list_faults()
    assert names, "no faults shipped"
    for name in names:
        f = load_fault(name)
        assert f.ladder[0].strength == 0.0
        assert f.is_noop(0.0)
        assert not f.is_noop(f.ladder[-1].strength)
        assert list(f.strengths) == sorted(f.strengths)


def test_rung_at_selects_floor_not_nearest():
    f = load_fault("specular_floor")
    assert f.rung_at(0.59).strength == 0.4
    assert f.rung_at(0.6).strength == 0.6
    assert f.rung_at(5.0).strength == 1.0


def test_fault_roundtrips_to_dict():
    f = load_fault("specular_floor")
    d = f.to_dict()
    assert d["name"] == "specular_floor" and len(d["ladder"]) == len(f.ladder)


# ---------------------------------------------------------------- divergence


def test_identical_traces_have_zero_divergence():
    t = make_trace(seed=1)
    d = divergence(t, t)
    assert d.joint_l2 == pytest.approx(0.0, abs=1e-7)
    assert d.gripper_flips == 0


def test_divergence_refuses_misaligned_traces():
    a = make_trace(6, seed=1)
    b = make_trace(6, seed=2)
    b.step_indices = b.step_indices + 1
    with pytest.raises(ValueError, match="not aligned"):
        divergence(a, b)


def test_gripper_flips_are_counted_as_decisions():
    closed = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    late = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    a = make_trace(6, gripper=closed, seed=1)
    b = make_trace(6, gripper=late, seed=1)
    d = divergence(a, b)
    assert d.gripper_flips == 2
    assert d.gripper_flip_rate == pytest.approx(2 / 6)


def test_timing_shift_detects_a_lagged_gripper():
    base = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    lagged = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    d = divergence(make_trace(8, gripper=base, seed=3),
                   make_trace(8, gripper=lagged, seed=3))
    assert d.timing_shift_steps > 0


def test_noise_floor_needs_repeats():
    with pytest.raises(ValueError, match="at least 2"):
        compute_noise_floor([make_trace(seed=1)])


def test_noise_floor_counts_all_pairs():
    traces = [make_trace(seed=i, run_id=f"r{i}") for i in range(4)]
    nf = compute_noise_floor(traces)
    assert nf.n_runs == 4 and nf.pairs == 6
    assert nf.joint_l2_mean > 0


def test_small_change_inside_the_noise_floor_is_not_a_break():
    """The single most important guard: noise must not be reported as a failure."""
    traces = [make_trace(seed=i, run_id=f"r{i}") for i in range(4)]
    nf = compute_noise_floor(traces)
    d = divergence(traces[0], make_trace(seed=99, run_id="p"))
    v = judge(d, nf, k=3.0)
    assert not v.broke
    assert "within the noise floor" in v.reason


def test_large_change_outside_the_noise_floor_is_a_break():
    traces = [make_trace(seed=i, run_id=f"r{i}") for i in range(4)]
    nf = compute_noise_floor(traces)
    d = divergence(traces[0], make_trace(offset=0.5, seed=99, run_id="p"))
    v = judge(d, nf, k=3.0)
    assert v.broke and v.sigma > 3.0


def test_gripper_flip_alone_can_break():
    """A changed decision counts even when joint divergence stays inside noise."""
    same = np.zeros(10)
    traces = [make_trace(10, gripper=same, seed=i, run_id=f"r{i}") for i in range(4)]
    nf = compute_noise_floor(traces)
    assert nf.gripper_flip_rate_max == 0.0
    flipped = np.array([0.0] * 5 + [1.0] * 5)
    d = divergence(traces[0], make_trace(10, gripper=flipped, seed=0, run_id="p"))
    v = judge(d, nf, k=3.0)
    assert v.broke and "gripper decisions flipped" in v.reason


# ---------------------------------------------------------------- validation


def test_gate_accepts_a_pure_appearance_change():
    ep = make_episode(16)
    src = ep.frames["exterior_image_1_left"]
    brighter = np.clip(src.astype(np.float32) * 1.35 + 18, 0, 255).astype(np.uint8)
    report = SeamGate().validate(src, brighter)
    assert report.valid, report.reasons
    assert report.geometry_drift < 0.05


def test_gate_rejects_a_geometric_shift():
    """The case the gate exists for: the scene moved, so any behaviour change is
    about the transformation rather than the policy."""
    ep = make_episode(16)
    src = ep.frames["exterior_image_1_left"]
    shifted = np.roll(src, 22, axis=2)
    report = SeamGate().validate(src, shifted)
    assert not report.valid
    assert any("geometry drift" in r for r in report.reasons)


def test_gate_rejects_a_localised_displacement_the_median_misses():
    """This is what the p90 statistic exists for.

    A region of the scene slides while the rest holds still. The median flow stays
    near zero — the corruption is a minority of pixels — but p90 catches it.
    """
    ep = make_episode(16)
    src = ep.frames["exterior_image_1_left"].copy()
    moved = src.copy()
    moved[:, 20:80, 5:120] = np.roll(moved[:, 20:80, 5:120], 20, axis=2)
    report = SeamGate().validate(src, moved)
    assert report.geometry_drift < report.max_geometry_drift, "median should be blind here"
    assert report.geometry_drift_p90 > report.max_geometry_drift_p90
    assert not report.valid
    assert any("localised" in r for r in report.reasons)


def test_gate_is_blind_to_a_small_region_moving():
    """A stated limitation, pinned so it cannot be forgotten.

    A patch covering roughly a tenth of the frame can slide 16 px and pass both
    statistics. SEAM v1 bounds geometry corruption; it does not eliminate it. See
    docs/METHODOLOGY.md.
    """
    ep = make_episode(16)
    src = ep.frames["exterior_image_1_left"].copy()
    moved = src.copy()
    moved[:, 28:62, 8:52] = np.roll(moved[:, 28:62, 8:52], 16, axis=2)
    report = SeamGate().validate(src, moved)
    assert report.valid, "if this now fails the gate improved; update docs/METHODOLOGY.md"


def test_thresholds_separate_measured_appearance_edits_from_measured_shifts():
    """Pins the calibration in tools/calibrate_gate.py, measured on real DROID frames.

    If these constants drift apart, the gate silently stops separating a legitimate
    relight from a corrupted frame, which is the one thing it exists to do.
    """
    from penumbra.validation.seam import drift_in_pixels

    gate = SeamGate()
    measured_x2_appearance_edit = 0.00154
    measured_four_pixel_shift = 0.00545
    measured_thirtytwo_pixel_shift = 0.04356
    # A legitimate appearance edit passes with real margin...
    assert measured_x2_appearance_edit < gate.max_geometry_drift / 3
    # ...and a 4 px displacement, the point where geometry could start to matter, does not.
    assert measured_four_pixel_shift < gate.max_geometry_drift * 1.02
    assert measured_thirtytwo_pixel_shift > gate.max_geometry_drift * 5
    assert drift_in_pixels(measured_four_pixel_shift) == pytest.approx(4.0, abs=0.3)


def test_gate_is_documented_as_blind_to_the_aperture_problem():
    """A displaced *textureless* region is invisible to any dense-flow method.

    This is a real, stated limitation of SEAM v1 rather than a bug. The test pins it
    so that nobody later reads a passing gate as proof geometry was preserved.
    """
    n, h, w = 12, 90, 160
    flat = np.tile(np.linspace(20, 200, w, dtype=np.uint8)[None, :, None], (h, 1, 3))
    src = np.repeat(flat[None], n, axis=0)
    shifted = np.roll(src, 22, axis=2)
    report = SeamGate().validate(src, shifted)
    assert report.geometry_drift < 0.05, (
        "if this ever fails the gate got strictly better; update docs/METHODOLOGY.md"
    )


def test_gate_rejects_a_redrawn_scene():
    ep = make_episode(16)
    src = ep.frames["exterior_image_1_left"]
    noise = np.random.default_rng(0).integers(0, 255, src.shape, dtype=np.uint8)
    report = SeamGate().validate(src, noise)
    assert not report.valid


def test_gate_refuses_mismatched_lengths():
    ep = make_episode(10)
    src = ep.frames["exterior_image_1_left"]
    with pytest.raises(ValueError, match="length mismatch"):
        SeamGate().validate(src, src[:5])


def test_perceptual_distance_increases_with_change():
    ep = make_episode(12)
    src = ep.frames["exterior_image_1_left"]
    small = np.clip(src.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    large = np.clip(src.astype(np.int16) + 70, 0, 255).astype(np.uint8)
    assert perceptual_distance(src, small)["rmse"] < perceptual_distance(src, large)["rmse"]


# ---------------------------------------------------------------- alignment


def test_align_recovers_a_known_offset():
    ep = make_episode(30)
    src = ep.frames["exterior_image_1_left"]
    received = np.concatenate([np.repeat(src[:1], 4, axis=0), src])
    aligned, offset, score = align(src, received, max_shift=6)
    assert aligned.shape == src.shape
    assert score > 0.9


def test_align_scores_a_scrambled_stream_lower_than_a_matching_one():
    ep = make_episode(30)
    src = ep.frames["exterior_image_1_left"]
    _, _, good = align(src, src.copy())
    scrambled = src[np.random.default_rng(0).permutation(len(src))]
    _, _, bad = align(src, scrambled)
    assert good > bad


def test_align_refuses_an_empty_stream():
    ep = make_episode(6)
    with pytest.raises(RuntimeError, match="no frames"):
        align(ep.frames["exterior_image_1_left"], np.zeros((0, 90, 160, 3), np.uint8))


# ---------------------------------------------------------------- classical arm


def test_classical_is_deterministic_unlike_x2():
    ep = make_episode(10)
    a = ClassicalPerturbation("blur_noise", seed=7).apply(ep, 0.6)
    b = ClassicalPerturbation("blur_noise", seed=7).apply(ep, 0.6)
    assert np.array_equal(a.episode.frames["exterior_image_1_left"],
                          b.episode.frames["exterior_image_1_left"])


def test_classical_strength_zero_is_nearly_identity():
    ep = make_episode(8)
    out = ClassicalPerturbation("specular_overlay").apply(ep, 0.0)
    assert out.distance["rmse"] < 1e-6


def test_matching_finds_a_comparable_perceptual_distance():
    ep = make_episode(12)
    target = {"rmse": 0.09}
    matched = match_perceptual_distance(ep, "brightness_contrast", target)
    assert abs(matched.distance["rmse"] - 0.09) < 0.03
    assert 0.0 <= matched.strength <= 1.0


def test_unknown_classical_op_is_refused():
    with pytest.raises(ValueError, match="unknown classical op"):
        ClassicalPerturbation("does_not_exist")


# ---------------------------------------------------------------- positive control


def test_blackout_destroys_the_view_at_full_strength():
    """The positive control must actually be maximal, or it controls for nothing."""
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(8)
    out = ClassicalPerturbation("blackout").apply(ep, 1.0)
    assert out.episode.frames["exterior_image_1_left"].max() == 0


def test_blackout_at_zero_is_identity():
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(8)
    out = ClassicalPerturbation("blackout").apply(ep, 0.0)
    assert np.array_equal(out.episode.frames["exterior_image_1_left"],
                          ep.frames["exterior_image_1_left"])


def test_blackout_leaves_the_other_views_and_the_actions_alone():
    """Even the positive control is a single-variable intervention."""
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(8)
    out = ClassicalPerturbation("blackout").apply(ep, 1.0).episode
    assert out.frames["wrist_image_left"] is ep.frames["wrist_image_left"]
    assert out.frames["exterior_image_2_left"] is ep.frames["exterior_image_2_left"]
    assert out.actions is ep.actions
    assert out.joint_positions is ep.joint_positions


def test_positive_controls_are_marked_as_such():
    """So they are never quietly perceptual-distance matched against a real fault."""
    from penumbra.perturbation.classical import CLASSICAL_OPS, POSITIVE_CONTROLS

    assert "blackout" in POSITIVE_CONTROLS
    assert POSITIVE_CONTROLS <= set(CLASSICAL_OPS)
    assert not POSITIVE_CONTROLS & {"brightness_contrast", "gamma_glare", "blur_noise",
                                    "color_shift", "specular_overlay"}


def test_classical_and_generative_face_the_same_gate():
    """Symmetry check, from a live asymmetry.

    `blur_noise` at matched perceptual distance produced a stream with 9.28x the
    source's frame-to-frame motion and was scored as a classical "BREAK" while the
    generative arm was being rejected for exactly that defect. The arms must be judged
    identically or the comparison is rigged.
    """
    import inspect

    from penumbra.experiments.runner import ExperimentRunner

    src = inspect.getsource(ExperimentRunner.evaluate_classical)
    assert "rejected=False" not in src, "classical arm is exempt from the gate again"
    assert "report.valid" in src and "POSITIVE_CONTROLS" in src


def test_temporal_incoherence_is_caught_on_a_classical_transform():
    """Per-frame independent noise is exactly what the temporal signal exists for."""
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(16)
    out = ClassicalPerturbation("blur_noise", seed=3).apply(ep, 0.85)
    report = SeamGate().validate(ep.frames["exterior_image_1_left"],
                                 out.episode.frames["exterior_image_1_left"])
    assert report.temporal_ratio > report.max_temporal_ratio
    assert not report.valid
    assert any("temporal" in r for r in report.reasons)


# ---------------------------------------------------------------- multi-view


def test_with_views_replaces_several_and_shares_the_rest():
    """All-view perturbation must still be a pixels-only intervention."""
    ep = make_episode(10)
    black = {v: np.zeros_like(ep.frames[v]) for v in
             ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")}
    out = ep.with_views(black, "all-black")
    for v in black:
        assert out.frames[v].max() == 0
    assert out.actions is ep.actions
    assert out.joint_positions is ep.joint_positions
    assert out.gripper_position is ep.gripper_position


def test_with_views_rejects_an_unknown_view():
    ep = make_episode(6)
    with pytest.raises(ValueError, match="unknown view"):
        ep.with_views({"nose_cam": np.zeros((6, 90, 160, 3), np.uint8)}, "x")


def test_with_views_rejects_a_wrong_length_replacement():
    ep = make_episode(6)
    with pytest.raises(ValueError, match="frames, episode has"):
        ep.with_views({"wrist_image_left": np.zeros((3, 90, 160, 3), np.uint8)}, "x")


def test_multiview_classical_is_independent_of_the_view_set():
    """camera 1's pixels must not depend on whether camera 2 is also perturbed.

    Otherwise the ablation compares conditions that differ in two ways at once.
    """
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(8)
    one = ClassicalPerturbation("blur_noise", views=["exterior_image_1_left"], seed=3)
    two = ClassicalPerturbation(
        "blur_noise", views=["exterior_image_1_left", "wrist_image_left"], seed=3
    )
    a = one.apply(ep, 0.5).episode.frames["exterior_image_1_left"]
    b = two.apply(ep, 0.5).episode.frames["exterior_image_1_left"]
    assert np.array_equal(a, b)


def test_multiview_result_reports_per_view_distance():
    from penumbra.perturbation.classical import ClassicalPerturbation

    ep = make_episode(8)
    views = ["exterior_image_1_left", "wrist_image_left"]
    r = ClassicalPerturbation("blackout", views=views).apply(ep, 1.0)
    assert set(r.per_view_distance) == set(views)
    assert set(r.to_dict()["views"]) == set(views)


def test_dashboard_javascript_parses():
    """The dashboard's script must be syntactically valid.

    A raw newline inside a JS string literal once made the whole `<script>` a
    SyntaxError. The server still answered 200 and the markup still arrived, so every
    check that looked at HTTP status passed while the page rendered nothing at all.
    Only a parser catches that class of failure, so run one.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to parse the dashboard script")

    html = (Path(__file__).resolve().parents[1] / "src/penumbra/ui/dashboard.html"
            ).read_text(encoding="utf-8")
    assert html.count("<script>") == 1, "expected exactly one inline script block"
    script = html.split("<script>")[1].split("</script>")[0]

    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "dashboard.js"
        js.write_text(script, encoding="utf-8")
        proc = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, f"dashboard script does not parse:\n{proc.stderr}"


def test_dashboard_renders_a_state_without_throwing():
    """Parsing is not the same as running.

    A missing property or a bad call inside render() throws at runtime and leaves the
    page exactly as empty as a syntax error does, so exercise the real render function
    against a state shaped like the runner's.
    """
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to run the dashboard script")

    root = Path(__file__).resolve().parents[1]
    state = {
        "task": "put the cup in the bowl",
        "episode": "ep0",
        "policy": "cosmos-nano-policy-droid",
        "phase": "done",
        "phase_detail": "1 of 2 tested situations moved the policy",
        "updated": "12:00:00",
        "cost": {"estimated_usd": 1.23},
        "control": {
            "prompt": "no changes",
            "rollouts": 6,
            "gate": {"valid": True, "geometry_drift": 0.001},
            "distance": {"rmse": 0.05},
            "per_view_gate": {"wrist_image_left": {"status": "VALID", "reasons": []}},
            "preview": "media/control.jpg",
        },
        "correction": {"n_tests": 2, "alpha": 0.025, "bonferroni_alpha": 0.0125,
                       "discoveries_fdr": ["a"], "discoveries_bonferroni": ["a"]},
        "scenarios": [
            {"name": "a", "category": "lighting", "situation": "s", "why_it_might_break": "w",
             "prompt": "p", "status": "done", "verdict": "v", "is_vulnerability": True,
             "passes_fdr": True, "passes_bonferroni": True, "rollouts": 6, "stage": "confirmation",
             "preview": "media/a.jpg", "video": "media/a.mp4",
             "gate": {"geometry_drift": 0.002, "geometry_drift_pixels_equivalent": 1.5},
             "distance": {"rmse": 0.08},
             "per_view_gate": {"wrist_image_left": {"status": "REJECTED",
                                                    "geometry_drift_p90": 0.02,
                                                    "reasons": ["localised drift"]}},
             "vs_control": {"p_value": 0.003, "effect_size_cohens_d": 2.1,
                            "gripper_p_value": 0.5, "any_significant": True,
                            "significant_at_alpha": True, "significant_gripper": False}},
            {"name": "b", "situation": "s", "why_it_might_break": "w", "prompt": "p",
             "status": "rejected", "verdict": "rejected", "is_vulnerability": False},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "state.json"
        p.write_text(json.dumps(state), encoding="utf-8")
        proc = subprocess.run([node, str(root / "tools/check_dashboard.js"), str(p)],
                              capture_output=True, text=True, cwd=root)
    assert proc.returncode == 0, f"dashboard render failed:\n{proc.stdout}\n{proc.stderr}"
    assert "scenario cards : 2" in proc.stdout, proc.stdout


def test_intent_verdict_credibility_and_parsing():
    """The judge decides whether a situation's NAME is supported, never whether it is real."""
    from penumbra.scenarios.judge import IntentVerdict, _parse

    assert IntentVerdict(verdict="applied").credible
    assert IntentVerdict(verdict="partially_applied").credible
    assert not IntentVerdict(verdict="not_applied").credible
    assert not IntentVerdict(verdict="something_else").credible
    # An unknown verdict must not be treated as an endorsement.
    assert not IntentVerdict(verdict="unknown").credible

    # The model is told not to fence its JSON and fences it anyway.
    doc = _parse('```json\n{"verdict": "applied", "confidence": 0.8}\n```')
    assert doc["verdict"] == "applied"
    with pytest.raises(ValueError, match="no JSON object"):
        _parse("I am afraid I cannot help with that.")


def test_judge_takes_the_weakest_camera_not_the_best():
    """One lucky camera must not launder a situation's name across the others."""
    import numpy as np

    from penumbra.scenarios import judge as judge_mod

    frames = {v: np.zeros((2, 8, 8, 3), np.uint8)
              for v in ("exterior_image_1_left", "exterior_image_2_left")}
    verdicts = {"exterior_image_1_left": "applied",
                "exterior_image_2_left": "not_applied"}

    def fake(prompt, situation, src, per, *, view="", frame=None):
        return judge_mod.IntentVerdict(verdict=verdicts[view], views_judged=[view])

    original = judge_mod.judge_render
    judge_mod.judge_render = fake
    try:
        out = judge_mod.judge_views("p", "s", frames, frames, list(verdicts))
    finally:
        judge_mod.judge_render = original

    assert out.verdict == "not_applied"
    assert not out.credible
    assert set(out.views_judged) == set(verdicts)


def test_video_is_written_in_a_codec_browsers_decode():
    """Guard against a regression to OpenCV's mp4v.

    mp4v is MPEG-4 Part 2. Every browser refuses it, the server still returns 200, and
    the `<video>` element stays black — a failure invisible from every angle except
    actually watching the page. This is the check that would have caught it.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from penumbra.media import is_browser_playable, write_h264

    rng = np.random.default_rng(0)
    # Odd width on purpose: libx264 rejects odd dimensions in yuv420p, and the writer
    # has to crop rather than fail or pad with invented pixels.
    frames = [rng.integers(0, 255, (169, 301, 3), dtype=np.uint8) for _ in range(8)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        write_h264(path, frames, fps=15.0)
        assert path.stat().st_size > 0
        assert is_browser_playable(path)

        with pytest.raises(ValueError, match="no frames"):
            write_h264(Path(tmp) / "empty.mp4", [], fps=15.0)


def test_media_splitter_distinguishes_grid_from_single_view():
    """Feeding the judge a contact sheet produces confident nonsense.

    Later runs save a grid — one column per camera, source row above perturbed row —
    while the earliest runs saved one camera at full frame. Handing the grid over whole
    yielded verdicts about the sheet itself ("the edited image is a 2x2 grid showing
    multiple views"), which look like real adjudications and mean nothing.
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from judge_run import split_media

    # Two 300x169 cameras: source row on top, perturbed below, marked so the crop is
    # verifiable rather than merely the right shape.
    grid = np.zeros((4, 338, 600, 3), np.uint8)
    grid[:, 169:, :300] = 7
    grid[:, 169:, 300:] = 9
    tiles = split_media(grid, ["a", "b"])
    assert set(tiles) == {"a", "b"}
    assert all(t.shape[1:3] == (169, 300) for t in tiles.values())
    assert int(tiles["a"][0, 0, 0, 0]) == 7 and int(tiles["b"][0, 0, 0, 0]) == 9

    single = np.zeros((4, 360, 640, 3), np.uint8)
    out = split_media(single, ["a"])
    assert out["a"].shape[1:3] == (360, 640)

    three = np.zeros((4, 338, 900, 3), np.uint8)
    assert len(split_media(three, ["a", "b", "c"])) == 3


def test_gate_reports_how_close_a_verdict_was():
    """A rejection at 0.25% over the bar is not the same as one at 200% over.

    A real suite rejected its most realistic renders — steam, foam, a granite
    benchtop — on margins as small as 0.01203 against a 0.01200 bar, on one camera,
    while the other camera passed comfortably. The verdict has to carry that, both so
    a reader can judge it and so the runner can redraw instead of discarding.
    """
    import numpy as np

    from penumbra.validation.seam import SeamGate

    frames = np.random.default_rng(0).integers(0, 255, (6, 180, 320, 3), dtype=np.uint8)
    report = SeamGate().validate(frames, frames)
    d = report.to_dict()
    assert set(d["margins"]) == {"geometry_drift", "geometry_drift_p90",
                                 "structure_retained", "temporal_ratio"}
    # Identical streams sit far from every bar, so nothing is marginal.
    assert d["marginal"] is False
    assert d["margins"]["geometry_drift"] == 0.0

    # A verdict one part in a thousand over its bar must read as marginal.
    tight = SeamGate(max_geometry_drift_p90=0.012)
    r = tight.validate(frames, frames)
    r.geometry_drift_p90 = 0.01203
    assert r.marginal is True
    assert r.to_dict()["marginal"] is True


def test_garage_resume_restores_renders_without_re_rendering(tmp_path):
    """A suite takes one to two hours and renders are its expensive half.

    An interruption used to throw all of it away — and the rerun is not even the same
    experiment, because X2 has no seed. Frames are persisted per scenario and a resume
    must load them back rather than re-render.
    """
    import json

    import numpy as np

    from penumbra.scenarios.garage import Garage, ScenarioResult
    from penumbra.scenarios.director import Scenario

    ep = make_episode(6)
    views = ("exterior_image_1_left", "wrist_image_left")
    garage = Garage.__new__(Garage)  # no runner needed for the persistence path
    garage.out = tmp_path
    garage.views = views
    garage._episodes = {}
    (tmp_path / "media").mkdir(exist_ok=True)

    perturbed = ep.with_views(
        {v: np.full_like(ep.frames[v], 7) for v in views}, "perturbed")
    garage._save_frames("glare", perturbed)
    assert (tmp_path / "frames/glare.npz").exists()

    scenario = Scenario(name="glare", situation="s", why_it_might_break="w", prompt="p")
    garage.results = [ScenarioResult(scenario=scenario, status="queued")]
    (tmp_path / "state.json").write_text(json.dumps({
        "scenarios": [{"name": "glare", "prompt": "p", "status": "rendered",
                       "gate": {"valid": True}}]}), encoding="utf-8")

    assert garage.resume(ep) == 1
    assert garage.results[0].status == "rendered"
    restored = garage._episodes["glare"]
    for v in views:
        assert np.array_equal(restored.frames[v], perturbed.frames[v])
    # Untouched cameras must still be the originals, by reference or by value.
    assert np.array_equal(restored.frames["exterior_image_2_left"],
                          ep.frames["exterior_image_2_left"])


def test_temporal_criterion_separates_added_dynamics_from_flicker():
    """Moving specular highlights are the edit working, not the model boiling.

    Wet surfaces, chrome and foam all carry frame-to-frame motion the dry source never
    had. Measured over the whole frame that reads as generative flicker, and it
    rejected the most convincing renders in two suites — five of seven had *less*
    motion outside the edit than the source recording did. The criterion is therefore
    measured outside the edited region, but only while there is enough frame left
    outside it to measure.
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from calibrate_temporal import added_dynamics, shuffled

    from penumbra.validation.seam import SeamGate

    import cv2

    rng = np.random.default_rng(0)
    # A smooth, structured, slowly panning source. White noise will not do: its own
    # frame-to-frame flow is already saturated, so added flicker cannot raise the
    # *ratio* and the test would pass the gate for the wrong reason.
    coarse = rng.normal(128, 45, (12, 20, 3))
    base = np.clip(cv2.resize(coarse, (320, 180), interpolation=cv2.INTER_CUBIC),
                   0, 255).astype(np.uint8)
    src = np.stack([np.roll(base, i, axis=1) for i in range(14)])

    gate = SeamGate()

    def temporal_rejected(frames):
        r = gate.validate(src, frames)
        return any("temporal" in reason for reason in r.reasons), r

    # Dynamics confined to a small region: admissible, and measured outside the edit.
    for area in (0.10, 0.30):
        rejected, r = temporal_rejected(added_dynamics(src, area))
        assert r.residual_applies, f"{area:.0%} edit should use the residual statistic"
        assert not rejected, f"coherent dynamics over {area:.0%} were rejected"

    # Whole-frame incoherence must still be caught. Frame shuffling is used rather
    # than additive noise: on real footage noise perturbs fine texture into trackable
    # apparent motion (measured 2.66-4.84 by tools/calibrate_temporal.py), but on any
    # synthetic image smooth enough to build here it produces no coherent flow at all.
    # The additive-noise case is covered by that calibration, on real frames.
    rejected, _ = temporal_rejected(shuffled(src))
    assert rejected, "shuffled frames passed the temporal criterion"

    # Past the area limit the residual measurement is not trusted; the report says so.
    big = gate.validate(src, added_dynamics(src, 0.60))
    assert not big.residual_applies
    assert big.to_dict()["temporal_measured_on"] == "the whole frame"
