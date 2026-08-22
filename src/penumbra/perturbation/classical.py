"""Classical augmentation control arm.

This is the honest competitor, and the reason PENUMBRA's central claim is falsifiable
at all. If ordinary pixel-level augmentation finds the same failures at the same
perceptual distance, the generative machinery buys nothing and we should say so.

Every operator here is scene-blind by construction — it applies a global pixel
transform with no notion of what a table, a floor, or a cup is. That is precisely the
ceiling being tested: a global gamma change cannot make one surface reflective while
leaving the objects standing on it alone.

Deterministic given `(name, strength, seed)`, unlike X2, which has no seed. So the
control arm is exactly reproducible and the generative arm is not — a real asymmetry,
stated rather than hidden.

Calibration: `match_perceptual_distance` searches the classical strength axis for the
setting whose perceptual distance from the source best matches a given generative
perturbation. Comparing the two arms at *matched* distance is what stops the
comparison from being "one change was simply bigger".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Callable

import cv2
import numpy as np

from ..episodes.types import PERTURBED_VIEW, Episode
from ..validation.seam import perceptual_distance


def _brightness_contrast(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    gain = 1.0 + 1.6 * s
    bias = 60.0 * s
    return np.clip(frame.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def _gamma_glare(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """Gamma lift plus a synthetic radial bloom — the classical stand-in for glare."""
    g = 1.0 / (1.0 + 1.8 * s)
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    out = cv2.LUT(frame, lut).astype(np.float32)
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.18, w * 0.72
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    bloom = np.exp(-(r ** 2) / (2 * (0.28 ** 2))) * 235.0 * s
    return np.clip(out + bloom[..., None], 0, 255).astype(np.uint8)


def _blur_noise(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    k = int(1 + 2 * round(6 * s))
    out = cv2.GaussianBlur(frame, (k, k), 0) if k > 1 else frame.copy()
    noise = rng.normal(0, 26.0 * s, out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _color_shift(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 40.0 * s) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + 1.2 * s), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _specular_overlay(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """The strongest *classical* attempt at a wet floor: a bright gradient over the
    lower half of the frame, screen-blended. Scene-blind by construction — it does not
    know where the table ends or that the cup should not become reflective."""
    h, w = frame.shape[:2]
    ramp = np.clip(np.linspace(-0.4, 1.0, h), 0, 1)[:, None] ** 1.5
    sheen = (ramp * np.ones((1, w))) * 255.0 * s
    base = frame.astype(np.float32)
    screened = 255.0 - (255.0 - base) * (255.0 - sheen[..., None]) / 255.0
    return np.clip(screened, 0, 255).astype(np.uint8)


def _blackout(frame: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    """Fade the whole view to black. **This is a positive control, not a fault.**

    Every null result PENUMBRA has produced so far carries the same confound: the
    policy reads three camera views, and only `exterior_image_1_left` is perturbed. If
    the policy barely uses that view, no perturbation of it could ever move behaviour,
    and every "robust" verdict would be measuring the wiring rather than the policy.

    At strength 1.0 this destroys the view completely. If *that* does not shift the
    action distribution, the experiment has no sensitivity on this channel and no null
    from it means anything. A positive control is the only thing that distinguishes
    "the policy is robust" from "the measurement cannot see".
    """
    return np.clip(frame.astype(np.float32) * (1.0 - s), 0, 255).astype(np.uint8)


CLASSICAL_OPS: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
    "brightness_contrast": _brightness_contrast,
    "gamma_glare": _gamma_glare,
    "blur_noise": _blur_noise,
    "color_shift": _color_shift,
    "specular_overlay": _specular_overlay,
    "blackout": _blackout,
}

#: Ops that exist to test the apparatus rather than to model a real-world fault.
#: They are never perceptual-distance matched against a generative arm, because the
#: whole point is that they are as large a change as the channel admits.
POSITIVE_CONTROLS = frozenset({"blackout"})


@dataclass
class ClassicalResult:
    episode: Episode
    op: str
    strength: float
    seed: int
    view: str
    distance: dict
    views: tuple[str, ...] = ()
    per_view_distance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "strength": round(self.strength, 4),
            "seed": self.seed,
            "view": self.view,
            "views": list(self.views) or [self.view],
            "per_view_distance": {
                v: {k: round(x, 5) for k, x in d.items()}
                for v, d in self.per_view_distance.items()
            },
            "distance": {k: round(v, 5) for k, v in self.distance.items()},
            "model": "classical/opencv",
            "deterministic": True,
        }


class ClassicalPerturbation:
    """Deterministic, scene-blind, free. The baseline the thesis must beat.

    Operates on one view by default, or on several at once. Multi-view matters here
    because the policy reads three cameras plus proprioception: perturbing one leaves
    it two clean views to fall back on, so a single-view null may be measuring
    redundancy rather than robustness.
    """

    model = "classical/opencv"

    def __init__(
        self,
        op: str = "specular_overlay",
        *,
        view: str = PERTURBED_VIEW,
        views: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if op not in CLASSICAL_OPS:
            raise ValueError(f"unknown classical op {op!r}. Available: {sorted(CLASSICAL_OPS)}")
        self.op = op
        self.views = tuple(views) if views else (view,)
        self.view = self.views[0]
        self.seed = seed

    def apply(self, episode: Episode, strength: float, *, run_id: str = "0") -> ClassicalResult:
        fn = CLASSICAL_OPS[self.op]
        replacements: dict[str, np.ndarray] = {}
        per_view: dict[str, dict] = {}
        for view in self.views:
            source = episode.frames[view]
            # A fresh generator per view, so a view's pixels never depend on which
            # other views happen to be in the set - otherwise "camera 1 only" and
            # "camera 1 + 2" would apply different noise to camera 1.
            rng = np.random.default_rng(self.seed)
            out = np.stack([fn(f, float(strength), rng) for f in source])
            replacements[view] = out
            per_view[view] = perceptual_distance(source, out)

        tag = "+".join(v.split("_")[0] + v.split("_")[-2] for v in self.views) \
            if len(self.views) > 1 else self.views[0]
        perturbed = episode.with_views(
            replacements,
            episode_id=f"{episode.episode_id}#classical-{self.op}[{tag}]@{strength:g}r{run_id}",
        )
        # The headline distance is the mean over perturbed views. Reported per view as
        # well, because averaging hides which channel actually changed.
        keys = per_view[self.views[0]].keys()
        mean_distance = {
            k: float(np.mean([per_view[v][k] for v in self.views])) for k in keys
        }
        return ClassicalResult(
            episode=perturbed,
            op=self.op,
            strength=float(strength),
            seed=self.seed,
            view=self.view,
            views=self.views,
            distance=mean_distance,
            per_view_distance=per_view,
        )


def match_perceptual_distance(
    episode: Episode,
    op: str,
    target: dict,
    *,
    view: str = PERTURBED_VIEW,
    seed: int = 0,
    key: str = "rmse",
    steps: int = 24,
) -> ClassicalResult:
    """Find the classical strength whose perceptual distance best matches `target`.

    A coarse-to-fine scan rather than a solver: the map from strength to distance is
    monotone-ish but not analytic, and 24 evaluations of a cheap OpenCV op costs
    nothing next to one Reactor session.
    """
    goal = float(target[key])
    perturber = ClassicalPerturbation(op, view=view, seed=seed)
    lo, hi = 0.0, 1.0
    best: ClassicalResult | None = None
    best_gap = float("inf")
    for _ in range(3):
        for s in np.linspace(lo, hi, steps // 3):
            result = perturber.apply(episode, float(s))
            gap = abs(result.distance[key] - goal)
            if gap < best_gap:
                best, best_gap = result, gap
        assert best is not None
        span = (hi - lo) / 4
        lo, hi = max(0.0, best.strength - span), min(1.0, best.strength + span)
    assert best is not None
    return best
