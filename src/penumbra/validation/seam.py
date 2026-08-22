"""SEAM — the validity gate.

A generative transformation may quietly move the scene. If it does, a policy's
changed behaviour tells us about the transformation, not about the policy. SEAM's
job is to reject those cases *before* anything is reported as a finding.

This is not perfect geometry estimation and does not claim to be. It is a
**defensible rejection mechanism** built from three cheap, independent signals:

1. **Geometry drift** — dense optical flow (Farnebäck) computed source→perturbed on
   the *same* timestamp. If the perturbation only changed appearance, corresponding
   points stay put and the flow field is near zero. Normalised by the frame diagonal,
   so the threshold is resolution-independent.

   Two statistics, because they fail differently. The **median** catches a whole scene
   sliding or warping but is blind to a small region moving — exactly the aperture
   problem, and a 22-pixel shift of a low-texture image can score a median of zero.
   The **90th percentile** catches that localised case but is noisy on textureless
   regions. Both are gated, with separate thresholds.

   **The thresholds are measured, not guessed.** `tools/calibrate_gate.py` runs known
   appearance-only transforms and known geometric shifts over the actual episode and
   reports where they separate. On 640-wide DROID footage:

       identity                      0.00000
       gamma 0.5                     0.00037
       brightness x1.5 + 40          0.00048
       X2 specular_floor @ 1.0       0.00154   <- real appearance-only edit
       2 px global shift             0.00272
       3 px global shift             0.00408   (interpolated)
       ---- threshold 0.00550 ----
       4 px global shift             0.00545
       32 px global shift            0.04356

   The research document proposed 0.05. That would only have rejected shifts larger
   than about 36 pixels — it would have passed almost any corruption. Numbers
   inherited from a design document are not measurements.

   The threshold sits at the 4-pixel mark rather than as low as the data allows, and
   that choice is a judgement worth stating. The gate's question is not "did any pixel
   move" but "did geometry move enough to plausibly explain a behaviour change". The
   policy consumes downscaled frames, so 4 px on a 640-wide source is on the order of
   1 px at its input — below what could drive an 8-DoF action change on its own.
   Setting the bar at the noise floor of the flow estimator instead would reject
   legitimate appearance edits and teach us nothing. The measured drift, and its
   pixel equivalent, appear in every report so a reader can apply a stricter bar.

2. **Structure retained** — gradient-orientation agreement between source and
   perturbed. Edges are where geometry lives; appearance edits move intensities but
   should leave the edge *structure* largely in place. Low agreement means the model
   redrew the scene rather than relit it.

3. **Temporal consistency** — flow magnitude *within* the perturbed stream compared
   with flow magnitude within the source. A generative stream that flickers or boils
   has motion the recording never had, and that motion is a confound.

A perturbation is VALID only if all three pass. Anything else is REJECTED and is
never reported as a policy failure.

### Why the gate compares against the SOURCE and not against the control

The statistics compare each scenario to a no-op re-render, so that X2's own redrawing
cancels. The gate does not: it compares against the raw recording. That asymmetry is
deliberate, and it was checked rather than assumed.

Geometry validity is a property relative to *reality*. If the control and the scenario
both drifted the same way, gating one against the other would pass a render whose
geometry no longer matches the real scene - which is exactly the corruption this exists
to catch.

Measured on 21 renders (`runs/gatesweep-ep0-20260821-162003`), gating against the
control instead of the source lowers median localised drift from 0.00284 to 0.00219 on
`exterior_image_1_left` and from 0.00248 to 0.00165 on `exterior_image_2_left` - and
changes no verdict at all: 16/21 valid under either reference. The choice is defensible
on principle and, on this data, does not matter in practice.

### Honest limitations
- Farnebäck flow is unreliable in low-texture regions and under large illumination
  change; the glare fault will stress it. Signal 2 exists partly to cover that.
- A displacement of a locally-featureless region (a flat gradient, a blank wall) is
  invisible to *any* dense-flow method — the aperture problem. The p90 statistic
  narrows this but does not close it.
- These signals detect *displacement*, not *hallucinated content that happens to sit
  still*. An object cleanly inpainted out of a static region can pass the gate. That
  is exactly why the PROVENANCE fault's result is framed as a hypothesis.
- No depth, no SfM, no segmentation. Adding any of them would strengthen the gate.
- **A small region can move a long way and the gate will not see it.** Measured
  (`tools/calibrate_localised.py`): displacing 5% of the frame by 16 px scores a p90 of
  0.00039, below even a gamma curve. p90 only rises once the moved region is large
  enough to reach the top decile of the flow field, which needs roughly 10% of the
  frame. The original calibration could not have found this, because it used only
  *global* shifts - whose p90 equals their median by construction, so the p90 bar had
  never been measured against localised displacement at all.
- Two candidate replacements were tested and both failed their own calibration.
  Restricting drift to pixels whose colour did *not* change does not separate: a
  brightness change alone scores 0.0159, above several genuine corruptions. ORB keypoint
  matching does separate the synthetic set, but by about a third of a pixel, and on real
  renders heavy lighting changes destroy the descriptors (190 matches versus 900), so it
  is not demonstrably better than the incumbent.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ValidationReport:
    """The gate's verdict, with every number that produced it."""

    valid: bool
    geometry_drift: float
    geometry_drift_p90: float
    structure_retained: float
    temporal_ratio: float
    residual_temporal_ratio: float
    edited_area: float
    max_geometry_drift: float
    max_geometry_drift_p90: float
    min_structure_retained: float
    max_temporal_ratio: float
    max_residual_temporal_ratio: float
    residual_area_limit: float
    reasons: list[str]
    per_frame_drift: list[float]

    @property
    def residual_applies(self) -> bool:
        """Is the edit small enough for the outside-the-edit measurement to be usable?

        Past roughly 40% of the frame there is too little left outside the edit to
        measure, and the statistic degrades badly (2.19 at 40%, 3.07 at 50%, 4.73 at
        60% for the same kind of coherent added motion). Beyond the limit the
        whole-frame ratio is the honest fallback, even though it over-charges dynamics
        to flicker.
        """
        return self.edited_area <= self.residual_area_limit

    @property
    def status(self) -> str:
        return "VALID" if self.valid else "REJECTED"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "valid": self.valid,
            "geometry_drift": round(self.geometry_drift, 5),
            "geometry_drift_pixels_equivalent": round(drift_in_pixels(self.geometry_drift), 2),
            "geometry_drift_p90": round(self.geometry_drift_p90, 5),
            "structure_retained": round(self.structure_retained, 4),
            "temporal_ratio": round(self.temporal_ratio, 4),
            "residual_temporal_ratio": round(self.residual_temporal_ratio, 4),
            "edited_area": round(self.edited_area, 4),
            "temporal_measured_on": ("outside the edit" if self.residual_applies
                                     else "the whole frame"),
            "thresholds": {
                "max_geometry_drift": self.max_geometry_drift,
                "max_geometry_drift_p90": self.max_geometry_drift_p90,
                "min_structure_retained": self.min_structure_retained,
                "max_temporal_ratio": self.max_temporal_ratio,
                "max_residual_temporal_ratio": self.max_residual_temporal_ratio,
                "residual_area_limit": self.residual_area_limit,
            },
            "reasons": self.reasons,
            # How close each criterion came to its bar, as a fraction of the bar. A
            # render rejected at 0.25% over the line is a different thing from one at
            # 200% over, and a verdict that hides the margin invites both complacency
            # and second-guessing.
            "margins": {
                "geometry_drift": round(self.geometry_drift / self.max_geometry_drift, 3),
                "geometry_drift_p90": round(
                    self.geometry_drift_p90 / self.max_geometry_drift_p90, 3),
                "structure_retained": round(
                    self.min_structure_retained / max(self.structure_retained, 1e-6), 3),
                "temporal_ratio": round(self.temporal_ratio / self.max_temporal_ratio, 3),
            },
            "marginal": self.marginal,
        }

    #: A criterion within this fraction of its bar either way is called marginal -
    #: close enough that X2's own run-to-run variation could flip the verdict.
    MARGIN = 0.10

    @property
    def marginal(self) -> bool:
        """Did any criterion land within 10% of its threshold?

        X2 has no seed, so a render this close to a bar would plausibly fall on the
        other side of it on a second draw. Saying so is not an argument for moving the
        bar. It is an argument for drawing again.
        """
        ratios = (
            self.geometry_drift / self.max_geometry_drift,
            self.geometry_drift_p90 / self.max_geometry_drift_p90,
            (self.residual_temporal_ratio / self.max_residual_temporal_ratio
             if self.residual_applies
             else self.temporal_ratio / self.max_temporal_ratio),
            self.min_structure_retained / max(self.structure_retained, 1e-6),
        )
        return bool(any(1.0 - self.MARGIN <= r <= 1.0 + self.MARGIN for r in ratios))


#: Measured on real DROID footage by tools/calibrate_gate.py: a global shift of one
#: pixel on a 640-wide frame produces this much normalised median flow. Used only to
#: translate a drift number into something a human can picture.
DRIFT_PER_PIXEL_AT_640 = 0.00136


#: A pixel counts as edited when its mean RGB moves this far. Shared with
#: tools/calibrate_temporal.py and tools/diagnose_rejection.py so all three speak
#: about the same regions.
EDIT_COLOUR_DELTA = 26.0


def drift_in_pixels(drift: float, width: int = 640) -> float:
    """`drift` expressed as the equivalent global shift, in source pixels."""
    return drift / DRIFT_PER_PIXEL_AT_640 * (width / 640.0)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _flow_magnitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return np.linalg.norm(flow, axis=2)


def _gradient_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine agreement of image gradients, weighted by gradient strength.

    Illumination-robust in a way raw SSIM is not: a uniform brightening leaves
    gradient *direction* intact even as magnitudes change, so a relit scene scores
    high while a redrawn one does not.
    """
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    dot = ax * bx + ay * by
    na = np.sqrt(ax * ax + ay * ay)
    nb = np.sqrt(bx * bx + by * by)
    cos = dot / (na * nb + 1e-6)
    weight = np.minimum(na, nb)
    total = weight.sum()
    if total < 1e-6:
        return 0.0
    return float((cos * weight).sum() / total)


class SeamGate:
    """Decides whether a perturbed stream is still the same physical scene."""

    def __init__(
        self,
        *,
        max_geometry_drift: float = 0.0055,
        max_geometry_drift_p90: float = 0.012,
        min_structure_retained: float = 0.35,
        max_temporal_ratio: float = 2.5,
        max_residual_temporal_ratio: float = 2.48,
        residual_area_limit: float = 0.40,
        sample: int = 12,
        work_width: int = 320,
    ) -> None:
        self.max_geometry_drift = max_geometry_drift
        # Looser than the global statistic, and deliberately so: a real appearance
        # edit does move some edge pixels around, and the calibration shows the p90 of
        # a legitimate X2 edit (0.0036) already exceeds that of a 2 px global shift
        # (0.0027). So p90 cannot separate small global shifts — it is here only to
        # catch large *localised* displacement that the median misses.
        self.max_geometry_drift_p90 = max_geometry_drift_p90
        self.min_structure_retained = min_structure_retained
        self.max_temporal_ratio = max_temporal_ratio
        # Temporal motion INSIDE the edited region is the edit doing its job: wet
        # surfaces, chrome and foam all carry moving specular highlights that the dry
        # source never had. Measuring the whole frame charges that to flicker, and it
        # rejected the most convincing renders in two suites - five of seven had LESS
        # frame-to-frame motion outside the edit than the source recording did.
        #
        # So the criterion is measured outside the edit when it can be. Calibrated
        # blind on synthetic streams (tools/calibrate_temporal.py): coherent dynamics
        # confined to <= 40% of the frame score at most 2.187 residual, while the
        # smallest genuine flicker scores 2.809. The bar is their geometric mean.
        # Beyond 40% the region outside the edit is too small to measure and the
        # statistic degrades (3.07 at 50%, 4.73 at 60%), so past that limit the
        # whole-frame ratio is used instead.
        self.max_residual_temporal_ratio = max_residual_temporal_ratio
        self.residual_area_limit = residual_area_limit
        self.sample = sample
        self.work_width = work_width

    def _prep_rgb(self, frames: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
        """Colour frames at working resolution, for locating the edited region."""
        h, w = frames.shape[1:3]
        tw = self.work_width
        th = int(round(h * tw / w))
        return [cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA) for i in idx]

    def _prep(self, frames: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
        h, w = frames.shape[1:3]
        tw = self.work_width
        th = int(round(h * tw / w))
        return [_gray(cv2.resize(frames[i], (tw, th), interpolation=cv2.INTER_AREA)) for i in idx]

    def validate(self, source: np.ndarray, perturbed: np.ndarray) -> ValidationReport:
        if len(source) != len(perturbed):
            raise ValueError(f"length mismatch: {len(source)} vs {len(perturbed)}")
        n = len(source)
        idx = np.unique(np.linspace(0, n - 1, min(self.sample, n)).astype(int))
        src = self._prep(source, idx)
        per = self._prep(perturbed, idx)
        diag = float(np.hypot(*src[0].shape))

        flows = [_flow_magnitude(a, b) for a, b in zip(src, per)]
        drifts = [float(np.median(f) / diag) for f in flows]
        drifts_p90 = [float(np.percentile(f, 90) / diag) for f in flows]
        structures = [_gradient_agreement(a, b) for a, b in zip(src, per)]

        # Temporal: consecutive-frame motion inside each stream, compared.
        pair_idx = np.unique(np.linspace(0, n - 2, min(self.sample, max(1, n - 1))).astype(int))
        src_pairs_a = self._prep(source, pair_idx)
        src_pairs_b = self._prep(source, pair_idx + 1)
        per_pairs_a = self._prep(perturbed, pair_idx)
        per_pairs_b = self._prep(perturbed, pair_idx + 1)
        src_flows = [_flow_magnitude(a, b) for a, b in zip(src_pairs_a, src_pairs_b)]
        per_flows = [_flow_magnitude(a, b) for a, b in zip(per_pairs_a, per_pairs_b)]
        src_motion = float(np.mean([np.median(f) for f in src_flows]))
        per_motion = float(np.mean([np.median(f) for f in per_flows]))
        temporal_ratio = per_motion / (src_motion + 1e-6)

        # The same comparison restricted to pixels the edit did NOT recolour. Motion
        # inside the edit is the edit working; motion outside it is the model boiling.
        pair_src_rgb = self._prep_rgb(source, pair_idx)
        pair_per_rgb = self._prep_rgb(perturbed, pair_idx)
        out_src, out_per, areas = [], [], []
        for a_rgb, b_rgb, fs, fp in zip(pair_src_rgb, pair_per_rgb, src_flows, per_flows):
            delta = np.abs(a_rgb.astype(np.float32) - b_rgb.astype(np.float32)).mean(axis=2)
            changed = cv2.morphologyEx((delta > EDIT_COLOUR_DELTA).astype(np.uint8),
                                       cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
            areas.append(float(changed.mean()))
            if (~changed).sum() > 64:
                out_src.append(float(np.median(fs[~changed])))
                out_per.append(float(np.median(fp[~changed])))
        edited_area = float(np.mean(areas)) if areas else 0.0
        residual_temporal = ((float(np.mean(out_per)) / (float(np.mean(out_src)) + 1e-6))
                             if out_src else temporal_ratio)

        drift = float(np.median(drifts))
        drift_p90 = float(np.median(drifts_p90))
        structure = float(np.median(structures))

        reasons: list[str] = []
        if drift > self.max_geometry_drift:
            reasons.append(
                f"geometry drift {drift:.5f} exceeds {self.max_geometry_drift:.5f} "
                f"(equivalent to a {drift_in_pixels(drift):.1f} px shift; the scene "
                f"moved, it was not just relit)"
            )
        if drift_p90 > self.max_geometry_drift_p90:
            reasons.append(
                f"localised geometry drift (p90) {drift_p90:.5f} exceeds "
                f"{self.max_geometry_drift_p90:.5f} (equivalent to a "
                f"{drift_in_pixels(drift_p90):.1f} px shift of part of the scene)"
            )
        if structure < self.min_structure_retained:
            reasons.append(
                f"structure retained {structure:.3f} below {self.min_structure_retained:.3f} "
                f"(edges redrawn rather than preserved)"
            )
        if edited_area <= self.residual_area_limit:
            if residual_temporal > self.max_residual_temporal_ratio:
                reasons.append(
                    f"temporal motion ratio outside the edited region "
                    f"{residual_temporal:.2f} exceeds {self.max_residual_temporal_ratio:.2f} "
                    f"(the untouched part of the scene is boiling, which is flicker "
                    f"rather than the edit; whole-frame ratio {temporal_ratio:.2f}, "
                    f"edit covers {edited_area:.0%} of the frame)"
                )
        elif temporal_ratio > self.max_temporal_ratio:
            reasons.append(
                f"temporal motion ratio {temporal_ratio:.2f} exceeds "
                f"{self.max_temporal_ratio:.2f} (generative flicker not present in the "
                f"source; the edit covers {edited_area:.0%} of the frame, too much to "
                f"measure outside it)"
            )

        return ValidationReport(
            valid=not reasons,
            geometry_drift=drift,
            geometry_drift_p90=drift_p90,
            structure_retained=structure,
            temporal_ratio=temporal_ratio,
            residual_temporal_ratio=residual_temporal,
            edited_area=edited_area,
            max_geometry_drift=self.max_geometry_drift,
            max_geometry_drift_p90=self.max_geometry_drift_p90,
            min_structure_retained=self.min_structure_retained,
            max_temporal_ratio=self.max_temporal_ratio,
            max_residual_temporal_ratio=self.max_residual_temporal_ratio,
            residual_area_limit=self.residual_area_limit,
            reasons=reasons,
            per_frame_drift=[round(d, 5) for d in drifts],
        )


def perceptual_distance(source: np.ndarray, perturbed: np.ndarray, sample: int = 12) -> dict:
    """How different do the two streams *look*?

    Used to calibrate the classical control arm to matched perceptual distance, so the
    generative-vs-classical comparison is not confounded by one arm simply being a
    bigger change than the other. Three complementary measures; no LPIPS network is
    used because it would need weights and a GPU for no gain at this scale.
    """
    n = len(source)
    idx = np.unique(np.linspace(0, n - 1, min(sample, n)).astype(int))
    rmse, ssim_vals, hist = [], [], []
    for i in idx:
        a = cv2.resize(source[i], (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(perturbed[i], (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        rmse.append(float(np.sqrt(np.mean((a - b) ** 2)) / 255.0))
        ga, gb = _gray(a.astype(np.uint8)), _gray(b.astype(np.uint8))
        ssim_vals.append(_ssim(ga, gb))
        ha = cv2.calcHist([a.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
        hb = cv2.calcHist([b.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
        ha /= ha.sum() + 1e-8
        hb /= hb.sum() + 1e-8
        hist.append(float(np.abs(ha - hb).sum() / 2.0))
    return {
        "rmse": float(np.mean(rmse)),
        "ssim": float(np.mean(ssim_vals)),
        "ssim_drop": float(1.0 - np.mean(ssim_vals)),
        "hist_distance": float(np.mean(hist)),
    }


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on grayscale, Gaussian-windowed (standard Wang et al. formulation)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float(np.mean(num / den))
