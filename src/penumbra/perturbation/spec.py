"""Faults as structured, versioned programs — not prose scattered through the code.

A `FaultSpec` is loaded from YAML in `faults/`. It owns the one thing X2 does not
give us: a **strength axis**.

X2 exposes no numeric strength parameter (verified absent — see
docs/REACTOR_CAPABILITIES.md §1). The only channel to intensity is prompt language.
So PENUMBRA constructs the axis explicitly, as an ordered ladder of prompts, and says
so everywhere it reports a number.

    strength 0.0  -> the source, untouched (the ladder's zero rung is a no-op)
    strength 0.5  -> the rung at or below 0.5
    strength 1.0  -> the top rung

Two honesty constraints follow, and both are enforced rather than assumed:

1. The axis is **ordinal by construction**. That the model's response is monotonic in
   rung index is a hypothesis. `measured_distance` records the perceptual distance
   actually achieved at each rung so monotonicity can be checked, not asserted.
2. Because X2 has no seed, the same rung run twice is not the same video. Repeat runs
   are the only way to know the generative variance, and the report carries it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import REPO_ROOT

FAULTS_DIR = REPO_ROOT / "faults"


@dataclass(frozen=True)
class Rung:
    """One step on a fault's strength ladder."""

    strength: float
    prompt: str
    label: str = ""


@dataclass(frozen=True)
class FaultSpec:
    """A named, parameterised, reproducible perturbation program."""

    name: str
    family: str
    model: str
    description: str
    ladder: tuple[Rung, ...]
    version: str = "1"
    pointer: dict | None = None
    reference_image: str | None = None
    validation: dict = field(default_factory=dict)
    corroborate_with: str | None = None

    @property
    def strengths(self) -> tuple[float, ...]:
        return tuple(r.strength for r in self.ladder)

    def pointer_for(self, view: str) -> dict | None:
        """The drag pointer for one camera view, or None.

        A spatial anchor is only meaningful in the frame it was measured in. The same
        object sits at very different image coordinates across the DROID rig - the
        yellow cup is at (0.587, 0.642) in exterior 1, (0.589, 0.755) in exterior 2,
        and (0.304, 0.466) in the wrist view, where it also covers roughly fifty times
        more pixels. Reusing one coordinate across views would aim the intervention at
        empty table in two of the three.

        So `pointer` may be either a flat mapping (one anchor, used for every view -
        correct only for single-view faults) or a mapping keyed by view name.
        """
        if not self.pointer:
            return None
        if any(k in self.pointer for k in ("x", "y", "active")):
            return self.pointer
        return self.pointer.get(view)

    @property
    def max_geometry_drift(self) -> float:
        return float(self.validation.get("max_geometry_drift", 0.05))

    @property
    def min_structure_retained(self) -> float:
        return float(self.validation.get("min_structure_retained", 0.35))

    def rung_at(self, strength: float) -> Rung:
        """The rung at or below `strength`. Clamped to the ladder's range."""
        if strength <= self.ladder[0].strength:
            return self.ladder[0]
        chosen = self.ladder[0]
        for rung in self.ladder:
            if rung.strength <= strength + 1e-9:
                chosen = rung
            else:
                break
        return chosen

    def is_noop(self, strength: float) -> bool:
        """True when this strength means 'do not perturb at all'."""
        return not self.rung_at(strength).prompt.strip()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "model": self.model,
            "description": self.description,
            "ladder": [{"strength": r.strength, "label": r.label, "prompt": r.prompt}
                       for r in self.ladder],
            "pointer": self.pointer,
            "reference_image": self.reference_image,
            "validation": self.validation,
            "corroborate_with": self.corroborate_with,
        }


def load_fault(name: str, directory: Path | None = None) -> FaultSpec:
    path = (directory or FAULTS_DIR) / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in (directory or FAULTS_DIR).glob("*.yaml")))
        raise FileNotFoundError(f"no fault named {name!r}. Available: {available}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    ladder = tuple(
        Rung(strength=float(r["strength"]), prompt=str(r.get("prompt", "")), label=str(r.get("label", "")))
        for r in sorted(doc["ladder"], key=lambda r: float(r["strength"]))
    )
    if not ladder or ladder[0].strength != 0.0:
        raise ValueError(f"{name}: ladder must start at strength 0.0 (the untouched control)")
    if ladder[0].prompt.strip():
        raise ValueError(f"{name}: the 0.0 rung must be a no-op with an empty prompt")
    return FaultSpec(
        name=doc.get("name", name),
        family=doc.get("family", "unspecified"),
        model=doc.get("model", "xmax/x2"),
        description=doc.get("description", ""),
        ladder=ladder,
        version=str(doc.get("version", "1")),
        pointer=doc.get("pointer"),
        reference_image=doc.get("reference_image"),
        validation=doc.get("validation", {}) or {},
        corroborate_with=doc.get("corroborate_with"),
    )


def list_faults(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or FAULTS_DIR).glob("*.yaml"))
