"""Experiment records and their on-disk artifacts.

Every run writes a directory that contains enough to reproduce it and enough to
argue with it:

    runs/<run_id>/
        experiment.json      the full record: inputs, all metrics, the verdict
        source.mp4           the real episode's perturbed-view stream, untouched
        perturbed.mp4        what the policy actually saw
        side_by_side.mp4     the two, with the strength and gate status burned in
        actions_baseline.npy
        actions_perturbed.npy
        plots/              divergence and gripper traces
        report.html         the human-readable failure report

No result is ever written that the pipeline did not actually produce. Rejected
perturbations are stored with `validation.status == "REJECTED"` and are not counted
as failures anywhere.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from ..config import RUNS_DIR

SCHEMA_VERSION = "penumbra-experiment/1"


@dataclass
class ExperimentRecord:
    """One perturbation evaluated against one episode with one policy."""

    run_id: str
    created_at: str
    episode: dict
    policy: dict
    perturbation: dict
    validation: dict
    noise_floor: dict
    divergence: dict
    verdict: dict
    cost: dict = field(default_factory=dict)
    corroboration: dict | None = None
    classical_control: dict | None = None
    environment: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    schema: str = SCHEMA_VERSION
    #: Anything a later version of PENUMBRA - or a correction applied by hand - added
    #: to the record. Carried through load/save rather than dropped, so annotating an
    #: artifact (e.g. marking it superseded) never makes it unreadable.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # `extra` is flattened back to the top level so a record round-trips to the
        # same shape it was read in as.
        d.update(d.pop("extra", {}) or {})
        return d


def environment_fingerprint() -> dict:
    """Enough to reproduce, with nothing sensitive in it."""
    try:
        import reactor_sdk

        sdk = reactor_sdk.__version__
    except Exception:  # noqa: BLE001
        sdk = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "reactor_sdk": sdk,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_video(frames: np.ndarray, path: Path, fps: float = 15.0) -> Path:
    """Write RGB frames to mp4. Falls back to MJPG/AVI if the mp4 encoder is absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def write_side_by_side(
    source: np.ndarray,
    perturbed: np.ndarray,
    path: Path,
    *,
    fps: float = 15.0,
    caption: str = "",
    right_label: str = "PERTURBED",
) -> Path:
    """Source | perturbed, with a caption strip. The evidence artifact."""
    n = min(len(source), len(perturbed))
    h, w = source.shape[1:3]
    strip = 34
    out = np.zeros((h + strip, w * 2, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        out[:] = 0
        out[strip:, :w] = source[i]
        out[strip:, w:] = cv2.resize(perturbed[i], (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(out, "SOURCE (real robot log)", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(out, right_label, (w + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1, cv2.LINE_AA)
        if caption:
            cv2.putText(out, caption, (w - 6 - 7 * len(caption) // 2, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        frames.append(out.copy())
    return write_video(np.stack(frames), path, fps)


def save_record(record: ExperimentRecord, directory: Path) -> Path:
    path = directory / "experiment.json"
    path.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_record(directory: Path) -> ExperimentRecord:
    """Read a record, tolerating fields this version does not know about.

    Records are long-lived evidence. A reader that crashes on an unrecognised key
    makes every artifact unreadable the moment anyone annotates one - which is exactly
    what happened when a superseded run was marked as such.
    """
    doc = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    doc.pop("schema", None)
    known = {f.name for f in fields(ExperimentRecord)} - {"schema", "extra"}
    extra = {k: v for k, v in doc.items() if k not in known}
    fixed = {k: v for k, v in doc.items() if k in known}
    return ExperimentRecord(schema=SCHEMA_VERSION, extra=extra, **fixed)


def list_runs(prefix: str | None = None) -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    out = [
        d for d in sorted(RUNS_DIR.iterdir())
        if d.is_dir() and (d / "experiment.json").exists()
        and (prefix is None or d.name.startswith(prefix))
    ]
    return out
