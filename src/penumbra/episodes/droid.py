"""Loader for DROID episodes in LeRobot v3 packed format.

Source: ``nvidia/Cosmos3-DROID`` (HuggingFace). Chosen because it is NVIDIA's own
DROID packaging and its columns line up exactly with what
``reactor/cosmos-nano-policy-droid`` consumes and emits:

    observation.state.joint_positions  (7)   -> set_proprio_json joint_position
    observation.state.gripper_position (1)   -> set_proprio_json gripper_position
    action.joint_position (7) + action.gripper_position (1)
                                             -> the same 8-DoF action the policy predicts

So the ground-truth action and the predicted action are directly comparable, in the
same units, without any convention guessing.

LeRobot v3 packs many episodes into one mp4; an episode is a timestamp window into
it. Decoding uses PyAV (bundled libdav1d) because the videos are AV1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from ..config import DATA_DIR
from .types import VIEWS, Episode

REPO = "nvidia/Cosmos3-DROID"
SPLIT = "success"
_VIDEO_KEY = "observation.image.{view}"


def _cached(rel: str) -> Path:
    """Locate a file already fetched by tools/fetch_droid.py, without re-downloading."""
    root = DATA_DIR / "hf"
    hits = sorted(root.glob(f"**/snapshots/*/{rel}"))
    if not hits:
        raise FileNotFoundError(
            f"{rel} is not in the local cache. Run: python tools/fetch_droid.py"
        )
    return hits[0]


@dataclass(frozen=True)
class EpisodeRef:
    """Where one episode lives inside the packed shard."""

    episode_index: int
    task: str
    length: int
    row_start: int
    row_stop: int
    data_file: int
    video: dict[str, tuple[int, float, float]]  # view -> (file_index, t_from, t_to)


def list_episodes(shard: int = 0, meta_file: int = 0) -> list[EpisodeRef]:
    """Episodes whose data *and* all three videos live entirely in `shard`.

    Restricting to one shard is what keeps the download to ~700 MB instead of the
    full multi-terabyte dataset. It still yields 27 real episodes.
    """
    path = _cached(f"{SPLIT}/meta/episodes/chunk-000/file-{meta_file:03d}.parquet")
    cols = ["episode_index", "tasks", "length", "data/file_index",
            "dataset_from_index", "dataset_to_index"]
    for v in VIEWS:
        for s in ("file_index", "from_timestamp", "to_timestamp"):
            cols.append(f"videos/{_VIDEO_KEY.format(view=v)}/{s}")
    df = pq.ParquetFile(path).read(columns=cols).to_pandas()

    keep = df["data/file_index"] == shard
    for v in VIEWS:
        keep &= df[f"videos/{_VIDEO_KEY.format(view=v)}/file_index"] == shard
    out: list[EpisodeRef] = []
    for _, r in df[keep].iterrows():
        tasks = list(r["tasks"])
        out.append(
            EpisodeRef(
                episode_index=int(r["episode_index"]),
                task=str(tasks[0]) if tasks else "",
                length=int(r["length"]),
                row_start=int(r["dataset_from_index"]),
                row_stop=int(r["dataset_to_index"]),
                data_file=shard,
                video={
                    v: (
                        int(r[f"videos/{_VIDEO_KEY.format(view=v)}/file_index"]),
                        float(r[f"videos/{_VIDEO_KEY.format(view=v)}/from_timestamp"]),
                        float(r[f"videos/{_VIDEO_KEY.format(view=v)}/to_timestamp"]),
                    )
                    for v in VIEWS
                },
            )
        )
    return sorted(out, key=lambda e: e.episode_index)


def _decode_window(path: Path, t_from: float, t_to: float, count: int) -> np.ndarray:
    """Decode exactly `count` RGB frames covering [t_from, t_to) from a packed mp4.

    Seeks to just before the window (AV1 needs to start from a keyframe), decodes
    forward, and keeps frames whose presentation time falls in the window. If the
    stream yields more than `count` — packing boundaries are not always exact — the
    first `count` are used, which is what the row range refers to.
    """
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        target = int(max(0.0, t_from - 0.2) / tb)
        container.seek(target, stream=stream, backward=True, any_frame=False)
        eps = 1e-4
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if t < t_from - eps:
                continue
            if t >= t_to - eps and len(frames) >= count:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    if len(frames) < count:
        raise RuntimeError(
            f"{path.name}: decoded {len(frames)} frames for window "
            f"[{t_from:.2f},{t_to:.2f}), expected {count}"
        )
    return np.stack(frames[:count])


def load_episode(ref: EpisodeRef) -> Episode:
    """Materialise one real episode: three camera streams, proprio, ground-truth actions."""
    data_path = _cached(f"{SPLIT}/data/chunk-000/file-{ref.data_file:03d}.parquet")
    cols = [
        "observation.state.joint_positions",
        "observation.state.gripper_position",
        "action.joint_position",
        "action.gripper_position",
        "frame_index",
        "episode_index",
    ]
    tbl = pq.ParquetFile(data_path).read(columns=cols)
    ep_idx = np.asarray(tbl.column("episode_index").to_pylist()).reshape(-1)
    sel = np.flatnonzero(ep_idx == ref.episode_index)
    if len(sel) == 0:
        raise RuntimeError(f"episode {ref.episode_index} not present in {data_path.name}")
    lo, hi = int(sel[0]), int(sel[-1]) + 1

    def col(name: str) -> np.ndarray:
        raw = tbl.column(name).to_pylist()[lo:hi]
        return np.asarray(raw, dtype=np.float32)

    joints = col("observation.state.joint_positions")
    grip = col("observation.state.gripper_position").reshape(-1, 1)
    act = np.concatenate(
        [col("action.joint_position"), col("action.gripper_position").reshape(-1, 1)], axis=1
    )
    n = len(joints)

    frames = {}
    for v in VIEWS:
        file_index, t_from, t_to = ref.video[v]
        vp = _cached(f"{SPLIT}/videos/{_VIDEO_KEY.format(view=v)}/chunk-000/file-{file_index:03d}.mp4")
        frames[v] = _decode_window(vp, t_from, t_to, n)

    return Episode(
        episode_id=f"cosmos3droid-{SPLIT}-ep{ref.episode_index}",
        source=f"{REPO}:{SPLIT}/data/chunk-000/file-{ref.data_file:03d}.parquet",
        task=ref.task,
        fps=15.0,
        frames=frames,
        joint_positions=joints,
        gripper_position=grip,
        actions=act,
        notes={"dataset": REPO, "split": SPLIT, "episode_index": ref.episode_index},
    )


def save_episode(ep: Episode, path: Path) -> Path:
    """Persist an episode as a single .npz plus a sidecar manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_positions=ep.joint_positions,
        gripper_position=ep.gripper_position,
        actions=ep.actions,
        **{f"frames__{k}": v for k, v in ep.frames.items()},
    )
    meta = {
        "episode_id": ep.episode_id,
        "source": ep.source,
        "task": ep.task,
        "fps": ep.fps,
        "length": len(ep),
        "frame_offset": ep.frame_offset,
        "notes": ep.notes,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_saved_episode(path: Path) -> Episode:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(path)
    frames = {k[len("frames__"):]: z[k] for k in z.files if k.startswith("frames__")}
    return Episode(
        episode_id=meta["episode_id"],
        source=meta["source"],
        task=meta["task"],
        fps=meta["fps"],
        frames=frames,
        joint_positions=z["joint_positions"],
        gripper_position=z["gripper_position"],
        actions=z["actions"],
        frame_offset=meta.get("frame_offset", 0),
        notes=meta.get("notes", {}),
    )
