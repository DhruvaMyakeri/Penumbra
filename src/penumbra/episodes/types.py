"""Episode types.

An `Episode` is a real robot recording: the physics, geometry, camera trajectory,
ground-truth actions and language instruction all come from reality. PENUMBRA never
generates any of it. Only `frames["exterior_image_1_left"]` is ever perturbed, and
even then the perturbation is a separate object that produces a *new* Episode rather
than mutating this one.

`ObservationSource` is the seam a live robot camera would slot into later: anything
that can yield per-step views + proprio + a task string can drive the policy runner,
so replacing `DroidEpisode` with `LiveCamera` needs no change downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

#: The three camera views the DROID rig records, and the Reactor policy consumes.
#: Keys are dataset-side names; `POLICY_TRACKS` maps them to Reactor track names.
VIEWS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")

#: dataset view name -> `reactor/cosmos-nano-policy-droid` sendonly track name
POLICY_TRACKS = {
    "exterior_image_1_left": "exterior_view_1",
    "exterior_image_2_left": "exterior_view_2",
    "wrist_image_left": "wrist_view",
}

#: The view PENUMBRA perturbs. The other two, and proprio, are held fixed so the
#: intervention is single-variable.
PERTURBED_VIEW = "exterior_image_1_left"


@dataclass
class Episode:
    """One real robot episode, fully in memory.

    Attributes:
        episode_id: Stable identifier, e.g. ``cosmos3droid-success-ep0``.
        source: Provenance string (dataset repo + file), for the experiment record.
        task: Natural-language instruction, verbatim from the recording.
        fps: Control/recording rate.
        frames: view name -> ``(T, H, W, 3)`` uint8 RGB.
        joint_positions: ``(T, 7)`` float32, recorded proprioception.
        gripper_position: ``(T, 1)`` float32, recorded proprioception.
        actions: ``(T, 8)`` float32 ground-truth action = 7 joint positions + gripper,
            the same convention the policy predicts.
        frame_offset: Index into the original episode this window starts at.
    """

    episode_id: str
    source: str
    task: str
    fps: float
    frames: dict[str, np.ndarray]
    joint_positions: np.ndarray
    gripper_position: np.ndarray
    actions: np.ndarray
    frame_offset: int = 0
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {k: len(v) for k, v in self.frames.items()}
        lengths["joint_positions"] = len(self.joint_positions)
        lengths["gripper_position"] = len(self.gripper_position)
        lengths["actions"] = len(self.actions)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode {self.episode_id}: length mismatch {lengths}")

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def height(self) -> int:
        return int(self.frames[PERTURBED_VIEW].shape[1])

    @property
    def width(self) -> int:
        return int(self.frames[PERTURBED_VIEW].shape[2])

    def proprio_at(self, t: int) -> dict:
        """Proprio payload in the exact shape `set_proprio_json` documents."""
        return {
            "joint_position": [[float(x) for x in self.joint_positions[t]]],
            "gripper_position": [[float(self.gripper_position[t][0])]],
        }

    def window(self, start: int, length: int) -> Episode:
        """A contiguous sub-episode. Used to keep experiments short and cheap."""
        stop = min(start + length, len(self))
        return Episode(
            episode_id=f"{self.episode_id}@{start}+{stop - start}",
            source=self.source,
            task=self.task,
            fps=self.fps,
            frames={k: v[start:stop] for k, v in self.frames.items()},
            joint_positions=self.joint_positions[start:stop],
            gripper_position=self.gripper_position[start:stop],
            actions=self.actions[start:stop],
            frame_offset=self.frame_offset + start,
            notes=dict(self.notes),
        )

    def with_view(self, view: str, frames: np.ndarray, episode_id: str) -> Episode:
        """A copy with one camera view replaced. Everything else is shared, by value.

        This is how a perturbed episode is built: reality unchanged, one stream of
        pixels swapped.
        """
        return self.with_views({view: frames}, episode_id)

    def with_views(self, replacements: dict[str, np.ndarray], episode_id: str) -> Episode:
        """A copy with one *or more* camera views replaced.

        Single-view interventions can be compensated for: this policy reads three
        cameras plus proprioception, so blanking one leaves two clean views and the
        arm state intact. Perturbing several at once is a different intervention, not
        a bigger one, and it needs to be expressible.

        Proprioception and the ground-truth actions are still shared by reference, so
        even an all-view perturbation changes only pixels.
        """
        new_frames = dict(self.frames)
        for view, frames in replacements.items():
            if view not in new_frames:
                raise ValueError(
                    f"unknown view {view!r}; episode has {sorted(new_frames)}"
                )
            if len(frames) != len(self):
                raise ValueError(
                    f"replacement view {view!r} has {len(frames)} frames, "
                    f"episode has {len(self)}"
                )
            new_frames[view] = frames
        return Episode(
            episode_id=episode_id,
            source=self.source,
            task=self.task,
            fps=self.fps,
            frames=new_frames,
            joint_positions=self.joint_positions,
            gripper_position=self.gripper_position,
            actions=self.actions,
            frame_offset=self.frame_offset,
            notes=dict(self.notes),
        )


@runtime_checkable
class ObservationSource(Protocol):
    """What the policy runner needs. A recording satisfies it; so would a live rig."""

    task: str
    fps: float

    def __len__(self) -> int: ...

    def views_at(self, t: int) -> dict[str, np.ndarray]: ...

    def proprio_at(self, t: int) -> dict: ...


class EpisodeSource:
    """Adapts an `Episode` to `ObservationSource`."""

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.task = episode.task
        self.fps = episode.fps

    def __len__(self) -> int:
        return len(self.episode)

    def views_at(self, t: int) -> dict[str, np.ndarray]:
        return {k: v[t] for k, v in self.episode.frames.items()}

    def proprio_at(self, t: int) -> dict:
        return self.episode.proprio_at(t)

    def steps(self) -> Iterator[int]:
        return iter(range(len(self)))
