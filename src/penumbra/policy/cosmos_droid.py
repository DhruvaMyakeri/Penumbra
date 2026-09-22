"""`reactor/cosmos-nano-policy-droid` as a PENUMBRA `Policy`.

Cosmos3-Nano-Policy-DROID: three camera views plus proprioception in, an
``action_prediction`` chunk of shape ``[32][8]`` out (7 joint positions + gripper,
DROID joint-position convention). Verified against the live schema — see
docs/REACTOR_CAPABILITIES.md.

**Replay protocol (teacher-forced, open-loop).**

The policy's predicted action never moves the robot, because the robot is a
recording. On every control step PENUMBRA echoes back the *logged* action chunk, not
the predicted one, via ``set_executed_step_json``. That keeps the conditioning
history pinned to the real trajectory, so the baseline run and the perturbed run
differ in exactly one thing: the pixels of ``exterior_view_1``.

That is deliberate, and it is what makes this a controlled experiment rather than a
divergent simulation. It also means the measurement is *action divergence under
intervention*, not task success — see docs/METHODOLOGY.md for what that does and does
not license us to claim.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from reactor_sdk import time_micros

from ..episodes.types import POLICY_TRACKS, Episode
from ..reactor.session import ReactorSession
from .base import PolicyTrace

log = logging.getLogger("penumbra.policy")

MODEL = "reactor/cosmos-nano-policy-droid"
#: Documented action shape for this checkpoint.
HORIZON, DOF = 32, 8


class CosmosDroidPolicy:
    """Runs one episode through the policy and returns its action trace.

    Args:
        chunk: how many logged actions to report as "executed" per control step, and
            therefore how far the frame pointer advances. Must be <= HORIZON.
        step_timeout: seconds to wait for one ``action_prediction`` before giving up.
        warmup_frames: frames pushed before the first prediction is requested, so the
            model has a short observation history rather than a single frame.
    """

    name = MODEL

    def __init__(
        self,
        *,
        chunk: int = 8,
        step_timeout: float = 45.0,
        warmup_frames: int = 4,
        max_steps: int | None = None,
    ) -> None:
        if chunk > HORIZON:
            raise ValueError(f"chunk {chunk} exceeds policy horizon {HORIZON}")
        self.chunk = chunk
        self.step_timeout = step_timeout
        self.warmup_frames = warmup_frames
        self.max_steps = max_steps

    async def rollout(self, episode: Episode, *, run_id: str = "0") -> PolicyTrace:
        predictions: asyncio.Queue = asyncio.Queue()
        task_text = episode.task.split("|")[0].strip()[:300]

        actions: list[np.ndarray] = []
        step_indices: list[int] = []
        latencies: list[float] = []
        rejected: list[dict] = []

        async with ReactorSession(MODEL) as s:

            def _on_event(msg: dict) -> None:
                kind = msg.get("type")
                if kind == "action_prediction":
                    predictions.put_nowait(msg)
                elif kind == "command_error":
                    rejected.append(msg)

            s.on_event(_on_event)

            tracks = {}
            for view, track_name in POLICY_TRACKS.items():
                tracks[view] = await s.publish(track_name)
            await s.command("reset", {})
            await s.command("set_task_description", {"task_description": task_text})

            def push(t: int) -> None:
                now = time_micros()
                for view, track in tracks.items():
                    track.push_frame(
                        np.ascontiguousarray(episode.frames[view][t]), capture_time_us=now
                    )

            # Warm-up: give the model a short observation history before asking.
            for t in range(min(self.warmup_frames, len(episode))):
                push(t)
                await asyncio.sleep(1.0 / episode.fps)

            t = 0
            step = 0
            while t < len(episode):
                if self.max_steps is not None and step >= self.max_steps:
                    break
                push(t)
                await s.command("set_proprio_json", {"proprio_json": json.dumps(episode.proprio_at(t))})
                sent = time.time()
                try:
                    msg = await asyncio.wait_for(predictions.get(), timeout=self.step_timeout)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"{MODEL}: no action_prediction within {self.step_timeout}s at step {step} "
                        f"(frame {t}). Rejected commands: {rejected}"
                    ) from exc
                latencies.append(time.time() - sent)

                data = msg.get("data", msg)
                arr = np.asarray(data.get("action"), dtype=np.float32)
                if arr.ndim != 2:
                    raise RuntimeError(f"{MODEL}: unexpected action shape {arr.shape}")
                actions.append(arr)
                step_indices.append(t)

                # Echo the *logged* action chunk, so conditioning stays on reality.
                executed = episode.actions[t : t + self.chunk]
                await s.command(
                    "set_executed_step_json",
                    {
                        "executed_step_json": json.dumps(
                            {"step": step + 1, "action": executed.tolist()}
                        )
                    },
                )
                t += self.chunk
                step += 1

            session_log = s.log

        if not actions:
            raise RuntimeError(f"{MODEL}: produced no predictions for {episode.episode_id}")

        horizon = min(a.shape[0] for a in actions)
        stacked = np.stack([a[:horizon] for a in actions]).astype(np.float32)
        return PolicyTrace(
            policy=MODEL,
            actions=stacked,
            step_indices=np.asarray(step_indices, dtype=np.int32),
            latencies=np.asarray(latencies, dtype=np.float32),
            run_id=run_id,
            meta={
                "episode_id": episode.episode_id,
                "task": task_text,
                "chunk": self.chunk,
                "warmup_frames": self.warmup_frames,
                "session_id": session_log.session_id,
                "connect_seconds": session_log.connect_seconds,
                "rejected_commands": rejected,
                "errors": session_log.errors,
            },
        )
