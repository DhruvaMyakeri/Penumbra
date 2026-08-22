"""Policy interface.

PENUMBRA measures a policy's *behaviour*, so the only thing it needs from one is a
trace of what it did. Everything downstream — divergence, noise floor, search,
reporting — works against `PolicyTrace` and never against a specific model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PolicyTrace:
    """What a policy did over one episode.

    Attributes:
        policy: Model identifier, e.g. ``reactor/cosmos-nano-policy-droid``.
        actions: ``(S, H, D)`` float32 — S control steps, H prediction horizon, D DoF.
            Kept at full horizon rather than collapsed, so timing shifts inside the
            chunk stay measurable.
        step_indices: ``(S,)`` frame index each prediction was conditioned on.
        latencies: per-step seconds from proprio-sent to prediction-received.
        run_id: distinguishes repeat runs of an identical condition (noise floor).
        meta: free-form provenance for the experiment record.
    """

    policy: str
    actions: np.ndarray
    step_indices: np.ndarray
    latencies: np.ndarray
    run_id: str = "0"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def dof(self) -> int:
        return int(self.actions.shape[2])

    @property
    def first_action(self) -> np.ndarray:
        """``(S, D)`` — the action that would actually execute at each step."""
        return self.actions[:, 0, :]

    @property
    def gripper(self) -> np.ndarray:
        """``(S, H)`` — the gripper channel, which is the last DoF in DROID order."""
        return self.actions[:, :, -1]

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "run_id": self.run_id,
            "steps": len(self),
            "horizon": self.horizon,
            "dof": self.dof,
            "mean_latency_s": float(np.mean(self.latencies)) if len(self.latencies) else None,
            "meta": self.meta,
        }


class Policy(Protocol):
    """Anything PENUMBRA can put under test."""

    name: str

    async def rollout(self, episode, *, run_id: str = "0") -> PolicyTrace: ...
