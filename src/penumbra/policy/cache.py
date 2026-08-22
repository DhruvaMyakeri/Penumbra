"""Persisted baseline rollouts.

Every experiment needs a baseline group, and re-measuring it costs the same GPU time
every time — six rollouts is ~2.5 minutes and roughly half the cost of a condition.
Worse, it made single experiments long enough to be terminated mid-run by the
execution environment, which is how twelve completed rollouts were lost.

So baselines are cached on disk, keyed by everything that could change them:

    episode id · policy · action chunk · warm-up frames · episode length

Anything that alters the observation the policy sees, or the protocol it sees it
under, changes the key. The cached traces are the *unperturbed* condition, so reusing
them across conditions is not a shortcut — it is the same control group, which is
also what makes conditions directly comparable to each other rather than each to its
own separately-drawn baseline.

The one thing this must never do is silently reuse a baseline from a different
setup. The key is explicit and the record carries `baseline_from_cache` so a reader
can see it happened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR
from .base import PolicyTrace

CACHE_DIR = RUNS_DIR / "_baseline_cache"


def baseline_key(
    *, episode_id: str, policy: str, chunk: int, warmup: int, length: int
) -> str:
    raw = f"{episode_id}|{policy}|chunk={chunk}|warmup={warmup}|len={length}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{digest}"


def save_traces(traces: list[PolicyTrace], key: str, meta: dict) -> Path:
    """Append-safe write: a later run with more repeats extends the group."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.npz"
    existing = load_traces(key)
    combined = existing + [t for t in traces if t.run_id not in {e.run_id for e in existing}]
    payload = {}
    for i, t in enumerate(combined):
        payload[f"actions_{i}"] = t.actions
        payload[f"steps_{i}"] = t.step_indices
        payload[f"lat_{i}"] = t.latencies
    np.savez_compressed(path, n=np.asarray(len(combined)), **payload)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(
            {
                **meta,
                "n_traces": len(combined),
                "run_ids": [t.run_id for t in combined],
                "policy": combined[0].policy if combined else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def load_traces(key: str, limit: int | None = None) -> list[PolicyTrace]:
    """Cached unperturbed rollouts, or an empty list when there are none."""
    path = CACHE_DIR / f"{key}.npz"
    meta_path = CACHE_DIR / f"{key}.json"
    if not path.exists() or not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    z = np.load(path)
    n = int(z["n"])
    out: list[PolicyTrace] = []
    run_ids = meta.get("run_ids", [])
    for i in range(n):
        out.append(
            PolicyTrace(
                policy=meta.get("policy", "cached"),
                actions=z[f"actions_{i}"],
                step_indices=z[f"steps_{i}"],
                latencies=z[f"lat_{i}"],
                run_id=run_ids[i] if i < len(run_ids) else f"cached{i}",
                meta={"from_cache": True, "cache_key": key},
            )
        )
    return out[:limit] if limit is not None else out


def describe(key: str) -> dict | None:
    meta_path = CACHE_DIR / f"{key}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))
