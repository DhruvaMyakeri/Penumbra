"""Measure the policy's run-to-run variability on the UNPERTURBED episode."""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import RUNS_DIR
from penumbra.episodes.droid import list_episodes, load_episode
from penumbra.policy.cosmos_droid import CosmosDroidPolicy
from penumbra.evaluation.divergence import compute_noise_floor, divergence

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    ref = [r for r in list_episodes() if r.episode_index == 0][0]
    ep = load_episode(ref).window(a.start, a.frames)
    pol = CosmosDroidPolicy(chunk=a.chunk)
    traces = []
    for i in range(a.runs):
        t0 = time.time()
        tr = await pol.rollout(ep, run_id=f"nf{i}")
        traces.append(tr)
        print(f"run {i}: {len(tr)} steps in {time.time()-t0:.1f}s, mean latency {tr.latencies.mean():.3f}s", flush=True)
    nf = compute_noise_floor(traces)
    print("\nPAIRWISE:")
    for s in nf.samples:
        print("  ", s["pair"], "joint_l2", s["joint_l2"], "grip_l1", s["gripper_l1"], "flips", s["gripper_flips"])
    print("\nNOISE FLOOR:", json.dumps(nf.to_dict(), indent=2))
    out = RUNS_DIR/"_probe"/"noise_floor"; out.mkdir(parents=True, exist_ok=True)
    np.save(out/"traces.npy", np.stack([t.actions for t in traces]))
    (out/"noise_floor.json").write_text(json.dumps({"noise_floor": nf.to_dict(), "samples": nf.samples,
        "episode": ep.episode_id, "frames": a.frames, "chunk": a.chunk}, indent=2), encoding="utf-8")
    print("saved ->", out)

asyncio.run(main())
