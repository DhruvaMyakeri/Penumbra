"""Smoke test: run the real DROID episode through reactor/cosmos-nano-policy-droid."""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.config import RUNS_DIR
from penumbra.episodes.droid import list_episodes, load_episode
from penumbra.policy.cosmos_droid import CosmosDroidPolicy

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--start", type=int, default=120)
    args = ap.parse_args()
    ref = [r for r in list_episodes() if r.episode_index == 0][0]
    ep = load_episode(ref).window(args.start, args.frames)
    print(f"episode {ep.episode_id} len={len(ep)} task={ep.task.split('|')[0].strip()!r}")
    pol = CosmosDroidPolicy(chunk=args.chunk)
    t0 = time.time()
    tr = await pol.rollout(ep, run_id="smoke")
    dt = time.time() - t0
    print(f"\nrollout ok in {dt:.1f}s")
    print("actions", tr.actions.shape, "steps", len(tr), "horizon", tr.horizon, "dof", tr.dof)
    print("mean latency", float(tr.latencies.mean()).__round__(3), "s")
    print("step indices", tr.step_indices.tolist())
    print("\npredicted first-action (step 0):", tr.actions[0,0].round(4))
    print("logged  action    (frame 0):", ep.actions[0].round(4))
    print("predicted first-action (step 1):", tr.actions[1,0].round(4))
    print("logged  action    (frame %d):"%tr.step_indices[1], ep.actions[tr.step_indices[1]].round(4))
    gt = np.stack([ep.actions[i] for i in tr.step_indices])
    err = np.abs(tr.first_action - gt).mean(axis=0)
    print("\nmean |pred - logged| per DoF:", err.round(4))
    print("meta:", json.dumps(tr.meta, default=str)[:500])
    out = RUNS_DIR/"_probe"/"policy_smoke"; out.mkdir(parents=True, exist_ok=True)
    np.save(out/"actions.npy", tr.actions)
    (out/"trace.json").write_text(json.dumps(tr.to_dict(), indent=2, default=str), encoding="utf-8")
    print("saved ->", out)

asyncio.run(main())
