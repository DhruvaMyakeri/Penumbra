"""Which observation channels does this policy actually use?

PENUMBRA has been perturbing `exterior_image_1_left` because that is the view a human
would call "the scene camera". That was an assumption. The policy reads three cameras
plus proprioception, and if the information it acts on lives somewhere else, then
every perturbation of that one view was aimed at the wrong channel.

This ablates by destruction: black out each subset of views and measure whether the
action distribution moves. Blackout is used rather than a subtle fault because the
question here is not "is the policy robust" but "is this channel load-bearing at all" -
and for that, the maximal intervention is the right instrument.

    camera 1 only        exterior_image_1_left
    camera 2 only        exterior_image_2_left
    wrist only           wrist_image_left
    1 + 2                both exteriors
    1 + wrist
    2 + wrist
    all three

Each condition is compared against the SAME cached baseline group, so conditions are
comparable to one another and not merely each to its own draw.

Runs one subset per invocation by default, because the execution environment
terminates long jobs; the results accumulate in a single JSON that is rewritten after
every condition.

    python tools/channel_ablation.py --sets all
    python tools/channel_ablation.py --sets ext1 ext2 wrist
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.experiments.runner import ExperimentRunner  # noqa: E402
from penumbra.policy.cosmos_droid import CosmosDroidPolicy  # noqa: E402

EXT1 = "exterior_image_1_left"
EXT2 = "exterior_image_2_left"
WRIST = "wrist_image_left"

SETS: dict[str, tuple[str, ...]] = {
    "ext1": (EXT1,),
    "ext2": (EXT2,),
    "wrist": (WRIST,),
    "ext1+ext2": (EXT1, EXT2),
    "ext1+wrist": (EXT1, WRIST),
    "ext2+wrist": (EXT2, WRIST),
    "all": (EXT1, EXT2, WRIST),
}

OUT = RUNS_DIR / "_probe" / "channel_ablation.json"


def load_existing() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"purpose": "which observation channels does the policy use?", "conditions": {}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["all"],
                    help=f"subsets to ablate; one of {sorted(SETS)}")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()

    for name in args.sets:
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r}; available: {sorted(SETS)}")

    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))

    print(f"episode {ep.episode_id} ({len(ep)} frames)")
    nf = await runner.calibrate(ep, runs=args.runs)
    print(f"baseline: mean={nf.joint_l2_mean:.5f} sigma={nf.joint_l2_std:.5f} "
          f"({runner.cost['baseline_rollouts_from_cache']}/{args.runs} from cache)", flush=True)

    record = load_existing()
    record["episode"] = ep.episode_id
    record["noise_floor"] = nf.to_dict()

    for name in args.sets:
        views = SETS[name]
        print(f"\n[ablation] blackout {name}  ({len(views)} view(s))", flush=True)
        ev = await runner.evaluate_classical(
            ep, "blackout", 1.0, views=views, repeats=args.repeats, run_id=f"abl-{name}"
        )
        g = ev.group
        row = {
            "views": list(views),
            "n_views": len(views),
            "rmse_mean_over_perturbed_views": ev.distance["rmse"],
            "joint_p": g.p_value if g else None,
            "gripper_p": g.gripper_p_value if g else None,
            "effect_size": g.effect_size if g else None,
            "within_group": g.within_mean if g else None,
            "between_group": g.between_mean if g else None,
            "significant": bool(ev.broke),
            "gripper_flip_within": g.gripper_flip_rate_baseline if g else None,
            "gripper_flip_between": g.gripper_flip_rate_between if g else None,
        }
        record["conditions"][name] = row
        record["cost"] = runner.cost
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

        print(f"  joint   p={row['joint_p']:.4f}  d={row['effect_size']:+.2f}")
        print(f"  gripper p={row['gripper_p']:.4f}  "
              f"flips {row['gripper_flip_within']:.3f} -> {row['gripper_flip_between']:.3f}")
        print(f"  => {'LOAD-BEARING' if row['significant'] else 'no detectable effect'}",
              flush=True)

    print("\n" + "=" * 66)
    print(f"{'condition':<14} {'views':<6} {'joint p':<9} {'grip p':<9} {'d':<7} verdict")
    print("=" * 66)
    for name, row in sorted(record["conditions"].items(), key=lambda kv: kv[1]["n_views"]):
        v = "LOAD-BEARING" if row["significant"] else "-"
        print(f"{name:<14} {row['n_views']:<6} {row['joint_p']:<9.4f} "
              f"{row['gripper_p']:<9.4f} {row['effect_size']:<+7.2f} {v}")
    print(f"\ncost: {json.dumps(runner.cost)}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
