"""Positive control: does the policy use the perturbed camera view at all?

Every null PENUMBRA has produced shares one confound. The policy consumes three camera
views; PENUMBRA perturbs one of them. If the policy barely reads
`exterior_image_1_left`, then no perturbation of it can move behaviour, and every
"robust" verdict is a fact about the wiring rather than about the policy.

This destroys the view completely - fades it to black - and asks the same question with
the same group test. Interpretation:

  significant  -> the channel carries signal; nulls on it are informative
  null         -> the apparatus has no sensitivity here, and NO null measured on this
                  view means anything until that is fixed

Reuses the baseline group across both conditions, so the marginal cost is the perturbed
rollouts only.

    python tools/positive_control.py --runs 6 --repeats 6
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


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--strengths", type=float, nargs="*", default=[1.0, 0.6])
    args = ap.parse_args()

    ep = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))

    print(f"episode {ep.episode_id}  ({len(ep)} frames)")
    print(f"calibrating noise floor over {args.runs} unperturbed runs...", flush=True)
    nf = await runner.calibrate(ep, runs=args.runs)
    print(f"  mean={nf.joint_l2_mean:.5f}  sigma={nf.joint_l2_std:.5f}", flush=True)

    out = RUNS_DIR / "_probe"
    out.mkdir(parents=True, exist_ok=True)
    # Named for the strengths it measures. A fixed filename silently overwrote the
    # previous ladder the first time this ran twice.
    tag = "_".join(f"{s:g}" for s in args.strengths)
    path = out / f"positive_control_blackout_{tag}.json"

    def save(results: list, sensitive: bool | None, verdict: str) -> None:
        """Write after every rung, not only at the end.

        A killed run previously discarded two completed rungs - twelve policy rollouts
        of real GPU time - because the record was only written once the whole ladder
        finished. Each rung costs minutes and money; none of them should depend on the
        next one surviving.
        """
        path.write_text(json.dumps({
            "purpose": "positive control - does the policy use exterior_image_1_left?",
            "episode": ep.episode_id,
            "noise_floor": nf.to_dict(),
            "results": results,
            "complete": len(results) == len(args.strengths),
            "strengths_requested": args.strengths,
            "channel_sensitive": sensitive,
            "verdict": verdict,
            "cost": runner.cost,
        }, indent=2, default=str), encoding="utf-8")

    results: list = []
    save(results, None, "in progress")
    for strength in args.strengths:
        label = "fully black" if strength >= 1.0 else f"{strength:.0%} darkened"
        print(f"\n[positive control] blackout @ {strength} ({label})", flush=True)
        ev = await runner.evaluate_classical(
            ep, "blackout", strength, repeats=args.repeats, run_id=f"pc{strength:g}"
        )
        g = ev.group
        print(f"  perceptual: rmse={ev.distance['rmse']:.4f} ssim={ev.distance['ssim']:.3f}")
        print(f"  validity:   {ev.validation.status}  drift={ev.validation.geometry_drift:.5f}  "
              f"structure={ev.validation.structure_retained:.3f}")
        if g:
            print(f"  group test: within={g.within_mean:.4f}  between={g.between_mean:.4f}")
            print(f"              p={g.p_value:.4f} (floor {g.min_achievable_p:.4f})  "
                  f"d={g.effect_size:+.2f}")
            print(f"              gripper flips {g.gripper_flip_rate_baseline:.3f} -> "
                  f"{g.gripper_flip_rate_between:.3f}")
        print(f"  => {'SENSITIVE' if ev.broke else 'NO DETECTABLE EFFECT'}")
        results.append({"strength": strength, "label": label, **ev.summary()})
        save(results, None, "in progress")

    # Only the full-blackout rung can settle whether the channel carries signal at
    # all. A partial ladder that never ran it must NOT print a verdict about channel
    # sensitivity - an earlier version did exactly that, announcing "the apparatus has
    # no measurable sensitivity" from a run whose largest rung was 0.5, which the run
    # had no way to know.
    full = next((r for r in results if r["strength"] >= 1.0), None)
    if full is None:
        sensitive = None
        rungs = ", ".join(f"{r['strength']:g}" for r in results)
        verdict = (
            f"PARTIAL LADDER - this run measured blackout rung(s) {rungs} only and did "
            f"not include the full-blackout rung, so it says nothing about whether the "
            f"channel carries signal. It contributes bracket points to the ruler. Run "
            f"tools/reach_analysis.py for the aggregate."
        )
    else:
        sensitive = full["status"] == "BREAK"
        verdict = (
            "The perturbed view carries signal the policy uses. Nulls measured on this "
            "view are informative."
            if sensitive else
            "Destroying the view entirely did NOT shift the action distribution. The "
            "apparatus has no measurable sensitivity on this channel, and no null result "
            "measured on this view can be interpreted as policy robustness."
        )
    save(results, sensitive, verdict)
    print(f"\nVERDICT: {verdict}")
    print(f"cost: {json.dumps(runner.cost)}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
