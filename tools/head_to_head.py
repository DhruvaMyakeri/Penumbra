"""The thesis test: does a generative perturbation find what classical augmentation cannot?

PENUMBRA only earns its complexity if the answer is yes. A brightness curve and a blur
kernel are free, deterministic, and already in every robotics team's augmentation
pipeline; a streaming video model costs money, corrupts a camera, and needs a validity
gate. So the question is not "did the generative perturbation move the policy" - it is
"did it move the policy at a perceptual distance where the free alternative does not".

Design, and the two asymmetries it deliberately keeps:

  MATCHED DISTANCE   Each classical op is strength-searched until its perceptual
                     distance from the source matches the generative arm's. Comparing a
                     large classical change against a small generative one, or the
                     reverse, measures the size of the change, not its kind.

  DIFFERENT CONTROLS Generative scenarios are tested against the no-op re-render,
                     because X2 redraws every pixel and that redraw alone moves this
                     policy. Classical ops never pass through X2, so their control is
                     the raw baseline. Each arm is measured against the thing that
                     isolates *its* intervention. Using one control for both would
                     handicap whichever arm it did not belong to.

  SAME GATE          The classical arm faces the identical validity gate. Per-frame
                     independent noise is temporally incoherent in a way the recording
                     never was, and letting that count as a classical win while
                     rejecting the generative arm for the same defect would rig the
                     comparison.

    .venv/Scripts/python tools/head_to_head.py --from runs/garage-ep0-... --repeats 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import VIEWS  # noqa: E402
from penumbra.experiments.runner import ExperimentRunner  # noqa: E402
from penumbra.perturbation.classical import (  # noqa: E402
    CLASSICAL_OPS, POSITIVE_CONTROLS, ClassicalPerturbation,
)
from penumbra.perturbation.spec import FaultSpec, Rung  # noqa: E402
from penumbra.policy.cosmos_droid import CosmosDroidPolicy  # noqa: E402

log = logging.getLogger("penumbra.head_to_head")


def gate_fault(name: str) -> FaultSpec:
    """The same validity thresholds the generative arm was held to."""
    return FaultSpec(name=name, family="classical", model="classical/opencv",
                     description="matched-distance classical control arm",
                     ladder=(Rung(0.0, "", "untouched"), Rung(1.0, "", name)),
                     version="head-to-head-1",
                     validation={"max_geometry_drift": 0.0055,
                                 "min_structure_retained": 0.20})


def match_strength(episode, op: str, views, target: float, *, seed: int = 0,
                   tol: float = 0.004, iters: int = 12) -> tuple[float, float]:
    """Bisect this op's strength until its RMSE from the source matches `target`.

    Returns (strength, achieved_rmse). Perceptual distance is monotone in strength for
    every op here, which is what makes bisection legitimate; it is checked at the ends
    before searching, and an op that cannot reach the target says so rather than
    silently returning its maximum.
    """
    def rmse(s: float) -> float:
        r = ClassicalPerturbation(op, views=views, seed=seed).apply(episode, s)
        return float(r.distance["rmse"])

    lo, hi = 0.0, 1.0
    r_hi = rmse(hi)
    if r_hi < target - tol:
        return hi, r_hi  # cannot reach; the caller reports the shortfall
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rmse(mid)
        if abs(r - target) <= tol:
            return mid, r
        if r < target:
            lo = mid
        else:
            hi = mid
    return hi, rmse(hi)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_run", type=Path, required=True,
                    help="a finished garage run to take the target distance from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--baseline-runs", type=int, default=6)
    ap.add_argument("--rmse", type=float, default=None,
                    help="override the target perceptual distance")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of classical ops to run")
    ap.add_argument("--into", type=Path, default=None,
                    help="merge into an existing headtohead run instead of starting a "
                         "new one, keeping the ops it already measured")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("reactor_sdk").setLevel(logging.WARNING)

    state = json.loads((args.src_run / "state.json").read_text(encoding="utf-8"))
    views = tuple(state.get("views") or VIEWS)
    hits = [s for s in state["scenarios"] if s.get("is_vulnerability")]
    if not hits and args.rmse is None:
        raise SystemExit(
            f"{args.src_run.name} found no vulnerabilities, so there is no distance to "
            f"match. Pass --rmse to run the comparison anyway."
        )
    distances = [s["distance"]["rmse"] for s in hits if s.get("distance")]
    target = args.rmse if args.rmse is not None else float(statistics.median(distances))

    # Resuming: the surviving ops were strength-matched to the *old* target, so adopting
    # it is not a convenience — mixing two target distances in one table would make the
    # arms incomparable, which is the one thing this comparison exists to avoid.
    prior: list[dict] = []
    if args.into:
        existing = json.loads((args.into / "head_to_head.json").read_text(encoding="utf-8"))
        prior = existing.get("classical") or []
        if abs(existing["target_rmse"] - target) > 1e-6:
            log.warning("adopting the existing run's target rmse %.5f (this invocation "
                        "would have used %.5f) so all ops share one distance",
                        existing["target_rmse"], target)
        target = float(existing["target_rmse"])

    print(f"\ngenerative arm ({args.src_run.name}), {len(hits)} confirmed:")
    for s in hits:
        c = s.get("vs_control") or {}
        print(f"  {s['name']:<32} rmse={s['distance']['rmse']:.4f}  "
              f"p={c.get('p_value'):.4f}  d={c.get('effect_size_cohens_d'):+.2f}")
    print(f"\ntarget perceptual distance (median): rmse = {target:.4f}")
    print(f"cameras: {', '.join(views)}\n")

    episode = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)

    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    await runner.calibrate(episode, runs=args.baseline_runs)

    out = args.into or (
        RUNS_DIR / f"headtohead-ep{args.episode}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    ops = [o for o in CLASSICAL_OPS if o not in POSITIVE_CONTROLS]
    if args.only:
        wanted = [o.strip() for o in args.only.split(",") if o.strip()]
        unknown = [o for o in wanted if o not in ops]
        if unknown:
            raise SystemExit(f"unknown classical op(s): {', '.join(unknown)}. "
                             f"available: {', '.join(ops)}")
        ops = wanted
    # Whatever this invocation measures replaces the prior record for the same op; the
    # rest carry through, so the verdict is always computed over the full table.
    records: list[dict] = [r for r in prior if r["op"] not in ops]
    report = {
        "source_run": args.src_run.name,
        "episode": episode.episode_id,
        "views": list(views),
        "target_rmse": round(target, 5),
        "generative": [
            {"name": s["name"], "rmse": s["distance"]["rmse"],
             "p_value": (s.get("vs_control") or {}).get("p_value"),
             "effect_size": (s.get("vs_control") or {}).get("effect_size_cohens_d"),
             "gripper_p": (s.get("vs_control") or {}).get("gripper_p_value"),
             "at_resolution_floor": (s.get("vs_control") or {}).get("at_resolution_floor")}
            for s in hits
        ],
        "classical": records,
    }

    def flush():
        (out / "head_to_head.json").write_text(json.dumps(report, indent=2, default=str),
                                               encoding="utf-8")

    flush()
    for op in ops:
        strength, achieved = match_strength(episode, op, views, target)
        shortfall = achieved < target - 0.01
        log.info("%s: strength %.4f -> rmse %.4f%s", op, strength, achieved,
                 "  (cannot reach the target)" if shortfall else "")
        ev = await runner.evaluate_classical(
            episode, op, strength, fault=gate_fault(op), views=views,
            repeats=args.repeats, run_id="h2h",
        )
        g = ev.group
        rec = {
            "op": op,
            "strength": round(strength, 5),
            "rmse": round(achieved, 5),
            "matched": not shortfall,
            "gate": ev.validation.to_dict(),
            "rejected": ev.rejected,
            "rollouts": len(ev.traces),
            "p_value": g.p_value if g else None,
            "effect_size": g.effect_size if g else None,
            "gripper_p": g.gripper_p_value if g else None,
            "significant": bool(g.any_significant) if g else None,
            "at_resolution_floor": bool(g.at_resolution_floor) if g else None,
        }
        records.append(rec)
        flush()

    # -- verdict ---------------------------------------------------------------
    order = list(CLASSICAL_OPS)
    records.sort(key=lambda r: order.index(r["op"]) if r["op"] in order else 99)
    admissible = [r for r in records if not r["rejected"] and r["matched"]]
    classical_wins = [r for r in admissible if r["significant"]]
    gen_effects = [g["effect_size"] for g in report["generative"] if g["effect_size"]]
    cls_effects = [r["effect_size"] for r in admissible if r["effect_size"] is not None]

    if not admissible:
        verdict = ("INCONCLUSIVE - no classical op reached the matched distance while "
                   "passing the same validity gate, so the arms were never comparable.")
    elif not classical_wins:
        verdict = (
            f"GENERATIVE ADVANTAGE, on this episode. {len(admissible)} classical ops "
            f"reached rmse {target:.4f} and passed the gate; none separated from the "
            f"baseline, while {len(hits)} generative situations separated from their "
            f"control. The distinction is in the KIND of change, not its size."
        )
    else:
        # Two questions, and conflating them would misreport the result in whichever
        # direction happened to be convenient. (a) Does a free operator reach
        # significance at the same perceptual distance? If yes, "generative finds what
        # classical cannot" is falsified as stated, full stop. (b) How large is the
        # effect it reaches? A significant d = 0.2 and a significant d = 2.0 are not the
        # same discovery. Both go in the verdict.
        #
        # The effect sizes are only roughly comparable: each arm is standardised by the
        # spread of its own control, and the controls differ by design (see the header).
        gm = float(np.median(gen_effects)) if gen_effects else float("nan")
        cm = max(r["effect_size"] for r in classical_wins)
        verdict = (
            f"NO GENERATIVE ADVANTAGE DEMONSTRATED AT THE SIGNIFICANCE LEVEL. "
            f"{len(classical_wins)} of {len(admissible)} matched classical ops also "
            f"moved the policy ({', '.join(r['op'] for r in classical_wins)}), so the "
            f"claim that generative perturbation finds what classical augmentation "
            f"cannot is FALSIFIED as stated on this episode. Magnitude is a separate "
            f"question and the two answers differ: the strongest admissible classical "
            f"effect is d = {cm:+.2f} against a generative median of d = {gm:+.2f}. "
            f"Effect sizes are standardised against each arm's own control and so are "
            f"comparable only roughly. What survives is a weaker claim - larger "
            f"behavioural effect at equal perceptual distance - not a unique one."
        )
    report["verdict"] = verdict
    report["summary"] = {
        "classical_tested": len(records),
        "classical_admissible": len(admissible),
        "classical_significant": len(classical_wins),
        "generative_confirmed": len(hits),
        "median_effect_generative": round(float(np.median(gen_effects)), 3) if gen_effects else None,
        "median_effect_classical": round(float(np.median(cls_effects)), 3) if cls_effects else None,
    }
    report["cost"] = runner.cost
    flush()

    print("\n" + "=" * 78)
    print(f"{'op':<22}{'strength':<11}{'rmse':<9}{'gate':<11}{'p':<10}{'d':<9}moved?")
    print("-" * 78)
    for r in records:
        gate_txt = "REJECTED" if r["rejected"] else "valid"
        p_txt = "-" if r["p_value"] is None else format(r["p_value"], ".4f")
        d_txt = "-" if r["effect_size"] is None else format(r["effect_size"], "+.2f")
        moved = "YES" if r["significant"] else "no"
        print(f"{r['op']:<22}{r['strength']:<11.4f}{r['rmse']:<9.4f}"
              f"{gate_txt:<11}{p_txt:<10}{d_txt:<9}{moved}")
    print("-" * 78)
    print(verdict)
    print(f"\ncost: ${report['cost']['estimated_usd']}")
    print(f"written {out / 'head_to_head.json'}")


if __name__ == "__main__":
    asyncio.run(main())
