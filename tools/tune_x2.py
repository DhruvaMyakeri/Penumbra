"""Which streaming configuration makes X2 actually apply the edit? Measure it.

A suite came back with 7 of 18 renders showing no visible change at all - the model
declined. A declined render is as useless as a hallucinated one: it burns a render, it
counts toward the suite's multiplicity correction, and it teaches nothing about the
policy. Three knobs plausibly govern this and guessing between them is what produced
the failed run in the first place:

  LEAD          copies of the first frame pushed before the clip. Warm-up on a static
                image - the model has no motion to settle against.
  PRIME PASSES  the real clip streamed once and discarded before the pass that counts,
                so the edit is already committed when the kept frames arrive.
  PUSH FPS      how fast frames are fed. If the model is compute-bound, feeding slower
                gives it more work per frame.

Each configuration is rendered with the SAME prompt on the SAME window, then scored on
what actually matters: how far the render moved from the source, whether it survives the
validity gate, and - independently - whether a vision model agrees the edit happened and
whether it invented anything. Two failure directions, both reported, because a
configuration that maximises change by hallucinating a robot is not an improvement.

    .venv/Scripts/python tools/tune_x2.py --prompt "..." --view exterior_image_1_left
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import VIEWS  # noqa: E402
from penumbra.media import write_h264  # noqa: E402
from penumbra.perturbation.spec import FaultSpec, Rung  # noqa: E402
from penumbra.perturbation.x2 import X2Perturbation  # noqa: E402
from penumbra.scenarios.garage import NO_CHANGE_RMSE  # noqa: E402
from penumbra.scenarios.judge import judge_render  # noqa: E402
from penumbra.validation.seam import SeamGate, perceptual_distance  # noqa: E402

log = logging.getLogger("penumbra.tune")

#: Deliberately a prompt the previous run *declined* on, phrased vividly. If a
#: configuration cannot move this one, it cannot rescue the suite.
DEFAULT_PROMPT = ("the wooden tabletop is drenched in glossy dark liquid with bright "
                  "specular highlights across its whole surface")
DEFAULT_SITUATION = "someone spilled a large amount of dark liquid across the workbench"

#: name -> kwargs for X2Perturbation. Baseline first so its number anchors the rest.
CONFIGS = {
    "baseline_lead24_fps15": {"lead": 24, "push_fps": 15.0, "prime_passes": 0},
    "long_lead64": {"lead": 64, "push_fps": 15.0, "prime_passes": 0},
    "prime_1_pass": {"lead": 24, "push_fps": 15.0, "prime_passes": 1},
    "slow_push_fps7": {"lead": 24, "push_fps": 7.0, "prime_passes": 0},
    "prime_1_slow_fps7": {"lead": 24, "push_fps": 7.0, "prime_passes": 1},
}


def fault(prompt: str) -> FaultSpec:
    return FaultSpec(name="tune", family="scenario", model="xmax/x2", description="",
                     ladder=(Rung(0.0, "", "untouched"), Rung(1.0, prompt, "tune")),
                     version="tune-1",
                     validation={"max_geometry_drift": 0.0055,
                                 "min_structure_retained": 0.20})


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--situation", default=DEFAULT_SITUATION)
    ap.add_argument("--view", default=VIEWS[0])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--only", default="", help="comma-separated subset of configs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("reactor_sdk").setLevel(logging.WARNING)

    configs = ({k: CONFIGS[k] for k in args.only.split(",") if k in CONFIGS}
               if args.only else CONFIGS)

    episode = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = episode.frames[args.view]
    spec = fault(args.prompt)
    gate = SeamGate(max_geometry_drift=spec.max_geometry_drift,
                    min_structure_retained=spec.min_structure_retained)

    out = RUNS_DIR / f"tunex2-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    report = {"prompt": args.prompt, "view": args.view, "episode": episode.episode_id,
              "no_change_floor_rmse": NO_CHANGE_RMSE, "configs": records}

    def flush():
        (out / "tune.json").write_text(json.dumps(report, indent=2, default=str),
                                       encoding="utf-8")

    flush()
    for name, kw in configs.items():
        log.info("[%s] %s", name, kw)
        t0 = time.time()
        try:
            res = await X2Perturbation(view=args.view, **kw).apply(
                episode, spec, 1.0, run_id="tune")
        except Exception as exc:  # noqa: BLE001
            log.exception("%s failed", name)
            records.append({"config": name, **kw, "error": f"{type(exc).__name__}: {exc}"})
            flush()
            continue
        perturbed = res.episode.frames[args.view]
        d = perceptual_distance(source, perturbed)
        report_gate = gate.validate(source, perturbed)
        verdict = judge_render(args.prompt, args.situation, source, perturbed,
                               view=args.view)
        write_h264(out / f"{name}.mp4",
                   (frame for pair in zip(source, perturbed) for frame in pair), 15.0)
        rec = {
            "config": name, **kw,
            "seconds": round(time.time() - t0, 1),
            "rmse": round(d["rmse"], 5),
            "ssim_drop": round(d["ssim_drop"], 5),
            # The same bar the garage uses to decide the editor declined.
            "declined": bool(d["rmse"] < NO_CHANGE_RMSE),
            "gate": report_gate.to_dict(),
            "judge": verdict.to_dict(),
        }
        records.append(rec)
        flush()
        log.info("  rmse %.4f  gate %s  judge %s  %s", rec["rmse"],
                 report_gate.status, verdict.verdict,
                 ("invented " + ", ".join(verdict.added_objects)) if verdict.added_objects
                 else "nothing invented")

    ok = [r for r in records if "error" not in r]
    print("\n" + "=" * 88)
    print(f"prompt: {args.prompt}")
    print(f"{'config':<24}{'rmse':<9}{'declined':<10}{'gate':<11}{'judge':<20}invented")
    print("-" * 88)
    for r in ok:
        print(f"{r['config']:<24}{r['rmse']:<9.4f}{str(r['declined']):<10}"
              f"{r['gate']['status']:<11}{r['judge']['verdict']:<20}"
              f"{', '.join(r['judge']['added_objects']) or '-'}")
    print("-" * 88)
    usable = [r for r in ok if not r["declined"] and r["gate"]["valid"]
              and r["judge"]["credible"] and not r["judge"]["added_objects"]]
    if usable:
        best = max(usable, key=lambda r: r["rmse"])
        print(f"strongest render that is not declined, not rejected, matches the prompt "
              f"and invents nothing: {best['config']} (rmse {best['rmse']:.4f})")
    else:
        print("NO configuration produced a render that is simultaneously applied, valid "
              "and free of invented objects. That is a result about X2, not about the "
              "streaming settings.")
    print(f"\nwritten {out / 'tune.json'}")


if __name__ == "__main__":
    asyncio.run(main())
