"""Run a policy through directed scenarios, with a live dashboard.

    python tools/garage.py --episode 0 --scenarios 5

Opens http://127.0.0.1:8765 and streams every generation, gate verdict and
statistical result as it happens.

What it does, in order:
  1. calibrate the policy's own run-to-run noise
  2. ask the director for situations, grounded in this task and this workspace
  3. render a NO-OP control - the same episode through X2 asking for no change
  4. render each situation onto every camera view
  5. reject any render the validity gate refuses, before the policy sees it
  6. roll the policy out and test each situation against the CONTROL

Step 6 is the one that matters. Measuring against the raw recording would credit
the video model's re-rendering to the scenario; measuring against the no-op cancels it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import VIEWS  # noqa: E402
from penumbra.experiments.runner import ExperimentRunner  # noqa: E402
from penumbra.policy.cosmos_droid import CosmosDroidPolicy  # noqa: E402
from penumbra.scenarios.garage import Garage  # noqa: E402
from penumbra.ui.server import serve  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--runs", type=int, default=6, help="unperturbed rollouts (noise floor)")
    ap.add_argument("--repeats", type=int, default=6, help="rollouts for a confirmed candidate")
    ap.add_argument("--screen", type=int, default=3, help="rollouts used to screen every render")
    ap.add_argument("--scenarios", type=int, default=21,
                    help="situations to sweep, spread across 7 mechanism categories")
    ap.add_argument("--views", default=",".join(VIEWS),
                    help="cameras to perturb. X2 corrupts the wrist view on most content "
                         "prompts, so restricting to the exteriors raises the share of "
                         "renders that survive the validity gate")
    ap.add_argument("--attempts", type=int, default=3,
                    help="how many times the render agent may rewrite a prompt and try "
                         "again when the gate or the intent judge rejects a generation")
    ap.add_argument("--prime", type=int, default=1,
                    help="whole-clip passes streamed and discarded before the one that "
                         "counts, so X2 has committed to the edit (costs a render each)")
    ap.add_argument("--push-fps", type=float, default=7.0,
                    help="how fast frames are fed to X2; slower gives it more time")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue a run directory that was interrupted: its situations "
                         "and any renders saved to disk are reused, and only what is "
                         "still queued is rendered again")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("reactor_sdk").setLevel(logging.WARNING)

    ref = [r for r in list_episodes() if r.episode_index == args.episode]
    if not ref:
        raise SystemExit(f"episode {args.episode} not available")
    episode = load_episode(ref[0]).window(args.start, args.frames)

    out = args.resume or (
        RUNS_DIR / f"garage-ep{args.episode}-{time.strftime('%Y%m%d-%H%M%S')}")
    serve(out, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  dashboard: {url}\n  run dir:   {out}\n", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown view(s) {unknown}; known: {list(VIEWS)}")

    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    garage = Garage(runner, out, views=views, repeats=args.repeats,
                    screen_repeats=args.screen,
                    max_render_attempts=args.attempts,
                    render_settings={"prime_passes": args.prime,
                                     "push_fps": args.push_fps})

    try:
        state = await garage.run(episode, n_scenarios=args.scenarios,
                                 baseline_runs=args.runs,
                                 resume=args.resume is not None)
    except Exception as exc:  # noqa: BLE001
        logging.exception("garage run failed")
        garage._publish(phase="error", error=str(exc))
        raise

    print("\n" + "=" * 74)
    print(f"TASK: {state['task']}")
    print("=" * 74)
    # CONFIRMED first, then the effects whose cause is not established, then the rest.
    # Sorting these together - or marking them with the same star - is the reporting
    # error this whole layer exists to prevent.
    rank = {"CONFIRMED": 0, "UNATTRIBUTED": 1}
    for s in sorted(state["scenarios"], key=lambda x: (
            rank.get(x.get("finding_class"), 2),
            x.get("vs_control", {}).get("p_value", 1) if x.get("vs_control") else 1)):
        klass = s.get("finding_class", "")
        mark = "***" if klass == "CONFIRMED" else (" ? " if klass == "UNATTRIBUTED" else "   ")
        p = s.get("vs_control", {}).get("p_value") if s.get("vs_control") else None
        pv = f"p={p:.4f}" if p is not None else "        "
        print(f"{mark} {s['name']:<30} {s.get('category','')[:18]:<19} "
              f"{klass:<13} {pv}")
    summary = state.get("summary", {})
    print("-" * 74)
    print(f"CONFIRMED       : {summary.get('confirmed')}")
    print(f"moved the policy but the cause is NOT established:")
    print(f"                  {summary.get('unattributed')}")
    print(f"renders rejected: {summary.get('rejected_renders')}")
    print(f"editor declined : {summary.get('no_change_renders')}")
    print(f"cost            : ${state['cost']['estimated_usd']}")
    print(f"report          : {out / 'REPORT.md'}")
    print(f"dashboard       : {url}   (still serving; ctrl-c to stop)")
    print(f"state           : {out / 'state.json'}")

    # Keep serving so the dashboard stays readable after the run finishes.
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
