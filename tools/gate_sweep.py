"""Render a suite and gate it per camera. No policy, no statistics, no money on rollouts.

Two questions the finished suites could not answer, because they saved only one camera's
media and recorded only the strictest of the three verdicts:

  1. WHICH CAMERA drives the rejections? Re-measuring the saved view of a completed run
     puts every render at 0.0022-0.0100 p90, comfortably under the 0.012 bar - yet the
     recorded verdicts ran 0.008-0.019 and rejected two thirds. The difference has to
     come from a view that was never written to disk.

  2. Does the REFERENCE matter? Each scenario's p-value is computed against the no-op
     control, so that X2's own re-rendering cancels. The gate compares against the raw
     source instead. Measuring both on the same render shows whether the gate is
     rejecting for something the experiment already controls for.

Reuses the prompts from an existing run so the numbers are directly comparable:

    .venv/Scripts/python tools/gate_sweep.py --from runs/garage-ep0-...

Renders cost Reactor time but no policy time, so a full 21-situation sweep is a
fraction of a full garage run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.episodes.types import VIEWS  # noqa: E402
from penumbra.media import write_h264  # noqa: E402
from penumbra.perturbation.spec import FaultSpec, Rung  # noqa: E402
from penumbra.perturbation.x2 import X2Perturbation  # noqa: E402
from penumbra.scenarios.garage import NULL_PROMPT  # noqa: E402
from penumbra.validation.seam import SeamGate  # noqa: E402

log = logging.getLogger("penumbra.gate_sweep")


def fault(name: str, prompt: str) -> FaultSpec:
    return FaultSpec(name=name, family="scenario", model="xmax/x2", description="",
                     ladder=(Rung(0.0, "", "untouched"), Rung(1.0, prompt, name)),
                     version="gate-sweep-1",
                     validation={"max_geometry_drift": 0.0055,
                                 "min_structure_retained": 0.20})


def save_views(out: Path, name: str, source, perturbed, fps: float,
               views=VIEWS) -> None:
    """Every camera, source above perturbed - the media the earlier runs did not keep."""
    tw, th = 320, 180
    rows = []
    for v in views:
        idx = np.linspace(0, len(perturbed.frames[v]) - 1, 4).astype(int)
        top = np.concatenate([cv2.resize(source.frames[v][i], (tw, th)) for i in idx], axis=1)
        bot = np.concatenate([cv2.resize(perturbed.frames[v][i], (tw, th)) for i in idx], axis=1)
        rows += [top, bot]
    cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(np.concatenate(rows, axis=0),
                                                       cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    n = min(len(perturbed.frames[v]) for v in views)

    def _grid():
        for i in range(n):
            top = np.concatenate([cv2.resize(source.frames[v][i], (tw, th)) for v in views], axis=1)
            bot = np.concatenate([cv2.resize(perturbed.frames[v][i], (tw, th)) for v in views], axis=1)
            yield np.concatenate([top, bot], axis=0)

    write_h264(out / f"{name}.mp4", _grid(), fps)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_run", type=Path, required=True,
                    help="an existing garage run whose prompts to re-render")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="only the first N situations")
    ap.add_argument("--views", default=",".join(VIEWS),
                    help="comma-separated camera views to perturb and gate")
    args = ap.parse_args()

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise SystemExit(f"unknown view(s) {unknown}; known: {list(VIEWS)}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("reactor_sdk").setLevel(logging.WARNING)

    state = json.loads((args.src_run / "state.json").read_text(encoding="utf-8"))
    items = [(s["name"], s["prompt"], s["status"],
              (s.get("gate") or {}).get("geometry_drift_p90"))
             for s in state["scenarios"] if s.get("prompt")]
    if args.limit:
        items = items[: args.limit]

    episode = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)

    out = RUNS_DIR / f"gatesweep-ep{args.episode}-{time.strftime('%Y%m%d-%H%M%S')}"
    (out / "media").mkdir(parents=True, exist_ok=True)
    media = out / "media"

    gate = SeamGate(max_geometry_drift=0.0055, max_geometry_drift_p90=0.012,
                    min_structure_retained=0.20)
    perturber = X2Perturbation(views=views)

    log.info("control: no-op re-render")
    control = (await perturber.apply(episode, fault("null_control", NULL_PROMPT), 1.0,
                                     run_id="gatesweep")).episode
    save_views(media, "control", episode, control, episode.fps, views)
    control_gate = {v: gate.validate(episode.frames[v], control.frames[v]).to_dict()
                    for v in views}
    records: list[dict] = []
    report = {
        "source_run": args.src_run.name,
        "episode": args.episode,
        "views": list(views),
        "control_vs_source": control_gate,
        "scenarios": records,
    }

    def flush() -> None:
        (out / "gate_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    flush()
    for i, (name, prompt, prev_status, prev_p90) in enumerate(items, 1):
        log.info("[%d/%d] %s", i, len(items), name)
        t0 = time.time()
        try:
            rendered = (await perturber.apply(episode, fault(name, prompt), 1.0,
                                              run_id="gatesweep")).episode
        except Exception as exc:  # noqa: BLE001
            log.exception("render failed: %s", name)
            records.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            flush()
            continue
        save_views(media, name, episode, rendered, episode.fps, views)

        vs_source = {v: gate.validate(episode.frames[v], rendered.frames[v]) for v in views}
        vs_control = {v: gate.validate(control.frames[v], rendered.frames[v]) for v in views}
        rec = {
            "name": name,
            "prompt": prompt,
            "previous_status": prev_status,
            "previous_gate_p90": prev_p90,
            "seconds": round(time.time() - t0, 1),
            "vs_source": {v: r.to_dict() for v, r in vs_source.items()},
            "vs_control": {v: r.to_dict() for v, r in vs_control.items()},
            "verdict_vs_source": "VALID" if all(r.valid for r in vs_source.values()) else "REJECTED",
            "verdict_vs_control": "VALID" if all(r.valid for r in vs_control.values()) else "REJECTED",
            "worst_view_vs_source": max(views, key=lambda v: vs_source[v].geometry_drift_p90),
            "failing_views_vs_source": [v for v in views if not vs_source[v].valid],
            "failing_views_vs_control": [v for v in views if not vs_control[v].valid],
        }
        records.append(rec)
        flush()
        log.info("  %s  worst=%s  p90 " + "  ".join(f"{v}=%.5f" for v in views),
                 rec["verdict_vs_source"], rec["worst_view_vs_source"],
                 *[vs_source[v].geometry_drift_p90 for v in views])

    # -- summary ---------------------------------------------------------------
    ok = [r for r in records if "error" not in r]
    blame = {v: sum(1 for r in ok if v in r["failing_views_vs_source"]) for v in views}
    sole = {v: sum(1 for r in ok if r["failing_views_vs_source"] == [v]) for v in views}
    report["summary"] = {
        "rendered": len(ok),
        "valid_vs_source": sum(1 for r in ok if r["verdict_vs_source"] == "VALID"),
        "valid_vs_control": sum(1 for r in ok if r["verdict_vs_control"] == "VALID"),
        "views_failing": blame,
        "sole_failing_view": sole,
        "median_p90_by_view_vs_source": {
            v: round(float(np.median([r["vs_source"][v]["geometry_drift_p90"] for r in ok])), 5)
            for v in views},
        "median_p90_by_view_vs_control": {
            v: round(float(np.median([r["vs_control"][v]["geometry_drift_p90"] for r in ok])), 5)
            for v in views},
    }
    flush()

    print("\n" + "=" * 78)
    print(f"{'situation':<32}{'vs source':<12}{'vs control':<12}worst view")
    print("-" * 78)
    for r in ok:
        print(f"{r['name']:<32}{r['verdict_vs_source']:<12}{r['verdict_vs_control']:<12}"
              f"{r['worst_view_vs_source']}")
    s = report["summary"]
    print("-" * 78)
    print(f"valid gating against the raw source : {s['valid_vs_source']}/{s['rendered']}")
    print(f"valid gating against the control    : {s['valid_vs_control']}/{s['rendered']}")
    print(f"views failing        : {s['views_failing']}")
    print(f"sole failing view    : {s['sole_failing_view']}")
    print(f"median p90 vs source : {s['median_p90_by_view_vs_source']}")
    print(f"median p90 vs control: {s['median_p90_by_view_vs_control']}")
    print(f"\nwritten {out / 'gate_sweep.json'}")


if __name__ == "__main__":
    asyncio.run(main())
