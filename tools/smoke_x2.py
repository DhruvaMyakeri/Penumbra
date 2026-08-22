"""Smoke test: push real DROID frames into X2 and count what comes back.

Answers, empirically:
  1. Does X2 accept arbitrary decoded video frames via publish_track/push_frame?
  2. What output resolution does it choose?
  3. What is the input:output frame ratio, and does keep_backlog change it?
  4. What is the observed end-to-end latency and throughput?
  5. Do the returned pixels actually differ from the source in the way the prompt asked?

Writes frames + a JSON report to runs/_probe/x2_smoke/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import RUNS_DIR  # noqa: E402
from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.reactor.session import ReactorSession  # noqa: E402

OUT = RUNS_DIR / "_probe" / "x2_smoke"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--keep-backlog", action="store_true")
    ap.add_argument("--prompt", default="the floor and table are wet and highly reflective, "
                                        "mirror-like specular reflections of overhead lights")
    ap.add_argument("--drain", type=float, default=25.0, help="seconds to wait after last push")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref = [r for r in list_episodes() if r.episode_index == 0][0]
    ep = load_episode(ref).window(120, args.frames)
    src = ep.frames["exterior_image_1_left"]
    print(f"source: {src.shape} from {ep.episode_id}")

    received: list[tuple[float, np.ndarray]] = []
    events: list[dict] = []
    t_start = None

    async with ReactorSession("xmax/x2") as s:
        s.on_event(lambda m: events.append({"t": round(time.time() - t0, 3), **m}))
        t0 = time.time()

        out_track = s.track("main_video")

        @out_track.on_frame
        def _on_frame(frame: np.ndarray) -> None:
            received.append((time.time() - t0, frame.copy()))

        await s.command("set_keep_backlog", {"keep_backlog": bool(args.keep_backlog)})
        await s.command("set_prompt", {"prompt": args.prompt})
        source = await s.publish("source")
        print("published source track; pushing frames...")

        t_start = time.time()
        period = 1.0 / args.fps
        for i, f in enumerate(src):
            source.push_frame(np.ascontiguousarray(f))
            nxt = t_start + (i + 1) * period
            await asyncio.sleep(max(0.0, nxt - time.time()))
        push_done = time.time()
        print(f"pushed {len(src)} frames in {push_done - t_start:.1f}s; "
              f"{len(received)} received so far. draining {args.drain}s...")

        deadline = time.time() + args.drain
        last_n = -1
        while time.time() < deadline:
            await asyncio.sleep(1.0)
            if len(received) != last_n:
                last_n = len(received)
                print(f"  received={last_n}", flush=True)

        sess = s.log

    report = {
        "source_frames": int(len(src)),
        "source_shape": list(src.shape[1:]),
        "push_fps_target": args.fps,
        "push_seconds": round(push_done - t_start, 2),
        "keep_backlog": bool(args.keep_backlog),
        "prompt": args.prompt,
        "received_frames": len(received),
        "out_shape": list(received[0][1].shape) if received else None,
        "first_frame_latency_s": round(received[0][0], 2) if received else None,
        "last_frame_t_s": round(received[-1][0], 2) if received else None,
        "output_fps": round(len(received) / max(1e-9, received[-1][0] - received[0][0]), 2)
        if len(received) > 1 else None,
        "connect_seconds": sess.connect_seconds,
        "session_id": sess.session_id,
        "errors": sess.errors,
        "events": events,
        "status_timeline": sess.status_timeline,
    }

    if received:
        arr = np.stack([f for _, f in received])
        np.save(OUT / "received.npy", arr)
        np.save(OUT / "source.npy", src)
        h, w = src.shape[1:3]
        n = min(5, len(arr))
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        sidx = np.linspace(0, len(src) - 1, n).astype(int)
        top = np.concatenate([cv2.resize(src[i], (w // 2, h // 2)) for i in sidx], axis=1)
        bot = np.concatenate([cv2.resize(arr[i], (w // 2, h // 2)) for i in idx], axis=1)
        cv2.imwrite(str(OUT / "compare.png"),
                    cv2.cvtColor(np.concatenate([top, bot], axis=0), cv2.COLOR_RGB2BGR))
        report["mean_abs_diff_resized"] = float(
            np.abs(cv2.resize(arr[len(arr) // 2], (w, h)).astype(np.float32)
                   - src[len(src) // 2].astype(np.float32)).mean()
        )

    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2, default=str))
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
