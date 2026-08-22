"""Do the two cameras of a finished run actually show the same situation?

Each camera is rendered in its own X2 session and X2 takes no seed, so nothing couples
the two passes. If they diverge, the policy is shown two contradictory worlds inside a
single observation and no situation name describes its input - which would make the
run's headline findings unattributable rather than merely mislabelled.

This audits a completed run from its saved frames, so it costs a few Gemini calls and
no render spend.

    .venv/Scripts/python tools/coherence_audit.py runs/garage-ep0-<stamp>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.episodes.droid import list_episodes, load_episode  # noqa: E402
from penumbra.scenarios.judge import check_coherence, judge_render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    args = ap.parse_args()

    state = json.loads((args.run / "state.json").read_text(encoding="utf-8"))
    frames_dir = args.run / "frames"
    # The untouched recording is the honest "before". The run saved only its renders.
    episode = load_episode(
        [r for r in list_episodes() if r.episode_index == args.episode][0]
    ).window(args.start, args.frames)
    source = episode.frames

    wanted = {n.strip() for n in args.only.split(",")} if args.only else None
    rows = []
    done = 0
    for s in state["scenarios"]:
        if wanted and s["name"] not in wanted:
            continue
        npz = frames_dir / f"{s['name']}.npz"
        if not npz.exists():
            continue
        if args.limit is not None and done >= args.limit:
            break
        z = np.load(npz)
        views = list(z.files)
        if len(views) < 2:
            continue

        per_view = {}
        for v in views:
            if v not in source:
                continue
            r = judge_render(s.get("prompt_used") or s["prompt"], s["situation"],
                             source[v], z[v], view=v)
            per_view[v] = {"verdict": r.verdict, "observed": r.observed,
                           "added_objects": r.added_objects, "error": r.error}
        coherent, note = check_coherence(per_view)
        rows.append({"name": s["name"], "is_vulnerability": s.get("is_vulnerability"),
                     "coherent": coherent, "note": note, "per_view": per_view})
        done += 1

        mark = {True: "SAME", False: "DIFFERENT", None: "unknown"}[coherent]
        print(f"\n{'='*76}\n{s['name']}   cameras: {mark}"
              f"{'   [counted as a vulnerability]' if s.get('is_vulnerability') else ''}")
        for v, d in per_view.items():
            print(f"  {v:<24} {d['verdict']:<18} {d['observed'][:90]}")
        print(f"  -> {note[:200]}")

    out = args.run / "coherence_audit.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n = len(rows)
    diff = sum(1 for r in rows if r["coherent"] is False)
    same = sum(1 for r in rows if r["coherent"] is True)
    vuln_diff = sum(1 for r in rows if r["coherent"] is False and r["is_vulnerability"])
    print(f"\n{'='*76}")
    print(f"audited {n} renders: {same} coherent, {diff} showing DIFFERENT situations "
          f"on the two cameras, {n - same - diff} undetermined")
    if diff:
        print(f"{vuln_diff} of those incoherent renders were reported as vulnerabilities.")
    print(f"written {out}")


if __name__ == "__main__":
    main()
