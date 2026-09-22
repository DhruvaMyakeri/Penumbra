"""Build the static showcase in demo/ from finished runs.

Everything the demo shows comes from real runs on disk: the state each suite wrote as
it ran, the media it rendered, and the policy's own rollouts. `runs/` is not versioned
(it is gigabytes of frames), so this script is how the demo's data was produced and how
it can be rebuilt after new runs.

    .venv/Scripts/python tools/build_demo.py

Writes demo/assets/data.js plus re-encoded media under demo/assets/media/. Videos are
re-encoded to H.264, because the earliest suites wrote MPEG-4 Part 2, which no browser
plays.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penumbra.evaluation.multiple import correct  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEMO = ROOT / "demo"
ASSETS = DEMO / "assets"
MEDIA = ASSETS / "media"

#: Completed suites, oldest first. Suites that crashed before testing anything are left
#: out because they have nothing to show.
SUITES = [
    "garage-ep0-20260821-141632",
    "garage-ep0-20260821-142420",
    "garage-ep0-20260821-143928",
    "garage-ep0-20260821-163821",
    "garage-ep0-20260821-175324",
    "garage-ep0-20260821-205044",
    "garage-ep0-20260821-220222",
    "garage-ep0-20260821-234747",
    "garage-ep0-20260823-062143",
]
BASELINE_CACHE = RUNS / "_baseline_cache" / "10da2a2327f1.npz"
#: Visual quality is set by the renderer, not the encode: these clips are 300x169 tiles
#: of an already-generated video. CRF 32 is visually indistinguishable at that size and
#: keeps the page light, which matters because a visitor may open dozens of them.
CRF = 32
#: Posters are what the page shows until a clip is played. Card thumbnails render around
#: 300px wide, so a 480px poster is sharp on a 1.5x screen without shipping the full frame.
POSTER_W = 480
#: How many contrast cases of each kind to include alongside the policy shifts.
CONTRAST = {"REJECTED": 2, "NO EFFECT": 3, "DECLINED": 1}


def finding_class(s: dict, n_views: int) -> str:
    """The same rule the pipeline applies today, applied uniformly to every suite."""
    status = s.get("status")
    if status == "no_change":
        return "DECLINED"
    if status == "rejected":
        return "REJECTED"
    if status == "error":
        return "ERROR"
    if status != "done":
        return "PENDING"
    if not s.get("is_vulnerability"):
        return "NO EFFECT"
    it = s.get("intent") or {}
    if not it or it.get("error"):
        return "UNATTRIBUTED"
    if it.get("coherent") is False:
        return "UNATTRIBUTED"
    if n_views > 1 and it.get("coherent") is not True:
        return "UNATTRIBUTED"
    if it.get("verdict") not in ("applied", "partially_applied"):
        return "UNATTRIBUTED"
    return "CONFIRMED"


def encode(src: Path, dst: Path) -> tuple[int, int]:
    """Re-encode to browser-playable H.264 and write a poster frame beside it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        fps = int(round(float(stream.average_rate or 15)))
        for fr in inp.decode(stream):
            frames.append(fr.to_ndarray(format="rgb24"))
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = w2, h2, "yuv420p"
        vs.options = {"crf": str(CRF), "preset": "slow"}
        for f in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h2, :w2]), format="rgb24")
            for p in vs.encode(vf.reformat(format="yuv420p")):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
    mid = frames[len(frames) // 2]
    ph = round(h2 * POSTER_W / w2)
    thumb = cv2.resize(mid, (POSTER_W, ph), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst.with_suffix(".jpg")), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 72, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
    return w2, h2


def layout(w: int, h: int) -> str:
    """How a clip is composed, so the page labels it correctly.

    The first two suites saved only the repainted view, as one 640x360 frame. From the
    third suite on, every clip is a grid of 300x169 tiles, original views on top and
    repainted views below. Aspect ratio cannot tell them apart - a two-camera grid is
    600x338, which is also 16:9 - so the tile size decides.
    """
    return "grid" if h == 2 * 169 and w % 300 == 0 else "single"


def stats(c: dict | None) -> dict | None:
    if not c:
        return None
    return {
        "p": c.get("p_value"),
        "d": c.get("effect_size_cohens_d"),
        "gp": c.get("gripper_p_value"),
        "within": c.get("within_group_divergence"),
        "between": c.get("between_group_divergence"),
        "gw": c.get("gripper_flip_rate_within_baseline"),
        "gb": c.get("gripper_flip_rate_between_groups"),
        "n": c.get("n_baseline"),
        "m": c.get("n_perturbed"),
        "floor": c.get("at_resolution_floor"),
        "sig_arm": c.get("significant_at_alpha"),
        "sig_grip": c.get("significant_gripper"),
    }


def vision(it: dict | None) -> dict | None:
    if not it or not it.get("verdict"):
        return None
    return {
        "verdict": it.get("verdict"),
        "observed": it.get("observed", ""),
        "added": it.get("added_objects") or [],
        "coherent": it.get("coherent"),
        "per_view": {v: {"verdict": d.get("verdict"), "observed": d.get("observed", "")}
                     for v, d in (it.get("per_view") or {}).items()},
    }


def gate(g: dict | None) -> dict | None:
    if not g:
        return None
    return {
        "status": g.get("status"),
        "drift": g.get("geometry_drift"),
        "drift_px": g.get("geometry_drift_pixels_equivalent"),
        "structure": g.get("structure_retained"),
        "temporal": g.get("temporal_ratio"),
        "reasons": g.get("reasons") or [],
    }


def main() -> None:
    if MEDIA.exists():
        shutil.rmtree(MEDIA)
    MEDIA.mkdir(parents=True)

    suites, scenarios, contrast_pool = [], [], []
    for idx, run in enumerate(SUITES, 1):
        rd = RUNS / run
        st = json.loads((rd / "state.json").read_text(encoding="utf-8"))
        views = st.get("views") or []
        audit = {}
        if (rd / "coherence_audit.json").exists():
            audit = {r["name"]: r for r in
                     json.loads((rd / "coherence_audit.json").read_text(encoding="utf-8"))}
        sid = f"S{idx:02d}"

        ctrl = st.get("control") or {}
        ctrl_media = None
        # The control's own clip is not shipped: the page never plays it, and it would
        # add a video per suite to every clone of the repo for nothing.
        ctrl_layout = None

        rows = st.get("scenarios") or []
        tested = [s for s in rows if s.get("status") == "done"]
        # The first suite ran before the suite-wide correction step existed, so its
        # records carry no verdict. Apply the pipeline's own correction to it - the same
        # function, on the same trajectory p-values - so every suite is held to the
        # same bar. Marked `fdr_retro` so it stays distinguishable from a live verdict.
        retro = False
        if tested and all(s.get("passes_fdr") is None for s in tested):
            fixed = correct({s["name"]: s["vs_control"]["p_value"] for s in tested
                             if s.get("vs_control")})
            verdicts = {c.name: c for c in fixed.results}
            for s in tested:
                c = verdicts.get(s["name"])
                if c is not None:
                    s["passes_fdr"], s["passes_bonferroni"] = c.passes_fdr, c.passes_bonferroni
                    if not c.passes_fdr:
                        s["is_vulnerability"] = False
            retro = True
        for s in rows:
            if s["name"] in audit and s.get("intent"):
                a = audit[s["name"]]
                s["intent"]["coherent"] = a.get("coherent")
                s["intent"]["coherence_note"] = a.get("note")
                s["intent"]["per_view"] = a.get("per_view") or {}
            klass = finding_class(s, len(views))
            rec = {
                "suite": sid,
                "name": s["name"],
                "category": s.get("category") or "",
                "situation": s.get("situation") or "",
                "why": s.get("why_it_might_break") or "",
                "prompt": s.get("prompt_used") or s.get("prompt") or "",
                "klass": klass,
                "stats": stats(s.get("vs_control")),
                "gate": gate(s.get("gate")),
                "vision": vision(s.get("intent")),
                "attempts": len(s.get("render_attempts") or []) or 1,
                "rollouts": s.get("rollouts") or 0,
                "fdr": s.get("passes_fdr"),
                "fdr_retro": retro,
                "bonf": s.get("passes_bonferroni"),
                "_src": str(rd / s["video"]) if s.get("video") else None,
            }
            if klass in ("CONFIRMED", "UNATTRIBUTED"):
                scenarios.append(rec)
            elif klass in CONTRAST and rec["_src"] and Path(rec["_src"]).exists():
                contrast_pool.append(rec)

        suites.append({
            "id": sid,
            "run": run,
            "started": run.split("-")[-2] + "-" + run.split("-")[-1],
            "episode": st.get("episode"),
            "task": (st.get("task") or "").split("|")[0].strip(),
            "views": views,
            "cost": (st.get("cost") or {}).get("estimated_usd", 0),
            "repeats": st.get("repeats"),
            "proposed": len(rows),
            "tested": len(tested),
            "shifted": sum(1 for s in rows if s.get("is_vulnerability")),
            "noise_floor": st.get("noise_floor"),
            "control": {
                "stats": stats(ctrl.get("vs_baseline")),
                "video": ctrl_media,
                "layout": ctrl_layout,
                "prompt": ctrl.get("prompt"),
            },
        })

    # Contrast cases: the clearest of each kind, so the gate and the policy's robustness
    # are visible next to the shifts. Rejections with the largest drift show the gate
    # catching obvious corruption; no-effect renders the judge saw applied show a real
    # change the policy shrugged off.
    picked = []
    for kind, n in CONTRAST.items():
        pool = [r for r in contrast_pool if r["klass"] == kind]
        if kind == "REJECTED":
            pool.sort(key=lambda r: -((r["gate"] or {}).get("drift") or 0))
        elif kind == "NO EFFECT":
            pool.sort(key=lambda r: (
                (r["vision"] or {}).get("verdict") not in ("applied", "partially_applied"),
                (r["stats"] or {}).get("d") or 0))
        picked += pool[:n]

    for rec in scenarios + picked:
        src = rec.pop("_src")
        rec["video"] = rec["poster"] = None
        if src and Path(src).exists():
            dst = MEDIA / rec["suite"] / f"{rec['name']}.mp4"
            w, h = encode(Path(src), dst)
            rec["layout"], rec["w"], rec["h"] = layout(w, h), w, h
            rec["video"] = f"assets/media/{rec['suite']}/{rec['name']}.mp4"
            rec["poster"] = f"assets/media/{rec['suite']}/{rec['name']}.jpg"
        rec["contrast"] = rec in picked

    # The policy's own output, from six real unperturbed rollouts: the first action of
    # each 32-step chunk, per control step, for all eight channels.
    z = np.load(BASELINE_CACHE)
    telemetry = {
        "rollouts": [np.round(z[f"actions_{i}"][:, 0, :], 4).tolist()
                     for i in range(int(z["n"]))],
        "chunk": np.round(z["actions_0"][0], 4).tolist(),
        "steps": z["steps_0"].tolist(),
    }

    shifted = [r for r in scenarios]
    data = {
        "repo": "https://github.com/DhruvaMyakeri/Penumbra",
        "policy": "reactor/cosmos-nano-policy-droid",
        "renderer": "xmax/x2",
        "totals": {
            "suites": len(suites),
            "proposed": sum(s["proposed"] for s in suites),
            "tested": sum(s["tested"] for s in suites),
            "shifted": len(shifted),
            "verified": sum(1 for r in shifted if r["klass"] == "CONFIRMED"),
            "cost": round(sum(s["cost"] for s in suites), 2),
        },
        "suites": suites,
        "scenarios": scenarios + picked,
        "telemetry": telemetry,
    }
    (ASSETS / "data.js").write_text(
        "window.PENUMBRA = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8")

    size = sum(p.stat().st_size for p in MEDIA.rglob("*") if p.is_file())
    print(f"suites {len(suites)}  shifts {len(shifted)}  contrast {len(picked)} "
          "(" + ", ".join(r["klass"] + ":" + r["name"] for r in picked) + ")")
    print(f"media {size / 1e6:.1f} MB, data.js {(ASSETS / 'data.js').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
