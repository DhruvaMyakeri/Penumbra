"""PENUMBRA command line.

    python -m penumbra verify-reactor
    python -m penumbra episodes
    python -m penumbra faults
    python -m penumbra baseline --episode 0 --runs 4
    python -m penumbra perturb  --fault specular_floor --strength 0.6
    python -m penumbra search   --fault specular_floor --mode sweep
    python -m penumbra compare  --fault specular_floor --strength 0.8
    python -m penumbra report   <run-id>
    python -m penumbra replay
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import RUNS_DIR
from .episodes.droid import list_episodes, load_episode
from .episodes.types import PERTURBED_VIEW, VIEWS
from .evaluation.divergence import divergence, judge
from .experiments.record import (
    ExperimentRecord,
    environment_fingerprint,
    list_runs,
    new_run_id,
    run_dir,
    save_record,
    write_side_by_side,
    write_video,
)
from .experiments.runner import ExperimentRunner
from .perturbation.classical import ClassicalPerturbation, match_perceptual_distance
from .perturbation.spec import list_faults, load_fault
from .perturbation.x2 import X2Perturbation
from .policy.cosmos_droid import CosmosDroidPolicy
from .report.html import write_report
from .search.bisect import SearchMode, search_boundary


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("reactor_sdk").setLevel(logging.WARNING)


#: Short names for view sets, so the CLI does not need full dataset column names.
VIEW_ALIASES = {
    "ext1": "exterior_image_1_left",
    "ext2": "exterior_image_2_left",
    "wrist": "wrist_image_left",
}


def _resolve_views(spec: list[str] | None) -> tuple[str, ...]:
    """Turn a --views spec into dataset view names. `all` means every camera."""
    if not spec:
        return (PERTURBED_VIEW,)
    if len(spec) == 1 and spec[0] == "all":
        return tuple(VIEWS)
    out = []
    for name in spec:
        resolved = VIEW_ALIASES.get(name, name)
        if resolved not in VIEWS:
            raise SystemExit(
                f"unknown view {name!r}; use one of {sorted(VIEW_ALIASES)} or 'all'"
            )
        out.append(resolved)
    return tuple(out)


def _load(episode_index: int, start: int, frames: int):
    refs = list_episodes()
    match = [r for r in refs if r.episode_index == episode_index]
    if not match:
        raise SystemExit(
            f"episode {episode_index} not available. Try: "
            f"{[r.episode_index for r in refs][:10]}"
        )
    ep = load_episode(match[0])
    return ep.window(start, frames) if frames > 0 else ep


# ---------------------------------------------------------------- commands


def cmd_episodes(args: argparse.Namespace) -> None:
    for r in list_episodes():
        print(f"ep{r.episode_index:>4}  len={r.length:>4}  {r.task.split('|')[0].strip()[:70]}")


def cmd_faults(args: argparse.Namespace) -> None:
    for name in list_faults():
        f = load_fault(name)
        print(f"{f.name:<20} v{f.version}  family={f.family:<12} model={f.model}")
        print(f"    ladder: {', '.join(f'{r.strength:g}={r.label}' for r in f.ladder)}")


def cmd_verify_reactor(args: argparse.Namespace) -> None:
    import subprocess

    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "tools" / "probe_reactor.py"),
         *(args.models or ["xmax/x2", "reactor/cosmos-nano-policy-droid"])],
        check=False,
    )


async def _baseline(args: argparse.Namespace) -> None:
    ep = _load(args.episode, args.start, args.frames)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    print(f"episode {ep.episode_id}  ({len(ep)} frames)  task={ep.task.split('|')[0].strip()!r}")
    print(f"calibrating noise floor over {args.runs} unperturbed runs...")
    nf = await runner.calibrate(ep, runs=args.runs,
                                use_cache=not getattr(args, 'fresh_baseline', False))
    print(json.dumps(nf.to_dict(), indent=2))
    out = RUNS_DIR / "_baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ep.episode_id.replace('/', '_').replace('@','_at_').replace('+','p')}.json").write_text(
        json.dumps({"episode": ep.episode_id, "noise_floor": nf.to_dict(),
                    "samples": nf.samples, "cost": runner.cost}, indent=2),
        encoding="utf-8",
    )
    print(f"\ncost: {json.dumps(runner.cost)}")


async def _perturb(args: argparse.Namespace) -> None:
    ep = _load(args.episode, args.start, args.frames)
    fault = load_fault(args.fault)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    print(f"calibrating noise floor ({args.runs} runs)...")
    nf = await runner.calibrate(ep, runs=args.runs,
                                use_cache=not getattr(args, 'fresh_baseline', False))
    print(f"  sigma={nf.joint_l2_std:.5f}  mean={nf.joint_l2_mean:.5f}  "
          f"k=3 threshold={nf.threshold(3.0):.5f}")

    print(f"perturbing with {fault.name} @ strength {args.strength} ...")
    views = _resolve_views(getattr(args, "views", None))
    print(f"perturbing view(s): {', '.join(views)}")
    ev = await runner.evaluate(ep, fault, args.strength,
                               perturber=X2Perturbation(views=views, lead=args.lead,
                                                        tail=args.tail),
                               k=args.k, repeats=args.repeats)
    _print_eval(ev)
    _write_artifacts(args, ep, fault, runner, [ev], nf, mode="perturb")


async def _search(args: argparse.Namespace) -> None:
    ep = _load(args.episode, args.start, args.frames)
    fault = load_fault(args.fault)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    print(f"episode {ep.episode_id} ({len(ep)} frames)")
    print(f"calibrating noise floor ({args.runs} runs)...")
    nf = await runner.calibrate(ep, runs=args.runs,
                                use_cache=not getattr(args, 'fresh_baseline', False))
    print(f"  sigma={nf.joint_l2_std:.5f}  mean={nf.joint_l2_mean:.5f}  "
          f"k={args.k} threshold={nf.threshold(args.k):.5f}  "
          f"noise gripper-flip max={nf.gripper_flip_rate_max:.3f}")

    views = _resolve_views(getattr(args, "views", None))
    print(f"perturbing view(s): {', '.join(views)}")
    res = await search_boundary(runner, ep, fault, mode=SearchMode(args.mode), k=args.k,
                                repeats=args.repeats,
                                perturber=X2Perturbation(views=views, lead=args.lead,
                                                         tail=args.tail))
    print("\nLADDER")
    for row in res.curve():
        mark = "BREAK" if row["broke"] else "pass "
        p = f"p={row['p_value']:.4f}" if row.get("p_value") is not None else "p=n/a"
        dd = f"d={row['effect_size']:+.2f}" if row.get("effect_size") is not None else "d=n/a"
        print(f"  {row['strength']:.2f}  {mark}  joint_l2={row['joint_l2']:.4f}  "
              f"{p}  {dd}  flip_rate={row['gripper_flip_rate']:.3f}  rmse={row['rmse']:.4f}")
    for s in res.rejected_strengths:
        print(f"  {s:.2f}  REJECTED by validity gate")
    print(f"\nMARGIN: {res.margin if res.margin is not None else 'ROBUST over this ladder'}")
    for n in res.notes:
        print(f"  note: {n}")
    _write_artifacts(args, ep, fault, runner, res.evaluations, nf, mode="search", search=res)


async def _compare(args: argparse.Namespace) -> None:
    """Generative vs classical at MATCHED perceptual distance — the research question."""
    ep = _load(args.episode, args.start, args.frames)
    fault = load_fault(args.fault)
    runner = ExperimentRunner(policy=CosmosDroidPolicy(chunk=args.chunk))
    print(f"calibrating noise floor ({args.runs} runs)...")
    nf = await runner.calibrate(ep, runs=args.runs,
                                use_cache=not getattr(args, 'fresh_baseline', False))
    print(f"  sigma={nf.joint_l2_std:.5f}  threshold(k={args.k})={nf.threshold(args.k):.5f}")

    # Both arms must perturb the SAME view set, or this is not a control: a
    # three-view generative arm against a one-view classical arm compares channel
    # coverage, not perturbation family.
    views = _resolve_views(getattr(args, "views", None))
    print(f"\n[generative] {fault.name} @ {args.strength}  views: {', '.join(views)}")
    gen = await runner.evaluate(ep, fault, args.strength, k=args.k, repeats=args.repeats,
                                perturber=X2Perturbation(views=views))
    _print_eval(gen)

    print(f"\nmatching classical arms to rmse={gen.distance['rmse']:.4f} ...")
    classical: list = []
    for op in args.ops:
        # Match on the view PENUMBRA measures distance against, then apply to the
        # SAME view set as the generative arm. Matching on one view and applying to
        # one view while the generative arm hit three would not be a control.
        matched = match_perceptual_distance(ep, op, gen.distance, view=PERTURBED_VIEW)
        print(f"  [{op}] strength {matched.strength:.3f} -> rmse {matched.distance['rmse']:.4f}")
        ev = await runner.evaluate_classical(ep, op, matched.strength, fault=fault,
                                             k=args.k, repeats=args.repeats, views=views)
        _print_eval(ev, prefix=f"  {op}: ")
        classical.append(ev)

    print("\nCOMPARISON at matched perceptual distance")
    print(f"  generative {fault.name}@{args.strength}: "
          f"{'BREAK' if gen.broke else 'pass'}  joint_l2={gen.divergence.joint_l2 if gen.divergence else float('nan'):.4f}")
    for ev in classical:
        print(f"  classical {ev.perturbation.op:<20}: "
              f"{'BREAK' if ev.broke else 'pass'}  joint_l2={ev.divergence.joint_l2:.4f}  "
              f"rmse={ev.distance['rmse']:.4f}")
    _write_artifacts(args, ep, fault, runner, [gen], nf, mode="compare", classical=classical)


def _print_eval(ev, prefix: str = "") -> None:
    v = ev.validation
    print(f"{prefix}validity: {v.status}  drift={v.geometry_drift:.4f}"
          f"(max {v.max_geometry_drift})  structure={v.structure_retained:.3f}"
          f"(min {v.min_structure_retained})  temporal={v.temporal_ratio:.2f}")
    if v.reasons:
        for r in v.reasons:
            print(f"{prefix}  ! {r}")
    print(f"{prefix}perceptual: rmse={ev.distance['rmse']:.4f} ssim={ev.distance['ssim']:.3f}")
    if ev.rejected:
        print(f"{prefix}=> REJECTED. No policy run, no failure claimed.")
        return
    d, verdict = ev.divergence, ev.verdict
    print(f"{prefix}divergence (single pair): joint_l2={d.joint_l2:.4f} rad  "
          f"gripper_l1={d.gripper_l1:.4f}  flips={d.gripper_flips}/{d.steps_compared}  "
          f"timing_shift={d.timing_shift_steps:+.0f} steps")
    g = ev.group
    if g is not None:
        print(f"{prefix}group test: N={g.n_baseline} baseline vs M={g.n_perturbed} perturbed")
        print(f"{prefix}  within-group  divergence = {g.within_mean:.4f} rad")
        print(f"{prefix}  between-group divergence = {g.between_mean:.4f} rad")
        print(f"{prefix}  joint    statistic = {g.statistic:+.4f}   p = {g.p_value:.4f}"
              f"{' *' if g.significant else ''}   Cohen's d = {g.effect_size:+.2f}")
        print(f"{prefix}  gripper  statistic = {g.gripper_statistic:+.4f}   "
              f"p = {g.gripper_p_value:.4f}{' *' if g.significant_gripper else ''}   "
              f"flip rate {g.gripper_flip_rate_baseline:.3f} -> "
              f"{g.gripper_flip_rate_between:.3f}")
        print(f"{prefix}  alpha/test = {g.alpha_per_test:.4f} (2 statistics tested)   "
              f"min attainable p = {g.min_achievable_p:.4f}   "
              f"{g.permutations} permutations")
        for n in g.notes:
            print(f"{prefix}  note: {n}")
    else:
        print(f"{prefix}group test: not run (needs >=2 repeats per group)")
    print(f"{prefix}=> {'BREAK' if ev.broke else 'PASS'}"
          f"   [single-pair view: {'break' if verdict.broke else 'pass'} at {verdict.sigma:+.1f} sigma]")


def _write_artifacts(args, ep, fault, runner, evaluations, nf, *, mode, search=None, classical=None) -> None:
    run_id = new_run_id(f"{mode}-{fault.name}")
    d = run_dir(run_id)
    source = ep.frames[PERTURBED_VIEW]
    artifacts = []
    artifacts.append(str(write_video(source, d / "source.mp4", ep.fps).name))

    interesting = next((e for e in evaluations if e.broke), None) or (
        evaluations[-1] if evaluations else None
    )
    if interesting is not None:
        pf = interesting.episode_frames if hasattr(interesting, "episode_frames") else \
            interesting.perturbation.episode.frames[PERTURBED_VIEW]
        artifacts.append(str(write_video(pf, d / "perturbed.mp4", ep.fps).name))
        gate = interesting.validation
        caption = (f"{fault.name} s={interesting.strength:g} | drift {gate.geometry_drift:.3f} "
                   f"{gate.status}")
        artifacts.append(str(write_side_by_side(source, pf, d / "side_by_side.mp4",
                                                fps=ep.fps, caption=caption).name))
        if interesting.trace is not None:
            np.save(d / "actions_perturbed.npy", interesting.trace.actions)
    np.save(d / "actions_baseline.npy", runner.baseline.actions)

    # Every rollout, not just the first of each group. Saving one trace per condition
    # made it impossible to pool perturbed rollouts across runs, which is exactly the
    # analysis a seedless generative model needs: each X2 sample is one draw from the
    # intervention's distribution, and pooling averages over that variance instead of
    # betting the conclusion on a single draw.
    groups: dict[str, np.ndarray] = {}
    for i, t in enumerate(runner.baseline_group):
        groups[f"baseline__{i}"] = t.actions
    for ev in evaluations:
        for i, t in enumerate(ev.traces):
            groups[f"perturbed_{ev.strength:g}__{i}"] = t.actions
    for ev in (classical or []):
        op = getattr(ev.perturbation, "op", "classical")
        for i, t in enumerate(ev.traces):
            groups[f"classical_{op}_{ev.strength:g}__{i}"] = t.actions
    if groups:
        np.savez_compressed(d / "all_traces.npz",
                            step_indices=runner.baseline.step_indices, **groups)

    record = ExperimentRecord(
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        episode={
            "episode_id": ep.episode_id,
            "source": ep.source,
            "task": ep.task,
            "fps": ep.fps,
            "frames": len(ep),
            "perturbed_view": PERTURBED_VIEW,
            "held_fixed": ["exterior_image_2_left", "wrist_image_left",
                           "joint_positions", "gripper_position", "ground_truth_actions"],
        },
        policy={
            "model": runner.policy.name,
            "chunk": runner.policy.chunk,
            "warmup_frames": runner.policy.warmup_frames,
            "protocol": "teacher-forced open-loop replay; logged actions echoed as executed",
            "baseline_runs": nf.n_runs,
        },
        perturbation=fault.to_dict(),
        validation={"gate": "SEAM v1 (flow drift + gradient agreement + temporal ratio)"},
        noise_floor=nf.to_dict(),
        divergence={"evaluations": [e.summary() for e in evaluations]},
        verdict=(search.to_dict() if search else
                 {"broke": bool(interesting and interesting.broke),
                  "detail": interesting.verdict.to_dict() if interesting and interesting.verdict else None}),
        cost=runner.cost,
        classical_control=[e.summary() for e in classical] if classical else None,
        environment=environment_fingerprint(),
        artifacts=artifacts,
    )
    save_record(record, d)
    html = write_report(record, d)
    print(f"\nartifacts -> {d}")
    print(f"report    -> {html}")


def cmd_report(args: argparse.Namespace) -> None:
    from .experiments.record import load_record

    d = RUNS_DIR / args.run_id
    if not (d / "experiment.json").exists():
        raise SystemExit(f"no experiment at {d}")
    rec = load_record(d)
    print(write_report(rec, d))


def cmd_replay(args: argparse.Namespace) -> None:
    runs = list_runs()
    if not runs:
        print("no experiments recorded yet")
        return
    for d in runs:
        rec = json.loads((d / "experiment.json").read_text(encoding="utf-8"))
        verdict = rec.get("verdict", {})
        margin = verdict.get("margin")
        broke = verdict.get("found_break", verdict.get("broke"))
        print(f"{d.name:<40} fault={rec['perturbation']['name']:<18} "
              f"break={broke}  margin={margin}  cost=${rec.get('cost',{}).get('estimated_usd')}")


# ---------------------------------------------------------------- wiring


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("penumbra", description="Visual fault injection for physical AI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def episode_args(sp):
        sp.add_argument("--episode", type=int, default=0)
        sp.add_argument("--start", type=int, default=120)
        sp.add_argument("--frames", type=int, default=96)
        sp.add_argument("--chunk", type=int, default=8)
        sp.add_argument("--runs", type=int, default=4, help="unperturbed runs for the noise floor")
        sp.add_argument("--fresh-baseline", action="store_true",
                        help="draw a new baseline group instead of reusing the cache. "
                             "Reusing it makes conditions directly comparable but makes "
                             "their p-values correlated; use this when replications must "
                             "be statistically independent of each other")
        sp.add_argument("--repeats", type=int, default=4,
                        help="policy rollouts per perturbed condition; the policy is "
                             "stochastic, so this is the group size the permutation test uses")
        sp.add_argument("--k", type=float, default=3.0, help="sigma multiple for a break")

    sub.add_parser("episodes").set_defaults(fn=cmd_episodes)
    sub.add_parser("faults").set_defaults(fn=cmd_faults)

    vr = sub.add_parser("verify-reactor")
    vr.add_argument("models", nargs="*")
    vr.set_defaults(fn=cmd_verify_reactor)

    b = sub.add_parser("baseline")
    episode_args(b)
    b.set_defaults(fn=lambda a: asyncio.run(_baseline(a)))

    pt = sub.add_parser("perturb")
    episode_args(pt)
    pt.add_argument("--fault", required=True)
    pt.add_argument("--strength", type=float, required=True)
    pt.add_argument("--views", nargs="*",
                    help="views to perturb: ext1 ext2 wrist, or all (default ext1)")
    pt.add_argument("--lead", type=int, default=24)
    pt.add_argument("--tail", type=int, default=12)
    pt.set_defaults(fn=lambda a: asyncio.run(_perturb(a)))

    s = sub.add_parser("search")
    episode_args(s)
    s.add_argument("--fault", required=True)
    s.add_argument("--mode", choices=[m.value for m in SearchMode], default="sweep")
    s.add_argument("--views", nargs="*",
                    help="views to perturb: ext1 ext2 wrist, or all (default ext1)")
    s.add_argument("--lead", type=int, default=24)
    s.add_argument("--tail", type=int, default=12)
    s.set_defaults(fn=lambda a: asyncio.run(_search(a)))

    c = sub.add_parser("compare")
    episode_args(c)
    c.add_argument("--fault", required=True)
    c.add_argument("--strength", type=float, default=0.8)
    c.add_argument("--ops", nargs="*",
                   default=["specular_overlay", "brightness_contrast", "gamma_glare"])
    c.add_argument("--lead", type=int, default=24)
    c.add_argument("--tail", type=int, default=12)
    c.add_argument("--views", nargs="*",
                   help="views to perturb in BOTH arms: ext1 ext2 wrist, or all")
    c.set_defaults(fn=lambda a: asyncio.run(_compare(a)))

    r = sub.add_parser("report")
    r.add_argument("run_id")
    r.set_defaults(fn=cmd_report)

    sub.add_parser("replay").set_defaults(fn=cmd_replay)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.fn(args)


if __name__ == "__main__":
    main()
