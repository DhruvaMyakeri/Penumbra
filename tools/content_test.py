"""Does the CONTENT of a perturbation matter, or only the re-render?

Every PENUMBRA result so far compares a perturbed group against the *untouched*
episode. That comparison cannot distinguish two very different claims:

    (a) "this fault changes the policy's behaviour"
    (b) "passing the views through a generative model at all changes it"

The `null_edit` control settled that X2 re-rendering alone is sufficient - a prompt
asking for no change scored p = 0.0030, d = +2.32, and moved the gripper channel that
no named fault ever moved. So (b) is established, and every named-fault p-value
measured against the untouched baseline is confounded by it.

The right control for a *content* claim is therefore not the untouched episode. It is
the no-op re-render. This tool runs the identical permutation test between two
perturbed groups:

    H0:  the named fault's rollouts and the no-op re-render's rollouts are
         exchangeable - the content added nothing beyond the re-render

Rejecting H0 is what a content-specific claim requires. Failing to reject it means the
fault, whatever its prompt asked for, is behaviourally indistinguishable from asking
the model for nothing.

Costs nothing: it reads rollouts already recorded in each run's all_traces.npz.

    python tools/content_test.py runs/perturb-target_occlusion-<id> runs/perturb-null_edit-<id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.evaluation.grouptest import compare_groups  # noqa: E402
from penumbra.policy.base import PolicyTrace  # noqa: E402


def load_group(run_dir: Path, kind: str) -> tuple[list[PolicyTrace], str]:
    """Load one condition's rollouts from a run's saved traces.

    `kind` is "perturbed" or "baseline"; the perturbed key carries the strength, so
    the first matching prefix is used and reported back for the record.
    """
    path = run_dir / "all_traces.npz"
    if not path.exists():
        raise SystemExit(
            f"{run_dir.name} has no all_traces.npz - it predates full-trace "
            f"persistence. Re-run it to compare."
        )
    z = np.load(path)
    steps = z["step_indices"]
    keys = [k for k in z.files if k.startswith(kind)]
    if not keys:
        raise SystemExit(f"{run_dir.name}: no {kind!r} traces (found {sorted(z.files)[:6]})")
    label = keys[0].rsplit("__", 1)[0]
    keys = sorted(k for k in keys if k.startswith(label + "__"))
    traces = [
        PolicyTrace(
            policy="recorded",
            actions=z[k],
            step_indices=steps,
            latencies=np.zeros(len(steps), dtype=np.float32),
            run_id=f"{run_dir.name[:18]}:{k}",
        )
        for k in keys
    ]
    return traces, label


def describe(run_dir: Path) -> dict:
    rec = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    ev = (rec.get("divergence") or {}).get("evaluations", [{}])[0]
    return {
        "run": run_dir.name,
        "fault": rec["perturbation"]["name"],
        "episode": rec["episode"]["episode_id"],
        "strength": ev.get("strength"),
        "views": len((ev.get("perturbation") or {}).get("views") or []),
        "rmse": (ev.get("distance") or {}).get("rmse"),
        "gate": (ev.get("validation") or {}).get("status"),
        "p_vs_baseline": (ev.get("group_test") or {}).get("p_value"),
        "d_vs_baseline": (ev.get("group_test") or {}).get("effect_size_cohens_d"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fault_run", type=Path, help="run directory of the named fault")
    ap.add_argument("control_run", type=Path, help="run directory of the no-op re-render")
    ap.add_argument("--permutations", type=int, default=20000)
    args = ap.parse_args()

    fault_meta, control_meta = describe(args.fault_run), describe(args.control_run)
    if fault_meta["episode"] != control_meta["episode"]:
        raise SystemExit(
            f"episodes differ ({fault_meta['episode']} vs {control_meta['episode']}); "
            f"a content test must hold the episode fixed"
        )

    fault_traces, fault_label = load_group(args.fault_run, "perturbed")
    control_traces, control_label = load_group(args.control_run, "perturbed")

    print("=" * 72)
    print("CONTENT TEST - does the fault differ from a no-op generative re-render?")
    print("=" * 72)
    for tag, m in (("fault  ", fault_meta), ("control", control_meta)):
        print(f"  {tag}  {m['fault']:<18} s={m['strength']}  views={m['views']}  "
              f"rmse={m['rmse']:.4f}  gate={m['gate']}")
        print(f"           vs untouched baseline: p={m['p_vs_baseline']:.4f} "
              f"d={m['d_vs_baseline']:+.2f}")
    print(f"\n  episode: {fault_meta['episode']}")
    print(f"  groups:  {len(fault_traces)} fault rollouts vs {len(control_traces)} control rollouts")

    result = compare_groups(fault_traces, control_traces, permutations=args.permutations)
    print("\n  permutation test, fault group vs no-op re-render group")
    print(f"    within-group  divergence = {result.within_mean:.5f} rad")
    print(f"    between-group divergence = {result.between_mean:.5f} rad")
    print(f"    joint    p = {result.p_value:.4f}"
          f"{' *' if result.significant else ''}   d = {result.effect_size:+.2f}")
    print(f"    gripper  p = {result.gripper_p_value:.4f}"
          f"{' *' if result.significant_gripper else ''}")
    print(f"    alpha/test = {result.alpha_per_test}   "
          f"min attainable p = {result.min_achievable_p:.4f}")
    for n in result.notes:
        print(f"    note: {n}")

    content_matters = result.any_significant
    verdict = (
        "The fault is behaviourally DISTINGUISHABLE from a no-op re-render. Its "
        "content is doing something the re-render alone does not."
        if content_matters else
        "The fault is behaviourally INDISTINGUISHABLE from asking the model for "
        "nothing. Its effect against the untouched baseline is explained by "
        "generative re-rendering, and no content-specific claim is supported."
    )
    print(f"\n  VERDICT: {verdict}")

    out = {
        "test": "named fault vs no-op generative re-render",
        "fault": fault_meta,
        "control": control_meta,
        "n_fault_rollouts": len(fault_traces),
        "n_control_rollouts": len(control_traces),
        "result": result.to_dict(),
        "content_specific_effect": bool(content_matters),
        "verdict": verdict,
    }
    path = args.fault_run / "content_test.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  saved -> {path}")


if __name__ == "__main__":
    main()
