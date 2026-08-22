"""Readable reports: one per situation, and one for the run.

A run already writes `state.json`, which has everything in it and is unreadable by a
person. These are for the human at the end: what was found, what it is allowed to mean,
and what would have to be true for it to be wrong.

The organising rule is the same one `ScenarioResult.finding_class` enforces. A statistic
buys the sentence "something in this render moved the policy". Only a render that the
gate accepted, that both cameras agree on, and that an independent judge says depicts
its own name buys the sentence "*this situation* moved the policy". The reports keep
those two claims visually separate and never total them together, because adding them
up is precisely how a suite came to advertise fifteen findings that adjudication cut to
three.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

#: Ordered worst-to-best so a reader meets the caveats before the headline.
CLASS_ORDER = ("CONFIRMED", "UNATTRIBUTED", "NO EFFECT", "REJECTED", "DECLINED", "ERROR")


def _fmt_p(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _stat_line(s: dict) -> str:
    c = s.get("vs_control") or {}
    if not c:
        return "not tested"
    bits = [f"p={_fmt_p(c.get('p_value'))}", f"d={c.get('effect_size_cohens_d', 0):+.2f}"]
    if c.get("gripper_p_value") is not None:
        bits.append(f"grasp p={_fmt_p(c.get('gripper_p_value'))}")
    if c.get("at_resolution_floor"):
        bits.append("at the design's p-value floor")
    return "  ".join(bits)


def scenario_report(s: dict, *, run_name: str = "") -> str:
    """One situation, in full: what was asked, what rendered, what it means."""
    klass = s.get("finding_class", "?")
    out = [
        f"# {s['name']}",
        "",
        f"**{klass}** - {s.get('verdict', '')}",
        "",
        f"- run: `{run_name}`",
        f"- category: {s.get('category', '-')}    "
        f"severity {s.get('severity', '-')}/5, realism {s.get('realism', '-')}/5",
        "",
        "## The situation as proposed",
        "",
        f"{s.get('situation', '-')}",
        "",
        f"*Why it might break the policy:* {s.get('why_it_might_break', '-')}",
        "",
        "## What was actually asked of the renderer",
        "",
        f"```\n{s.get('prompt_used') or s.get('prompt', '')}\n```",
    ]
    if (s.get("prompt_used") or s.get("prompt")) != s.get("prompt"):
        out += ["", f"The director's original wording was:", "",
                f"```\n{s.get('prompt')}\n```"]

    attempts = s.get("render_attempts") or []
    if len(attempts) > 1:
        out += ["", "## How many attempts it took", "",
                f"{len(attempts)} renders were made for this situation. A situation "
                f"that needed several tries is weaker evidence than one that worked "
                f"immediately, and the sequence is part of the result.", ""]
        for a in attempts:
            out.append(f"{a.get('attempt')}. **{a.get('outcome')}** - "
                       f"`{a.get('prompt', '')[:150]}`")
            if a.get("detail"):
                out.append(f"   - {a['detail'][:300]}")
            if a.get("agent_reasoning"):
                out.append(f"   - agent: {a['agent_reasoning'][:200]}")

    intent = s.get("intent") or {}
    if intent:
        out += ["", "## What the render actually shows",
                "", f"An independent vision model, shown only the before and after "
                    f"frames and never the statistics, judged this "
                    f"**{intent.get('verdict', '?')}**.", "",
                f"> {intent.get('observed', '-')}", ""]
        if intent.get("added_objects"):
            out.append(f"**Invented, not requested:** "
                       f"{', '.join(intent['added_objects'])}. An object nobody asked "
                       f"for is very likely what the policy reacted to.")
            out.append("")
        per_view = intent.get("per_view") or {}
        if per_view:
            out += ["### Camera by camera", "",
                    "| camera | verdict | what it shows |", "|---|---|---|"]
            for view, d in per_view.items():
                out.append(f"| `{view}` | {d.get('verdict', '?')} | "
                           f"{str(d.get('observed', '')).replace('|', '/')[:160]} |")
            out.append("")
        if intent.get("coherent") is False:
            out += ["**The cameras disagree.** " + str(intent.get("coherence_note", "")),
                    "",
                    "Each camera is rendered in its own pass with no shared seed, so "
                    "the two can diverge. When they do, the policy reads both in one "
                    "observation and is shown two different worlds - so no situation "
                    "name describes its input, whatever the statistics say.", ""]
        elif intent.get("coherent") is True:
            out += ["Both cameras agree on what changed.", ""]

    out += ["## The measurement", "",
            f"{_stat_line(s)}", "",
            f"Rollouts: {s.get('rollouts', 0)}. Compared against a no-op re-render of "
            f"the same episode, not the raw recording.", ""]
    c = s.get("vs_control") or {}
    for note in c.get("notes") or []:
        out.append(f"- {note}")
    if c.get("notes"):
        out.append("")

    gate = s.get("gate") or {}
    if gate:
        out += ["## Validity", "",
                f"Gate: **{gate.get('status', '?')}**. "
                f"Geometry drift {gate.get('geometry_drift', 0):.5f} "
                f"(bar {gate.get('thresholds', {}).get('max_geometry_drift', 0):.4f}), "
                f"structure retained {gate.get('structure_retained', 0):.4f}, "
                f"temporal ratio {gate.get('temporal_ratio', 0):.2f}.", ""]
        for reason in gate.get("reasons") or []:
            out.append(f"- rejected because: {reason}")
        if gate.get("reasons"):
            out.append("")

    if klass == "UNATTRIBUTED":
        out += ["## What this does NOT establish", "",
                f"The behaviour change is real and survived suite-wide correction. Its "
                f"**cause is not established**: the render does not depict "
                f"'{s['name']}'. Do not quote this situation by name. What it supports "
                f"is the narrower claim that some admissible visual change to this "
                f"scene moves this policy.", ""]
    elif klass == "CONFIRMED":
        out += ["## What this does NOT establish", "",
                "The policy's *intended actions* changed. It was never allowed to act, "
                "so this is not a dropped cup, a collision, or a failed task. One "
                "episode, one window, one policy.", ""]
    return "\n".join(out)


def run_report(state: dict, *, insights: dict | None = None) -> str:
    """The concise end-of-run report: what was found, and what it is worth."""
    summary = state.get("summary") or {}
    scenarios = state.get("scenarios") or []
    by_name = {s["name"]: s for s in scenarios}
    cost = (state.get("cost") or {}).get("estimated_usd", 0)

    confirmed = summary.get("confirmed") or []
    unattributed = summary.get("unattributed") or []
    reasons = summary.get("unattributed_reasons") or {}

    out = [
        f"# PENUMBRA run report",
        "",
        f"**{state.get('episode', '?')}** | policy `{state.get('policy', '?')}` | "
        f"{time.strftime('%Y-%m-%d')}",
        "",
        f"Task: *{state.get('task', '-')}*",
        "",
        f"Cameras perturbed: {', '.join(state.get('views') or [])}. "
        f"Left untouched: {', '.join(state.get('views_untouched') or []) or 'none'}.",
        "",
        "---",
        "",
        "## Headline",
        "",
        f"**{len(confirmed)} confirmed** of {summary.get('tested', 0)} tested "
        f"situations, from {summary.get('proposed', 0)} proposed. "
        f"${cost} and {len(scenarios)} situations.",
        "",
        "A finding is *confirmed* only when the render passed the validity gate, both "
        "cameras agree on what changed, an independent vision model says the render "
        "depicts the situation it is named after, AND the policy's behaviour separated "
        "from a no-op re-render after suite-wide correction.",
        "",
    ]

    if confirmed:
        out += ["## Confirmed findings", "",
                "| situation | statistics | what the render shows |", "|---|---|---|"]
        for name in confirmed:
            s = by_name.get(name, {})
            observed = str((s.get("intent") or {}).get("observed", "")).replace("|", "/")
            out.append(f"| **{name}** | {_stat_line(s)} | {observed[:120]} |")
        out.append("")
    else:
        out += ["## Confirmed findings", "",
                "None. No situation both moved the policy and rendered as described.",
                ""]

    if unattributed:
        out += ["## Moved the policy, but the cause is not established", "",
                f"{len(unattributed)} situations produced a real, correction-surviving "
                f"behaviour change whose *cause* the render does not support. These are "
                f"**not** findings about their own names and must not be quoted as "
                f"such. They are listed because suppressing them would hide the "
                f"renderer's actual failure rate.", "",
                "| situation | statistics | why the name is not supported |",
                "|---|---|---|"]
        for name in unattributed:
            s = by_name.get(name, {})
            out.append(f"| {name} | {_stat_line(s)} | "
                       f"{str(reasons.get(name, '')).replace('|', '/')[:110]} |")
        out.append("")

    out += ["## Everything else", "", "| outcome | n | situations |", "|---|---|---|"]
    for label, key in (("no effect", "no_effect"),
                       ("render rejected by the gate", "rejected_renders"),
                       ("editor declined - nothing rendered", "no_change_renders"),
                       ("errored", "errors")):
        names = summary.get(key) or []
        out.append(f"| {label} | {len(names)} | {', '.join(names[:8]) or '-'} |")
    out.append("")

    if insights:
        out += ["## What the renderer did, across this run", "",
                f"{insights.get('attempts', 0)} render attempts were made for "
                f"{len(scenarios)} situations.", "",
                "| outcome | n |", "|---|---|"]
        for label, key in (("usable", "usable"),
                           ("editor declined", "declined"),
                           ("gate rejected", "gate_rejected"),
                           ("rendered the wrong thing", "wrong_thing"),
                           ("cameras disagreed", "incoherent")):
            out.append(f"| {label} | {insights.get(key, 0)} |")
        out.append("")
        invented = insights.get("invented_objects") or []
        if invented:
            out += [f"**Objects the renderer invented unprompted:** "
                    f"{', '.join(invented[:15])}.", ""]

    out += [
        "---",
        "",
        "## What this run does not establish",
        "",
        "- **No robot failed.** What is measured is whether the policy's *predicted "
        "actions* changed. It was never allowed to act, so there is no task success, "
        "no dropped cup, no collision.",
        "- **One episode, one window, one policy.** Any claim about this policy in "
        "general is unsupported by this run alone.",
        "- **The renders are selected.** A situation that took three attempts to render "
        "acceptably is drawn from the renderer's output *conditional on being "
        "admissible*, and the attempt count is part of each result.",
        "- **Unattributed findings name nothing.** They establish that some admissible "
        "visual change moves this policy, not which one.",
        "",
    ]
    return "\n".join(out)


def write_reports(out_dir: Path, state: dict) -> dict:
    """Write one report per situation plus the run report. Returns what was written."""
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    run_name = out_dir.name

    written = []
    for s in state.get("scenarios") or []:
        path = reports / f"{s['name']}.md"
        path.write_text(scenario_report(s, run_name=run_name), encoding="utf-8")
        written.append(path.name)

    insights = state.get("insights")
    if insights is None:
        candidate = out_dir / "insights.json"
        if candidate.exists():
            insights = json.loads(candidate.read_text(encoding="utf-8")).get("summary")
    summary_path = out_dir / "REPORT.md"
    summary_path.write_text(run_report(state, insights=insights), encoding="utf-8")

    return {"run_report": str(summary_path), "scenario_reports": written}
