"""The failure report.

Designed to be argued with. It states what was held fixed as prominently as what
changed, shows the validity gate's numbers next to the verdict, and never renders a
"BREAK" for a perturbation the gate rejected.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..experiments.record import ExperimentRecord

_CSS = """
:root{--bg:#0c0d10;--panel:#15171c;--line:#262a33;--fg:#e6e8ee;--dim:#8b93a3;
--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--accent:#60a5fa}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1{font-size:22px;letter-spacing:.14em;margin:0 0 4px}
h2{font-size:12px;letter-spacing:.18em;color:var(--dim);text-transform:uppercase;
margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--dim);margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.card .k{color:var(--dim);font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.card .v{font-size:20px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:400;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.dim{color:var(--dim)}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
border:1px solid currentColor}
video{width:100%;border-radius:8px;border:1px solid var(--line);background:#000}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;
overflow-x:auto;font-size:12px;color:var(--dim)}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--panel);
margin:8px 0;color:var(--dim)}
"""


def _fmt(v, digits: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


def write_report(record: ExperimentRecord, directory: Path) -> Path:
    r = record.to_dict()
    ep, pol, fault = r["episode"], r["policy"], r["perturbation"]
    nf, verdict, cost = r["noise_floor"], r["verdict"], r.get("cost", {})
    evals = r["divergence"]["evaluations"]

    margin = verdict.get("margin")
    broke_any = verdict.get("found_break") or verdict.get("broke")
    headline = (
        f'<span class="bad">MARGIN {margin:g}</span>' if margin is not None
        else ('<span class="bad">BREAK</span>' if broke_any
              else '<span class="ok">ROBUST over this ladder</span>')
    )

    rows = []
    for e in evals:
        status = e["status"]
        cls = {"BREAK": "bad", "REJECTED": "warn"}.get(status, "ok")
        g = e.get("group_test") or {}
        v = e.get("validation", {})
        p_txt = _fmt(g.get("p_value"), 4) if g else "&mdash;"
        if g and g.get("p_value") is not None and g.get("min_achievable_p") is not None:
            if g["p_value"] <= g["min_achievable_p"] * 1.001:
                p_txt = f"&le; {g['min_achievable_p']:.4f} (floor)"
        rows.append(
            f"<tr><td>{e['strength']:g}</td>"
            f"<td class='{cls}'>{status}</td>"
            f"<td>{p_txt}</td>"
            f"<td>{_fmt(g.get('gripper_p_value'), 4) if g else '&mdash;'}</td>"
            f"<td>{_fmt(g.get('effect_size_cohens_d'), 2) if g else '&mdash;'}</td>"
            f"<td>{_fmt(g.get('within_group_divergence')) if g else '&mdash;'}</td>"
            f"<td>{_fmt(g.get('between_group_divergence')) if g else '&mdash;'}</td>"
            f"<td>{_fmt(v.get('geometry_drift'), 5)}"
            f"<span class='dim'> ({_fmt(v.get('geometry_drift_pixels_equivalent'), 1)} px)</span></td>"
            f"<td>{_fmt(v.get('structure_retained'), 3)}</td>"
            f"<td>{_fmt(e['distance'].get('rmse'))}</td>"
            f"<td class='dim'>{html.escape(str(e['perturbation'].get('prompt', ''))[:60])}</td></tr>"
        )

    classical_html = ""
    if r.get("classical_control"):
        crows = []
        for e in r["classical_control"]:
            g = e.get("group_test") or {}
            crows.append(
                f"<tr><td>{html.escape(str(e['perturbation'].get('op')))}</td>"
                f"<td>{_fmt(e['perturbation'].get('strength'), 3)}</td>"
                f"<td class=\"{'bad' if e['status']=='BREAK' else 'ok'}\">{e['status']}</td>"
                f"<td>{_fmt(g.get('p_value'), 4) if g else '&mdash;'}</td>"
                f"<td>{_fmt(g.get('gripper_p_value'), 4) if g else '&mdash;'}</td>"
                f"<td>{_fmt(g.get('effect_size_cohens_d'), 2) if g else '&mdash;'}</td>"
                f"<td>{_fmt(e['distance'].get('rmse'))}</td></tr>"
            )
        gen_row = next((e for e in evals if e["status"] != "REJECTED"), None)
        gen_g = (gen_row or {}).get("group_test") or {}
        gen_line = (
            f"<tr><td><strong>{html.escape(fault['name'])} (generative)</strong></td>"
            f"<td>{_fmt((gen_row or {}).get('strength'), 2)}</td>"
            f"<td class=\"{'bad' if (gen_row or {}).get('status')=='BREAK' else 'ok'}\">"
            f"{(gen_row or {}).get('status','&mdash;')}</td>"
            f"<td>{_fmt(gen_g.get('p_value'), 4) if gen_g else '&mdash;'}</td>"
            f"<td>{_fmt(gen_g.get('gripper_p_value'), 4) if gen_g else '&mdash;'}</td>"
            f"<td>{_fmt(gen_g.get('effect_size_cohens_d'), 2) if gen_g else '&mdash;'}</td>"
            f"<td>{_fmt(((gen_row or {}).get('distance') or {}).get('rmse'))}</td></tr>"
        ) if gen_row else ""
        classical_html = f"""
<h2>Classical augmentation control arm — matched perceptual distance</h2>
<p class="dim">Scene-blind pixel transforms, calibrated to the same RMSE distance from
the source as the generative perturbation, and given the same number of policy rollouts
and the same permutation test. If these move the policy too, the generative machinery
bought nothing, and the honest answer is that it bought nothing.</p>
<table><tr><th>arm</th><th>strength</th><th>result</th><th>p (joint)</th>
<th>p (gripper)</th><th>Cohen&rsquo;s d</th><th>rmse</th></tr>
{gen_line}{''.join(crows)}</table>"""

    notes = verdict.get("notes") or []
    notes_html = "".join(f'<div class="note">{html.escape(n)}</div>' for n in notes)

    # Did the intervention actually happen? A null result from a perturbation that
    # never landed is a statement about the prompt, not the policy, and the report
    # must say so rather than let the reader assume otherwise.
    landed_html = ""
    check_path = directory / "removal_check.json"
    if check_path.exists():
        try:
            chk = json.loads(check_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            chk = None
        if chk:
            landed = bool(chk.get("intervention_landed"))
            cls = "ok" if landed else "bad"
            verdict_txt = ("the intervention landed" if landed
                           else "the intervention did NOT land")
            landed_html = f"""
<h2>Did the intervention actually happen?</h2>
<div class="note"><strong class="{cls}">{html.escape(verdict_txt.upper())}</strong> &mdash;
{html.escape(str(chk.get('interpretation', '')))}</div>
<table>
<tr><th>Target-coloured pixels surviving, pointer region</th>
<td class="{cls}">{_fmt(chk.get('surviving_fraction_in_region'), 3)}</td></tr>
<tr><th>Target-coloured pixels surviving, whole frame</th>
<td>{_fmt(chk.get('surviving_fraction_total'), 3)}</td></tr>
<tr><th>Mean target pixels per frame</th>
<td>{_fmt(chk.get('mean_target_pixels_per_frame_source'), 1)} source &rarr;
{_fmt(chk.get('mean_target_pixels_per_frame_perturbed'), 1)} perturbed</td></tr>
<tr><th>Pointer</th><td class="dim">{html.escape(json.dumps(chk.get('pointer')))}</td></tr>
</table>"""

    videos = []
    for name, label in [("side_by_side.mp4", "Source vs perturbed"),
                        ("source.mp4", "Source"), ("perturbed.mp4", "Perturbed")]:
        if (directory / name).exists():
            videos.append(f'<div><div class="k dim">{label}</div>'
                          f'<video src="{name}" controls loop muted></video></div>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PENUMBRA {html.escape(record.run_id)}</title><style>{_CSS}</style></head><body>
<h1>PENUMBRA</h1>
<div class="sub">Visual fault injection for physical AI &middot; {html.escape(record.run_id)}
&middot; {html.escape(record.created_at)}</div>

<div class="grid">
  <div class="card"><div class="k">Result</div><div class="v">{headline}</div></div>
  <div class="card"><div class="k">Fault</div><div class="v">{html.escape(fault['name'])}
    <span class="dim" style="font-size:12px">v{html.escape(fault['version'])}</span></div></div>
  <div class="card"><div class="k">Policy</div><div class="v" style="font-size:14px">
    {html.escape(pol['model'])}</div></div>
  <div class="card"><div class="k">Noise floor &sigma;</div><div class="v">
    {_fmt(nf['joint_l2_std'], 5)}</div>
    <div class="dim" style="font-size:12px">mean {_fmt(nf['joint_l2_mean'],5)} over
    {nf['pairs']} pairs of {nf['n_runs']} runs</div></div>
  <div class="card"><div class="k">Cost</div><div class="v">${cost.get('estimated_usd','?')}</div>
    <div class="dim" style="font-size:12px">{cost.get('total_session_seconds','?')}s of session
    at ${cost.get('usd_per_hour','?')}/hr</div></div>
</div>

<h2>What was tested</h2>
<table>
<tr><th>Episode</th><td>{html.escape(ep['episode_id'])} &middot; {ep['frames']} frames @ {ep['fps']} fps</td></tr>
<tr><th>Source</th><td class="dim">{html.escape(ep['source'])}</td></tr>
<tr><th>Task</th><td>{html.escape(ep['task'].split('|')[0].strip())}</td></tr>
<tr><th>Changed</th><td class="warn">{html.escape(ep['perturbed_view'])} &mdash; appearance only, via {html.escape(fault['model'])}</td></tr>
<tr><th>Held fixed</th><td class="ok">{html.escape(', '.join(ep['held_fixed']))}</td></tr>
<tr><th>Protocol</th><td class="dim">{html.escape(pol['protocol'])}</td></tr>
</table>

<h2>Strength ladder</h2>
<p class="dim">X2 exposes no numeric strength parameter, so strength is an ordered
ladder of prompts constructed by PENUMBRA. The axis is ordinal by construction;
monotonicity of the response is reported, not assumed.</p>
<table>
<tr><th>strength</th><th>result</th><th>p (joint)</th><th>p (gripper)</th><th>Cohen&rsquo;s d</th>
<th>within-group L2</th><th>between-group L2</th>
<th>geometry drift</th><th>structure</th><th>rmse</th><th>prompt</th></tr>
{''.join(rows)}
</table>
{notes_html}
{landed_html}
{classical_html}

<h2>Why this counts</h2>
<table>
<tr><th>Validity gate</th><td>{html.escape(r['validation']['gate'])} &mdash; runs
<em>before</em> the policy, so a corrupted transformation costs a rejection, not a false finding.
Thresholds are measured, not inherited: an appearance-only X2 edit scores 0.0015 drift,
a 4&nbsp;px geometric shift scores 0.0055</td></tr>
<tr><th>Evidence unit</th><td>A <em>group</em> of rollouts, not a rollout. The policy is
stochastic &mdash; unperturbed runs differ from each other by {_fmt(nf['joint_l2_mean'],4)} rad on
average &mdash; so significance comes from a permutation test over group labels, not from a
single pairwise comparison</td></tr>
<tr><th>Noise floor</th><td>{nf['n_runs']} unperturbed rollouts, {nf['pairs']} pairwise
comparisons, mean {_fmt(nf['joint_l2_mean'],5)} rad, &sigma; = {_fmt(nf['joint_l2_std'],5)}.
Unperturbed gripper decisions already disagree on up to
{_fmt(nf['gripper_flip_rate_max'],3)} of steps</td></tr>
<tr><th>Reproducibility</th><td class="warn">X2 has no seed. The same rung run twice does
not produce the same video &mdash; in this project one <code>specular_floor</code> run at
strength 1.0 passed the gate (temporal ratio 1.92) and another was rejected (4.14). Any
margin is the lowest break observed, not a deterministic threshold</td></tr>
</table>

<h2>Limitations</h2>
<ul class="dim">
<li>This measures <strong>open-loop action divergence</strong>, not task success. The
policy's actions never move the robot, because the robot is a recording. That is a
deliberate design choice: holding the trajectory fixed is what isolates perception
sensitivity from control error. It is not evidence of closed-loop failure.</li>
<li>The strength axis is prompt-constructed and ordinal. Distances between rungs are
not physically meaningful; the measured perceptual distance column is what to compare across faults.</li>
<li>The validity gate detects displacement, not stationary hallucination. Content
cleanly edited into or out of a static region can pass it.</li>
<li>Sample size is {ep['frames']} frames of <strong>one</strong> episode with
{nf['n_runs']} rollouts per condition. Group sizes this small put a hard floor under the
attainable p-value, which is reported next to every p. This is a demonstration of method,
not a robustness benchmark, and no per-fault number here should be quoted as a property
of the policy in general.</li>
</ul>

<h2>Evidence</h2>
<div class="grid">{''.join(videos)}</div>

<h2>Reproduce</h2>
<pre>{html.escape(json.dumps({'environment': r['environment'],
                          'fault': fault['name'] + ' v' + fault['version'],
                          'episode': ep['episode_id'],
                          'policy': pol['model']}, indent=2))}</pre>
</body></html>"""

    path = directory / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
