"""Turn a finished run into one self-contained HTML file.

The dashboard is live and local; this is the artefact you send someone. Images are
inlined as data URIs so the file works with no server, no network and no run directory
beside it.

It deliberately leads with what the run does *not* establish. A findings page that
opens with five green checkmarks and buries the caveats is how a preliminary result
becomes a claim somebody repeats.

    .venv/Scripts/python tools/report.py runs/garage-ep0-... -o report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

CSS = """
:root{--ground:#0B0C0F;--surface:#14161B;--sunk:#191C23;--rule:#272B34;--ink:#EAEAF0;
 --dim:#969CAA;--faint:#69707E;--accent:#4FB3A8;--hit:#E0A050;--hit-dim:#33261A;
 --bad:#DE7477;--bad-dim:#331E1F;--ok:#6FB97E;--ok-dim:#18291B}
*{box-sizing:border-box}
html,body{background:var(--ground)}
body{margin:0;color:var(--ink);font:13px/1.6 "IBM Plex Mono",ui-monospace,monospace}
h1,h2,h3,.name{font-family:"Bricolage Grotesque",system-ui,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:34px 22px 80px}
h1{font-size:24px;letter-spacing:.16em;margin:0 0 4px}
.tagline{color:var(--faint);font-size:11px;letter-spacing:.06em}
.sub{color:var(--dim);margin-top:10px}
h2{font-size:10.5px;letter-spacing:.17em;text-transform:uppercase;color:var(--faint);
 margin:32px 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:1px;
 background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.cell{background:var(--surface);padding:12px 14px}
.cell .k{font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.cell .v{font-size:20px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.card{background:var(--surface);border:1px solid var(--rule);border-left-width:2px;
 border-radius:3px;padding:15px 16px;margin-bottom:12px}
.hit{border-left-color:var(--hit)}.clear{border-left-color:var(--ok)}
.rej{border-left-color:var(--bad)}.plain{border-left-color:var(--rule)}
.name{font-size:15px;font-weight:700}
.tag{font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;padding:3px 8px;
 border-radius:2px;border:1px solid currentColor;float:right}
.t-hit{color:var(--hit);background:var(--hit-dim)}.t-ok{color:var(--ok);background:var(--ok-dim)}
.t-bad{color:var(--bad);background:var(--bad-dim)}
.why{color:var(--dim);font-size:12px;margin:6px 0 10px}
.prompt{color:var(--faint);font-size:11px;background:var(--sunk);padding:8px 10px;
 border-radius:2px;margin-bottom:10px;word-break:break-word}
.verdict{font-size:12.5px;padding:9px 11px;border-radius:2px;margin-bottom:10px}
.v-hit{background:var(--hit-dim);color:var(--hit)}.v-ok{background:var(--sunk);color:var(--dim)}
.v-bad{background:var(--bad-dim);color:var(--bad)}
.intent{font-size:12px;padding:9px 11px;border-radius:2px;margin-bottom:10px;
 background:var(--sunk);border-left:2px solid var(--rule);color:var(--dim)}
.intent.bad{border-left-color:var(--bad)}.intent.ok{border-left-color:var(--ok)}
.metrics{display:flex;gap:18px;flex-wrap:wrap;font-size:11.5px;margin-bottom:10px}
.metrics div{color:var(--faint)}
.metrics b{color:var(--ink);font-variant-numeric:tabular-nums}
img{width:100%;border:1px solid var(--rule);border-radius:2px;display:block;background:#000}
figcaption{color:var(--faint);font-size:10.5px;margin-top:5px}
figure{margin:0}
.caveats{border:1px solid var(--bad);background:var(--bad-dim);border-radius:3px;
 padding:16px 18px;margin:22px 0}
.caveats h3{color:var(--bad);margin:0 0 8px;font-size:15px}
.caveats li{margin-bottom:6px;color:var(--ink)}
.headline{border:1px solid var(--hit);background:var(--hit-dim);border-radius:3px;
 padding:16px 18px;margin:22px 0}
.headline h3{color:var(--hit);margin:0 0 8px;font-size:17px}
footer{color:var(--faint);font-size:11px;margin-top:40px;border-top:1px solid var(--rule);
 padding-top:14px}
"""


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def num(v, d=4) -> str:
    return "—" if v is None else format(float(v), f".{d}f")


def data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def scenario_card(run_dir: Path, s: dict) -> str:
    hit = s.get("is_vulnerability")
    rejected = s["status"] in ("rejected", "error")
    cls = "hit" if hit else ("rej" if rejected else "clear")
    tag = ("t-hit", "vulnerability") if hit else (
        ("t-bad", "render rejected") if rejected else ("t-ok", "no effect"))
    c = s.get("vs_control") or {}
    iv = s.get("intent") or {}

    parts = [f'<div class="card {cls}">',
             f'<span class="tag {tag[0]}">{tag[1]}</span>',
             f'<div class="name">{esc(s["name"])}</div>',
             f'<div class="why">{esc(s.get("situation"))}<br>'
             f'<span style="color:var(--faint)">{esc(s.get("why_it_might_break"))}</span></div>',
             f'<div class="prompt">{esc(s.get("prompt"))}</div>']

    if s.get("verdict") and s["status"] != "queued":
        v = "v-hit" if hit else ("v-bad" if rejected else "v-ok")
        parts.append(f'<div class="verdict {v}">{esc(s["verdict"])}</div>')

    if iv.get("verdict") and iv["verdict"] != "unknown":
        bad = not iv.get("credible", True)
        parts.append(
            f'<div class="intent {"bad" if bad else "ok"}">'
            f'<b>render {esc(iv["verdict"].replace("_", " "))}</b> — '
            f'{esc(iv.get("observed") or iv.get("note"))}'
            f'{" <b>The measured effect stands; the name does not.</b>" if bad else ""}'
            f'<div style="color:var(--faint);font-size:10.5px;margin-top:5px">'
            f'independent vision-model adjudication, not measurement</div></div>')

    bits = []
    if c:
        bits.append(f"<div>vs control p <b>{num(c.get('p_value'))}</b>"
                    f"{' (at design floor)' if c.get('at_resolution_floor') else ''}</div>")
        bits.append(f"<div>effect d <b>{num(c.get('effect_size_cohens_d'), 2)}</b></div>")
        bits.append(f"<div>gripper p <b>{num(c.get('gripper_p_value'))}</b></div>")
    if s.get("gate"):
        bits.append(f"<div>geometry drift <b>{num(s['gate'].get('geometry_drift'), 5)}</b></div>")
    if s.get("rollouts"):
        bits.append(f"<div>rollouts <b>{s['rollouts']}</b></div>")
    if bits:
        parts.append(f'<div class="metrics">{"".join(bits)}</div>')

    if s.get("preview"):
        uri = data_uri(run_dir / s["preview"])
        if uri:
            parts.append(f'<figure><img src="{uri}" alt="every camera, source above '
                         f'perturbed below"><figcaption>every camera — source above, '
                         f'perturbed below</figcaption></figure>')
    parts.append("</div>")
    return "".join(parts)


def build(run_dir: Path) -> str:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    sc = state.get("scenarios", [])
    hits = [s for s in sc if s.get("is_vulnerability")]
    rej = [s for s in sc if s["status"] == "rejected"]
    tested = [s for s in sc if s.get("vs_control")]
    judged = [s for s in sc if (s.get("intent") or {}).get("verdict") not in (None, "unknown")]
    credible = [s for s in judged if s["intent"].get("credible")]
    unnamed = [s for s in hits if (s.get("intent") or {}).get("credible") is False]

    cells = [("Situations", len(sc)), ("Gate passed", len(sc) - len(rej)),
             ("Tested", len(tested)), ("Vulnerabilities", len(hits)),
             ("Renders rejected", len(rej)),
             ("Render matched prompt", f"{len(credible)}/{len(judged)}" if judged else "—"),
             ("Spend", "$" + format(state.get("cost", {}).get("estimated_usd", 0), ".2f"))]

    head = "".join(f'<div class="cell"><div class="k">{k}</div>'
                   f'<div class="v">{v}</div></div>' for k, v in cells)

    if hits:
        items = "".join(
            f"<li><b>{esc(s.get('situation'))}</b> — p={num((s.get('vs_control') or {}).get('p_value'))}, "
            f"d={num((s.get('vs_control') or {}).get('effect_size_cohens_d'), 2)}"
            f"{' · <b>name unsupported by the render</b>' if s in unnamed else ''}</li>"
            for s in hits)
        headline = (f'<div class="headline"><h3>{len(hits)} of {len(tested)} tested '
                    f"situations changed what this policy does</h3>"
                    f"<ul>{items}</ul></div>")
    else:
        headline = ('<div class="card plain"><div class="name">No situation moved this '
                    f"policy</div><div class=\"why\">{len(tested)} situations survived the "
                    "validity gate and were tested against the no-op control; none "
                    "separated from it after suite-wide correction.</div></div>")

    caveats = [
        "<b>No robot failed.</b> The measurement is open-loop action divergence: the "
        "policy is replayed over perturbed pixels and never allowed to act. A changed "
        "action distribution is a necessary condition for a real failure, not a "
        "sufficient one.",
        f"<b>One episode, one window.</b> Everything here is {esc(state.get('episode'))}, "
        "one task, one policy. Nothing about generalisation follows.",
    ]
    if unnamed:
        caveats.append(
            f"<b>{len(unnamed)} of the findings above is named after a situation its "
            f"render does not show.</b> The behaviour change is measured; the stated "
            f"cause is not established.")
    if judged and len(credible) < len(judged):
        caveats.append(
            f"<b>The video editor carried out the prompt in {len(credible)} of "
            f"{len(judged)} renders.</b> A situation's name is a hypothesis about the "
            f"render, not a fact about it.")
    caveats.append(
        "<b>The generative-versus-classical question is open.</b> Whether a free "
        "classical augmentation at matched perceptual distance finds these same "
        "situations has not been settled at suite scale.")

    body = "".join(scenario_card(run_dir, s) for s in sc)

    return f"""<title>PENUMBRA — {esc(run_dir.name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<h1>PENUMBRA</h1>
<div class="tagline">perception chaos engineering for physical AI</div>
<div class="sub">{esc(state.get('task'))} · {esc(state.get('episode'))} · {esc(state.get('policy'))}<br>
cameras perturbed: {esc(', '.join(state.get('views') or []))}
{('· left at true appearance: ' + esc(', '.join(state['views_untouched']))) if state.get('views_untouched') else ''}</div>

<h2>Result</h2>
<div class="grid">{head}</div>
{headline}

<div class="caveats"><h3>Read these first</h3><ul>{''.join(f'<li>{c}</li>' for c in caveats)}</ul></div>

<h2>How a situation earns the word "vulnerability"</h2>
<div class="card plain"><div class="why">
Each situation is rendered onto every perturbed camera, then checked by a validity gate
that rejects renders which moved the scene's geometry rather than its appearance. What
survives is rolled out against the policy and compared — by permutation test on two
statistics — not against the raw recording but against a <b>no-op re-render</b> of the
same episode. The video model redraws every pixel whatever the prompt says, and that
redraw alone shifts this policy; comparing to the no-op cancels it. Anything significant
must then survive Benjamini–Hochberg correction across the whole suite before it is
called a discovery.
</div></div>

<h2>Situations</h2>
{body}

<footer>Generated from {esc(run_dir.name)}. Every number here comes from that run's
<code>state.json</code>; the images are the frames the policy was given. No physical
robot was involved at any point.</footer>
</div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run_dir / "report.html")
    out.write_text(build(args.run_dir), encoding="utf-8")
    size = out.stat().st_size / 1e6
    print(f"written {out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
