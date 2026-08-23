"""An agent that keeps working on a generation until it is worth testing.

Every suite so far has wasted most of its situations. One run had 7 of 18 renders come
back with no visible change; another had 12 of 21 rejected by the validity gate; across
46 judged renders, 19 contained objects nobody asked for. Each of those consumed a
render, and worse, each one counted toward the suite's multiplicity correction - so a
wasted scenario makes every *real* candidate harder to confirm.

The loop here closes that. After each render the generation faces two independent
checks that already exist:

    the validity gate   did the scene survive, geometrically and temporally
    the intent judge    does a vision model agree this is the situation asked for,
                        and did the model invent anything

If either says no, the failure - in its own words - is handed back to a language model,
which rewrites the *prompt* and only the prompt. The situation being tested never
changes; what changes is the wording used to realise it. That distinction is the whole
of the method's honesty: rewriting the situation until something breaks would be
fishing, while rewriting the phrasing until the renderer cooperates is instrument
operation.

### What this does to the statistics, stated plainly

This is selection. Renders that survive are drawn from the distribution of X2 outputs
*conditional on being admissible and on-target*, not from its unconditional output. That
is the intended estimand - "does an admissible rendering of this situation move the
policy" - and it is what the gate has always implied. But it is selection, every attempt
is recorded in `render_attempts`, and a reader should treat the attempt count as part of
the result: a situation that took three tries to render is weaker evidence than one that
worked immediately.

The gate is never consulted for how to pass it. The agent sees only the failure reason,
the same text a human would read.
"""
from __future__ import annotations

import json
import logging

from .director import CAPABILITIES, _call
from .insights import rule_violations

log = logging.getLogger("penumbra.render_agent")

INSTRUCTIONS = """A video-editing model is being used to render a test situation onto a
robot camera recording. The situation is fixed. Your job is to rewrite the PROMPT so the
model renders it acceptably.

THE SITUATION (do not change what is being tested):
  {situation}

WHY IT MIGHT BREAK THE POLICY:
  {why}

WHAT HAS BEEN TRIED, AND WHAT WENT WRONG:
{history}

{capabilities}

The four ways a prompt fails, and what to do about each:

  NOTHING HAPPENED     The model returned the scene essentially unchanged. The prompt
                       was too mild or too abstract. Make it drastic and concrete: name
                       a specific colour, material, or covering, and describe it as
                       already present and unmistakable.

  SCENE CORRUPTED      The gate found the scene had moved, warped, or gained motion the
                       recording never had. The prompt asked for something the model can
                       only render by redrawing the scene. Keep the same situation but
                       ask for a STILLER version of it - a wet surface rather than
                       flowing liquid, a coating rather than a cloud, a stain rather
                       than a splash - and avoid anything that implies movement, steam,
                       droplets, or particles in the air.

  WRONG THING RENDERED The model produced a different change, or invented objects. The
                       single most effective repair is measured and mechanical: CUT THE
                       NUMBER OF NAMED OBJECTS TO ONE. Choose the one object the
                       situation is really about and describe only that. Do not mention
                       the others at all, even in passing, even as scenery - naming a
                       second object doubles the chance the model draws a duplicate.

  CAMERAS DISAGREED    Two cameras were rendered in separate passes and produced
                       different scenes - one saw a spill, the other saw a new object,
                       or one saw nothing. The prompt left too much to the model's
                       imagination, so the two draws went different ways. Repair it by
                       removing choice: one object, one named colour or material, one
                       sentence, no adjectives that could be interpreted several ways,
                       nothing that implies an object the scene does not already have.
                       A prompt that admits only one reading is one both passes can
                       agree on.

THE RULES THAT ARE NOT NEGOTIABLE - each is measured over 92 renders:

  1. Name AT MOST ONE manipulable object (cup, bowl, gripper, container). One object:
     36% chance of invention. Two: 73%. Three: 100%.
  2. Never target a rim, edge, lip, handle, gripper finger, or the chips inside a cup.
     0 of 10 such prompts produced a usable render. Whole objects and whole surfaces
     only.
  3. Never describe the camera, the lens, the frame or the image. This model edits the
     scene, not the optics.

Write ONE prompt, under 200 characters, present tense, describing how the scene looks.

Reply with JSON only, no markdown fence:
{{"prompt": "<the rewritten prompt>",
  "objects_named": ["<each manipulable object your prompt names - aim for exactly one>"],
  "reasoning": "<one sentence on what you changed and why>"}}
"""


def _describe(attempt: dict) -> str:
    """One attempt, as the agent sees it: what was asked, and what came back."""
    outcome = attempt.get("outcome", "unknown")
    lines = [f"  attempt {attempt.get('attempt', '?')}: \"{attempt.get('prompt', '')}\"",
             f"    result: {outcome}"]
    if attempt.get("detail"):
        lines.append(f"    detail: {attempt['detail']}")
    return "\n".join(lines)


def _ask(situation: str, why: str, attempts: list[dict], brief: str,
         correction: str = "", extra: str = "") -> tuple[str, str]:
    """One call to the agent. Returns (prompt, reasoning); ("", reason) on failure."""
    text = INSTRUCTIONS.format(
        situation=situation or "(none given)",
        why=why or "(none given)",
        history="\n".join(_describe(a) for a in attempts),
        capabilities=CAPABILITIES,
    )
    if extra:
        # A per-scenario constraint from the caller. It goes ABOVE the general
        # advice and says so, because the two genuinely conflict: the general
        # advice pushes every prompt toward one manipulable object, and a
        # scene-only null control must name none. Without this the agent
        # helpfully "fixed" three null controls into object prompts and the
        # control stopped being a control.
        text += ("\n\nCONSTRAINT ON THIS SCENARIO - it overrides the "
                 "general advice above wherever the two conflict:\n" + extra)
    if brief:
        text += f"\n\nWHAT THIS RUN HAS LEARNED ABOUT THE RENDERER SO FAR:\n{brief}"
    if correction:
        text += (f"\n\nYOUR PREVIOUS ANSWER BROKE A RULE AND WAS NOT USED:\n"
                 f"{correction}\nWrite it again, obeying every rule this time.")
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
    }
    try:
        raw = _call(payload, timeout=90.0, attempts=3)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        doc = json.loads(cleaned[cleaned.find("{"): cleaned.rfind("}") + 1])
        prompt = str(doc.get("prompt", "")).strip()[:1000]
        if not prompt:
            raise ValueError("agent returned an empty prompt")
        return prompt, str(doc.get("reasoning", ""))[:300]
    except Exception as exc:  # noqa: BLE001
        log.info("render agent could not revise the prompt: %s", exc)
        return "", f"{type(exc).__name__}: {exc}"


def revise_prompt(situation: str, why: str, attempts: list[dict],
                  brief: str = "", extra_rules: str = "",
                  validate=None) -> tuple[str, str]:
    """Rewrite the prompt in light of what the renderer actually did.

    Returns (prompt, reasoning). Never raises: if the agent is unreachable the caller
    keeps the prompt it had, which is the same behaviour as having no agent at all.

    The agent's answer is **checked against the measured rules before it is used**, and
    a violating prompt is sent back once with the specific violation quoted. Telling a
    model a rule and then not enforcing it is how the last suite ended up with eleven
    two-object prompts despite the guidance already saying not to: the instruction was
    there, nothing verified it, and a spent render is not the place to discover the
    rule was ignored. If the second answer still violates, the better of the two is
    returned with the violation recorded, because a rule-breaking prompt that renders
    is still worth more than no attempt at all.
    """
    check = validate or rule_violations
    prompt, reasoning = _ask(situation, why, attempts, brief, extra=extra_rules)
    if not prompt:
        return "", reasoning

    violations = check(prompt)
    if not violations:
        return prompt, reasoning

    retry, retry_reasoning = _ask(
        situation, why, attempts, brief,
        correction="\n".join(f"  - {v}" for v in violations),
    )
    if retry and not rule_violations(retry):
        return retry, f"{retry_reasoning} (rewritten after breaking: {violations[0]})"

    kept, kept_reason = (retry, retry_reasoning) if retry else (prompt, reasoning)
    remaining = check(kept)
    log.info("render agent prompt still breaks %d rule(s): %s",
             len(remaining), "; ".join(remaining)[:160])
    return kept, f"{kept_reason} [WARNING: still breaks {len(remaining)} measured rule(s)]"


def classify(gate: dict | None, intent: dict | None, declined: bool) -> tuple[str, str]:
    """Turn the two checks into one outcome the agent can act on.

    Returns (outcome, detail). `outcome` is "usable" when the render is worth testing.
    """
    if declined:
        return ("NOTHING HAPPENED",
                "the render was within the model's own no-op variation - visually "
                "indistinguishable from asking it for nothing")
    if gate is not None and not gate.get("valid", True):
        return "SCENE CORRUPTED", "; ".join(gate.get("reasons") or ["gate rejected it"])
    if intent:
        # Checked before the verdict, because incoherence is the more fundamental
        # failure: when the cameras disagree there is no single render to have an
        # opinion about, and "applied" on the luckier camera is not the situation the
        # policy was shown.
        if intent.get("coherent") is False:
            bits = [intent.get("coherence_note") or "the cameras rendered different scenes"]
            for view, d in (intent.get("per_view") or {}).items():
                bits.append(f"{view}: {d.get('observed', '')}")
            return "CAMERAS DISAGREED", " | ".join(b for b in bits if b)
        verdict = intent.get("verdict")
        if verdict in ("not_applied", "something_else"):
            bits = [f"a vision model judged the render '{verdict}'"]
            if intent.get("observed"):
                bits.append(f"what it saw: {intent['observed']}")
            if intent.get("added_objects"):
                bits.append("objects invented that nobody asked for: "
                            + ", ".join(intent["added_objects"]))
            return "WRONG THING RENDERED", ". ".join(bits)
    return "usable", ""


#: Which outcomes a rewritten prompt can plausibly repair. `SCENE CORRUPTED` is here
#: because a stiller phrasing genuinely fixes it; `CAMERAS DISAGREED` because a prompt
#: admitting only one reading gives two independent draws less room to diverge.
REPAIRABLE = ("NOTHING HAPPENED", "SCENE CORRUPTED", "WRONG THING RENDERED",
              "CAMERAS DISAGREED")
