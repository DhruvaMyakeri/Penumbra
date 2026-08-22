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

The three ways a prompt fails, and what to do about each:

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

  WRONG THING RENDERED The model produced a different change, or invented objects. Name
                       the surfaces and objects that are ALREADY in the scene and say
                       how they now look. Never phrase anything as adding, placing,
                       replacing or introducing an object, and never mention an object
                       that is not already visible.

Write ONE prompt, under 200 characters, present tense, describing how the scene looks.

Reply with JSON only, no markdown fence:
{{"prompt": "<the rewritten prompt>",
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


def revise_prompt(situation: str, why: str, attempts: list[dict]) -> tuple[str, str]:
    """Rewrite the prompt in light of what the renderer actually did.

    Returns (prompt, reasoning). Never raises: if the agent is unreachable the caller
    keeps the prompt it had, which is the same behaviour as having no agent at all.
    """
    payload = {
        "contents": [{"parts": [{"text": INSTRUCTIONS.format(
            situation=situation or "(none given)",
            why=why or "(none given)",
            history="\n".join(_describe(a) for a in attempts),
            capabilities=CAPABILITIES,
        )}]}],
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
