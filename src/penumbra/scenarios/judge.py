"""Did the render actually produce the situation it was asked for?

The gap this closes is attribution. A suite card that says *"amber safety lighting moved
this policy"* is only entitled to the words "amber safety lighting" if the render is
amber safety lighting. X2 has repeatedly done something other than what the prompt
asked: told to remove a cup it made the cup translucent and the target-coloured pixel
count went *up*; told to insert a box it never drew one and amplified the cup four to
five times instead. Without a check, every situation name in the suite is an
aspiration.

So a vision model that had no part in generating the render is shown the same frame
before and after and asked one narrow question: was this change applied, and if not,
what changed instead? It is deliberately shown the frames, not the prompt's intent
dressed up - and it is asked to describe what it sees before it judges, so the verdict
is anchored to the image rather than to the request.

**This is adjudication, not measurement.** A vision model's opinion is evidence about
the render, not ground truth, and it can be wrong in both directions. It is recorded
next to the statistics so a reader can discount a finding whose render the judge says
never happened - never to promote one. Nothing in the pipeline gates on it, and a
scenario is never called a vulnerability because the judge liked it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

from .director import MODELS, _call, _frame_to_b64

log = logging.getLogger("penumbra.judge")

#: Verdicts the judge may return, strongest first.
VERDICTS = ("applied", "partially_applied", "not_applied", "something_else")

INSTRUCTIONS = """You are auditing a video editing model's output for a robotics
experiment. Two frames from the same moment of the same robot recording are given: the
ORIGINAL first, then the EDITED version.

The editor was asked for this change:

  "{prompt}"

The situation it was meant to depict:

  "{situation}"

First describe, in one sentence, what visibly differs between the two frames. Base that
only on what you can see. Then judge whether the requested change was carried out.

Be strict and be willing to say no. Editors routinely produce a different change from
the one requested, or re-render the scene with no meaningful change at all. Reporting
"applied" for a render that merely looks slightly different is the failure mode that
matters here.

Also list any OBJECT that appears in the edited frame and is absent from the original
and was not requested. This editor's characteristic failure is inventing things - extra
cups, toys, hands, whole robots - and an invented object is very likely what a policy
looking at this frame would react to, so it matters more than the styling.

Reply with JSON only, no markdown fence:

{{"observed": "<one sentence on what actually differs>",
  "verdict": "applied" | "partially_applied" | "not_applied" | "something_else",
  "added_objects": ["<each unrequested object that appeared, or empty>"],
  "confidence": <0.0-1.0>,
  "note": "<one sentence justifying the verdict; if something_else, say what happened>"}}
"""


@dataclass
class IntentVerdict:
    """One adjudication of one render."""

    verdict: str
    observed: str = ""
    note: str = ""
    confidence: float = 0.0
    model: str = ""
    error: str | None = None
    views_judged: list = field(default_factory=list)
    #: Objects the render invented that nobody asked for. X2's characteristic failure,
    #: and the most likely thing a policy is actually reacting to.
    added_objects: list = field(default_factory=list)

    @property
    def credible(self) -> bool:
        """Did the render plausibly produce the situation its name claims?

        `partially_applied` counts: a partial glare is still glare. `not_applied` and
        `something_else` do not, and a scenario carrying either should have its *name*
        distrusted even when its statistics are sound.
        """
        return self.verdict in ("applied", "partially_applied")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "observed": self.observed,
            "note": self.note,
            "confidence": round(self.confidence, 2),
            "model": self.model,
            "credible": self.credible,
            "added_objects": self.added_objects,
            "hallucinated": bool(self.added_objects),
            "views_judged": self.views_judged,
            "error": self.error,
            "caveat": ("vision-model adjudication, not measurement. Recorded to let a "
                       "reader discount a finding, never to promote one; nothing in the "
                       "pipeline gates on it."),
        }


def _parse(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge output: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def judge_render(
    prompt: str,
    situation: str,
    source: np.ndarray,
    perturbed: np.ndarray,
    *,
    view: str = "",
    frame: int | None = None,
) -> IntentVerdict:
    """Adjudicate one camera's render. Never raises: a failed judge is not a failed run."""
    n = min(len(source), len(perturbed))
    if n == 0:
        return IntentVerdict(verdict="not_applied", error="no frames")
    # The middle of the window: the first frames of a streaming edit are the model
    # warming up, and the last can be truncated by the alignment.
    i = n // 2 if frame is None else min(frame, n - 1)

    payload = {
        "contents": [{"parts": [
            {"text": INSTRUCTIONS.format(prompt=prompt, situation=situation or prompt)},
            {"text": "ORIGINAL:"},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _frame_to_b64(source[i])}},
            {"text": "EDITED:"},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _frame_to_b64(perturbed[i])}},
        ]}],
        # Generous, because this model emits reasoning tokens before its answer and a
        # tight cap truncates the JSON mid-string - which reads as a parse failure
        # rather than as the budget problem it is.
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }
    try:
        raw = _call(payload, timeout=90.0, attempts=3)
        doc = _parse(raw)
    except Exception as exc:  # noqa: BLE001
        log.info("intent judge failed: %s", exc)
        return IntentVerdict(verdict="unknown", error=f"{type(exc).__name__}: {exc}",
                             views_judged=[view] if view else [])

    verdict = str(doc.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        verdict = "unknown"
    return IntentVerdict(
        verdict=verdict,
        observed=str(doc.get("observed", ""))[:400],
        added_objects=[str(o)[:80] for o in (doc.get("added_objects") or [])][:8],
        note=str(doc.get("note", ""))[:400],
        confidence=float(doc.get("confidence", 0.0) or 0.0),
        model=MODELS[0],
        views_judged=[view] if view else [],
    )


def judge_views(
    prompt: str,
    situation: str,
    source_frames: dict,
    perturbed_frames: dict,
    views,
) -> IntentVerdict:
    """Adjudicate every perturbed camera and return the *weakest* verdict.

    Weakest, not best: a situation is only credible if it is visible on the cameras the
    policy is actually reading. Taking the best camera would let one lucky render
    launder a name across the rest.
    """
    order = {v: i for i, v in enumerate(VERDICTS)}
    verdicts = []
    for v in views:
        if v not in source_frames or v not in perturbed_frames:
            continue
        verdicts.append(judge_render(prompt, situation, source_frames[v],
                                     perturbed_frames[v], view=v))
    if not verdicts:
        return IntentVerdict(verdict="unknown", error="no views to judge")
    worst = max(verdicts, key=lambda r: order.get(r.verdict, len(VERDICTS)))
    worst.views_judged = [v for r in verdicts for v in r.views_judged]
    # Union, not the worst camera's list: an object invented on any camera the policy
    # reads is in the policy's input, whatever the other cameras show.
    seen = []
    for r in verdicts:
        for o in r.added_objects:
            if o.lower() not in {x.lower() for x in seen}:
                seen.append(o)
    worst.added_objects = seen
    if len(verdicts) > 1:
        agree = sum(1 for r in verdicts if r.verdict == worst.verdict)
        worst.note = f"{worst.note} ({agree}/{len(verdicts)} cameras agree)".strip()
    return worst
