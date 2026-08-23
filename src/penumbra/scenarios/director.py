"""The scenario director: an LLM that proposes situations the robot might fail in.

PENUMBRA's first version tested *cosmetic* perturbations - wetter floor, harsher glare -
and got a degenerate answer: passing the video through a generative model at all moves
the policy, whatever the prompt asked for. That is a fact about the video model, not a
vulnerability of the robot. It names no situation, so nobody can act on it.

What a robotics team actually needs is the opposite shape:

    "Your policy mistimes the pour when a second, similar cup is on the table.
     Here is the video. Here is the statistical test. Here is the regression case."

A situation, a measured consequence, and evidence. That requires something that
understands what the robot is *trying to do* and what could plausibly go wrong while
it does it - which is what this module is. It reads the task instruction and looks at
the actual first frame of the episode, then proposes concrete, physically plausible
situations, each with the reason it might break this particular policy.

The director proposes; it never judges. Whether a scenario actually changes behaviour
is decided by the permutation test against the no-op control, not by the model that
invented it. An LLM's belief that something "should" confuse a robot is a hypothesis
generator, and is treated as one.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import gemini_api_key
from .insights import prompt_features, rule_violations

log = logging.getLogger("penumbra.director")

#: Tried in order. The embodied-reasoning model leads because this task is exactly
#: what it is for - reading a workspace and reasoning about physical interaction -
#: and the rest are fallbacks for when a shared endpoint is saturated. Model
#: availability on this key was measured, not assumed: `gemini-2.5-flash` 404s and
#: `gemini-flash-latest` returned 503 under load.
MODELS = (
    "gemini-robotics-er-2-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
)
MODEL = MODELS[0]
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: What the perturbation runtime can actually express. The director is told this
#: verbatim, because a scenario it cannot render is a scenario that wastes a run.
CAPABILITIES = """
You are writing prompts for a real-time video-to-video model (X2) that re-renders an
existing robot camera recording. VERIFIED capabilities and limits, measured, not guessed:

CAN DO:
- change surface appearance, lighting, colour, reflectivity, weather, haze
- change the appearance of objects already present in the scene
- add diffuse, atmospheric or surface-level content (spills, stains, glare, steam)
- alter how strongly an object stands out from its background

CANNOT DO (measured failures, do not propose these):
- insert a NEW discrete object that was not there (tested: a reference image of a
  cardboard box produced no box in any view, across three runs)
- remove an object (tested: asking to remove a cup made it translucent, still present)
- move objects to new positions
- change the robot's motion - the trajectory is a fixed recording

THE FAILURE MODE THAT RUINS EXPERIMENTS - and the measured rule that controls it:

This model invents objects. Told to make the cup reflective, it draws a SECOND, shiny
cup beside the first and leaves the original alone. Across 92 judged renders it invented
something unrequested in 58% of them: duplicate cups, extra bowls, a toy robot, a glass
goblet, a human hand. An invented object is almost certainly what the policy would react
to, so the experiment then measures "a strange object appeared" rather than the situation
you described, and the finding has to be thrown out.

The rate is governed almost entirely by ONE property of the prompt - how many
manipulable objects it names:

    objects named    renders showing what was asked    invented an object
        0                      53%                            40%
        1                      29%                            36%   <- use this
        2                      18%                            73%
        3 or more              20%                           100%

Naming a second object roughly DOUBLES the invention rate. Every prompt that named three
invented something.

RULE 1 - EVERY PROMPT NAMES AT MOST ONE MANIPULABLE OBJECT.
One cup, or one bowl, or the gripper - never two, never "the cup and the bowl", never
"both containers". If your situation is really about two objects, pick the one that
matters and write only that. Do not mention the other even in passing, even as scenery.

RULE 2 - NEVER TARGET A PART OF AN OBJECT.
Rims, edges, lips, handles, gripper fingers, "the chips inside the cup", "the inner
wall". Ten prompts did this; ZERO produced a usable render. This model works at the
scale of a whole object or a whole surface. Below that it substitutes something else.

RULE 3 - NEVER DESCRIBE THE CAMERA, LENS, FRAME OR IMAGE.
It edits the scene, not the optics. A "smudged lens" prompt gets you a new object on the
table instead.

DO NOT OVERCORRECT INTO BLANDNESS:
A run written as bare, mild scene descriptions produced NO visible change at all in 7 of
18 renders - the model declined and returned the scene as it was, which wastes the
scenario just as thoroughly. Within the rules above, be VIVID AND EXTREME: name a
specific colour, material or covering and state it as already present and unmistakable.
"The bowl is filled to the brim with dense fluffy white foam" is right - one object,
whole object, unambiguous. "The cup looks a bit wet" is not.

AND THE TWO CAMERAS MUST AGREE:
Each camera is rendered in a SEPARATE pass with no shared seed, so the two passes can
easily produce different scenes - and measured on the last suite, 19 of 21 renders did.
When the cameras disagree the policy is shown two contradictory worlds and the result is
unusable no matter how large the effect. A prompt that admits only one interpretation is
one both passes can agree on: one object, one named colour or material, one sentence, no
vague adjectives, nothing implying an object that is not already in the scene.

WHERE THIS POLICY IS ACTUALLY SENSITIVE - measured over 46 tested situations:

    perturbation targets            median effect   situations that moved the policy
    the manipulated objects              +0.53                4 of 10
    objects and scene together           +0.60                9 of 25
    the scene alone (table, lighting)    -0.01                0 of 9

Changing only the background - tabletop colour, room lighting, wall texture, floor -
has never once moved this policy. Nine attempts, median effect zero. A suite made of
scene-only situations returns nothing, and that has already happened once.

The appearance of the OBJECT THE ROBOT IS MANIPULATING is where the sensitivity lives:
the cup, or the bowl, or their contents, or the gripper - one of them per situation, as
a whole object, per Rules 1 and 2. Aim most situations there. Keep a couple of
scene-only ones as a check on the null above - it should keep being tested, not
assumed - but do not build the suite out of them.

Note the trap in combining that with Rule 1: the sensitivity lives in the objects, and
the renderer breaks when you name more than one of them. The situations that work are
therefore narrow and deep - ONE object, transformed drastically - not broad scene
dressing. "The bowl's interior is filled with glossy black liquid" beats "the workspace
is wet and grimy" for finding something, and beats "the cup and bowl are both greasy"
for actually rendering.

The camera geometry, the robot's motion, and the physics all come from a real
recording and cannot change. Only how the scene LOOKS can change.
"""

SYSTEM = """You are a robotics test engineer designing failure scenarios for a
vision-based manipulation policy. You are adversarial but physically honest: every
scenario must be a situation that could genuinely occur in a real workspace.

For each scenario give:
- name: short snake_case identifier
- situation: one sentence describing the real-world situation, as an operator would
  describe it ("someone spilled coffee on the workbench")
- why_it_might_break: the specific mechanism by which this could mislead a policy
  that maps pixels to arm motion. Be concrete about what visual cue is corrupted.
- prompt: the literal edit instruction for the video model. Describe the SCENE as it
  would appear, not the edit operation. Concrete and visual.
- severity: 1-5, how badly you expect it to affect the policy
- realism: 1-5, how likely this is to actually happen in a real deployment

Prioritise scenarios that attack the specific visual cues THIS task depends on.
Avoid generic "make it darker" scenarios unless there is a task-specific reason.
Return ONLY a JSON array. No prose, no markdown fence."""


@dataclass
class Scenario:
    """One proposed situation, before it has been tested."""

    name: str
    situation: str
    why_it_might_break: str
    prompt: str
    severity: int = 3
    realism: int = 3
    source: str = MODEL
    category: str = "general"
    #: Measured rules this prompt still breaks after the repair pass. Empty is normal;
    #: a non-empty list is a warning attached to the proposal, not a reason to drop it.
    rule_violations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "situation": self.situation,
            "why_it_might_break": self.why_it_might_break,
            "prompt": self.prompt,
            "severity": self.severity,
            "realism": self.realism,
            "source": self.source,
            "category": self.category,
            "rule_violations": self.rule_violations,
        }

    def to_fault_doc(self, views: tuple[str, ...]) -> dict:
        """A FaultSpec-shaped document, so a scenario is a first-class fault."""
        return {
            "name": self.name,
            "version": "director-1",
            "family": "scenario",
            "model": "xmax/x2",
            "description": f"{self.situation}\n\nWhy it might break the policy: "
                           f"{self.why_it_might_break}",
            "ladder": [
                {"strength": 0.0, "label": "untouched", "prompt": ""},
                {"strength": 1.0, "label": self.name, "prompt": self.prompt},
            ],
            "validation": {"max_geometry_drift": 0.0055, "min_structure_retained": 0.20},
            "scenario": self.to_dict(),
            "views": list(views),
        }


@dataclass
class DirectorResult:
    scenarios: list[Scenario] = field(default_factory=list)
    raw: str = ""
    error: str | None = None
    #: One line per proposal whose prompt broke a measured rule, saying whether the
    #: repair pass fixed it. Surfaced in the run record so a reader can see how often
    #: the director had to be corrected.
    repairs: list = field(default_factory=list)


def _frame_to_b64(frame: np.ndarray, width: int = 640) -> str:
    h, w = frame.shape[:2]
    scaled = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("could not encode frame")
    return base64.b64encode(buf.tobytes()).decode()


#: Transient upstream states worth retrying rather than failing the run for.
_RETRYABLE = {429, 500, 502, 503, 504}


def _call(payload: dict, timeout: float = 120.0, attempts: int = 4) -> str:
    """POST to the director model, retrying transient upstream failures.

    The director sits on the critical path of every suite: a 503 here would waste the
    whole run, and shared-capacity endpoints return them routinely. Backoff is
    exponential and bounded; a non-retryable status fails immediately, because
    retrying a 400 just burns time.
    """
    last: Exception | None = None
    models = list(MODELS)
    for attempt in range(attempts):
        model = models[min(attempt, len(models) - 1)]
        req = urllib.request.Request(
            ENDPOINT.format(model=model),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": gemini_api_key()},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                doc = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            if exc.code not in _RETRYABLE or attempt == attempts - 1:
                raise RuntimeError(f"director HTTP {exc.code}: {detail}") from exc
            wait = 1.5 * (attempt + 1)
            log.info("director %s HTTP %d (attempt %d/%d); next model in %.0fs",
                     model, exc.code, attempt + 1, attempts, wait)
            time.sleep(wait)
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"director unreachable: {exc}") from exc
            time.sleep(2.0 * (2 ** attempt))
            last = exc
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(f"director failed after {attempts} attempts: {last}")
    candidates = doc.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"director returned no candidates: {json.dumps(doc)[:300]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _parse(text: str) -> list[Scenario]:
    """Parse the model's JSON, tolerating a markdown fence it was told not to use."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in director output: {text[:200]}")
    items = json.loads(cleaned[start : end + 1])
    out = []
    for item in items:
        if not item.get("prompt") or not item.get("name"):
            continue
        out.append(
            Scenario(
                name=str(item["name"]).strip().lower().replace(" ", "_")[:48],
                situation=str(item.get("situation", "")).strip(),
                why_it_might_break=str(item.get("why_it_might_break", "")).strip(),
                prompt=str(item["prompt"]).strip()[:1000],
                severity=int(item.get("severity", 3)),
                realism=int(item.get("realism", 3)),
            )
        )
    return out


REPAIR_INSTRUCTIONS = """You wrote these test scenarios for a video model, and each one
breaks a rule that is measured rather than stylistic. Fix each one.

Some breakages are about WORDING - the prompt names two objects when it should name
one. For those, keep the situation exactly as it is and rewrite only the prompt.

Some are about the SITUATION ITSELF. A scenario in a scene-only category that is really
about the cup cannot be fixed by rewording, because the situation is the wrong situation
for that category. For those, change the situation to a genuine instance of what the
category is for, and write a matching prompt and name. Each entry below says which kind
it is.

{listing}

The rules, and what breaking them costs:

  1. At most ONE manipulable object named per prompt (cup, bowl, gripper, container).
     One object: 36% chance the model invents something. Two: 73%. Three: 100%.
     If the situation involves two objects, pick the one that matters and describe only
     that. Do not mention the other at all.
  2. Never target a part of an object - rim, edge, lip, handle, gripper finger, the
     chips inside a cup. 0 of 10 such prompts produced a usable render. Whole objects
     and whole surfaces only.
  3. Never describe the camera, lens, frame or image. The model edits the scene.

Keep them vivid and extreme - a specific colour, material or covering, stated as already
present. A mild prompt gets declined and wastes the scenario just as surely.

Reply with JSON only, an array in the same order, no markdown fence:
[{{"name": "<the name you were given, so it can be matched up>",
  "new_name": "<snake_case name matching the fixed scenario>",
  "situation": "<the situation, unchanged for a wording fix, replaced for a situation fix>",
  "prompt": "<the rewritten prompt>"}}]
"""


def _category_violation(prompt: str, spec) -> str | None:
    """Does this prompt break its category's own contract?

    Only the null-control categories set one. They exist to keep a measured
    null under test, and a director that quietly writes object prompts into
    them removes the control while the suite still claims to have it.
    """
    if not isinstance(spec, dict) or "max_objects" not in spec:
        return None
    named = prompt_features(prompt)["objects_named"]
    if named <= spec["max_objects"]:
        return None
    return (f"this category must name at most {spec['max_objects']} manipulable "
            f"object(s) and this prompt names {named}; it is a null control for "
            f"scene-only change, and an object prompt here silently removes it")


def _repair_prompts(scenarios: list, spec=None) -> tuple[list, list]:
    """Rewrite any proposed prompt that breaks a measured rule.

    Returns (scenarios, notes). The director is told the rules in its own briefing and
    still breaks them - the previous suite produced eleven two-object prompts under
    guidance that already said not to. An instruction nothing verifies is a suggestion,
    and the cost of this particular suggestion being ignored is a wasted render plus a
    wasted slot in the multiplicity correction. So the prompts are checked mechanically
    and the violators go back once.

    A prompt that still violates after the repair pass is kept and flagged rather than
    dropped: the scenario is the director's idea, the phrasing is the renderer's
    problem, and the render agent gets another go at it in the loop.
    """
    bad = []
    for s in scenarios:
        v = list(rule_violations(s.prompt))
        cat = _category_violation(s.prompt, spec)
        if cat:
            v.append(cat)
        if v:
            bad.append((s, v))
    if not bad:
        return scenarios, []

    listing = "\n\n".join(
        f'  name: {s.name}\n  prompt: "{s.prompt}"\n  breaks: '
        + "; ".join(v) for s, v in bad
    )
    payload = {
        "contents": [{"parts": [{"text": REPAIR_INSTRUCTIONS.format(listing=listing)}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }
    notes = []
    try:
        raw = _call(payload, timeout=120.0, attempts=3)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        fixed = json.loads(cleaned[cleaned.find("["): cleaned.rfind("]") + 1])
        by_name = {str(d.get("name")): d for d in fixed}
    except Exception as exc:  # noqa: BLE001
        log.warning("prompt repair pass failed: %s", exc)
        return scenarios, [f"prompt repair unavailable: {type(exc).__name__}"]

    for s, violations in bad:
        fix = by_name.get(s.name) or {}
        candidate = str(fix.get("prompt", "")).strip()
        clean = candidate and not rule_violations(candidate)             and not _category_violation(candidate, spec)
        if clean:
            # The SITUATION has to move with the prompt. A repair that rewrites only
            # the prompt leaves the scenario describing an object while its prompt
            # describes the table, and anything that later regenerates the prompt from
            # the situation - the mid-run re-brief does exactly this - reverts the fix
            # silently. Measured: three null-control scenarios were repaired at
            # proposal time and were object prompts again by the time they rendered.
            new_situation = str(fix.get("situation", "")).strip()
            if new_situation:
                s.situation = new_situation
            new_name = str(fix.get("new_name", "")).strip()
            if new_name and new_name.replace("_", "").isalnum():
                s.name = new_name
            notes.append(f"{s.name}: rewritten ({violations[0][:70]})")
            s.prompt = candidate
        else:
            s.rule_violations = violations
            notes.append(f"{s.name}: STILL BREAKS RULES - {violations[0][:70]}")
            log.info("director prompt for %s still violates: %s", s.name, violations)
    log.info("prompt repair: %d of %d proposals rewritten",
             sum(1 for n in notes if "rewritten" in n), len(bad))
    return scenarios, notes


def propose(
    task: str,
    frames: dict[str, np.ndarray],
    *,
    n: int = 6,
    avoid: list[str] | None = None,
    category: tuple[str, str] | None = None,
    category_spec: dict | None = None,
    brief: str = "",
) -> DirectorResult:
    """Propose `n` scenarios for this task, having looked at the actual scene.

    `frames` maps view name to a single representative frame. The director sees the
    real workspace, so it can name what is actually on the table rather than
    hypothesising about a generic one.
    """
    parts: list[dict] = []

    view_list = ", ".join(frames)
    avoid_note = (
        f"\n\nDo NOT repeat these already-tested scenarios: {', '.join(avoid)}"
        if avoid else ""
    )
    learned = f"\n\nWHAT THIS RUN HAS LEARNED SO FAR:\n{brief}" if brief else ""
    parts.append({"text":
        f"{SYSTEM}\n\n{CAPABILITIES}{learned}\n\n"
        f"ROBOT TASK: \"{task}\"\n"
        f"CAMERA VIEWS AVAILABLE: {view_list}\n"
        f"The images below are the actual workspace, one per camera view, in that order."
        f"\n\nPropose exactly {n} scenarios.{avoid_note}"
    })
    for view, frame in frames.items():
        parts.append({"text": f"view: {view}"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": _frame_to_b64(frame)}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 4096},
    }
    try:
        raw = _call(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("director call failed: %s", exc)
        return DirectorResult(error=str(exc))
    try:
        scenarios = _parse(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("director output unparseable: %s", exc)
        return DirectorResult(raw=raw, error=f"unparseable: {exc}")
    scenarios, repair_notes = _repair_prompts(scenarios, category_spec)
    log.info("director proposed %d scenarios", len(scenarios))
    return DirectorResult(scenarios=scenarios, raw=raw, repairs=repair_notes)

#: Categories the suite sweeps, with how many situations each is worth. A freeform
#: batch clusters - ask an LLM for twenty failure modes and it gives you fifteen
#: lighting variations - so sweeping named categories forces coverage of mechanisms
#: that are genuinely different from each other.
#:
#: The weights are measured, not aesthetic. Categories that target the manipulated
#: objects are where every vulnerability this project has ever found came from;
#: `environment` has gone 0 for 9 and `sensor_fault` describes the camera, which this
#: renderer does not edit (0 of 3 usable). Both are kept at weight 1 rather than
#: deleted, because a null that stops being tested quietly becomes an assumption - and
#: because if the renderer or the policy changes, that is where it would show first.
CATEGORIES: dict[str, dict] = {
    "object_appearance": {
        "weight": 3,
        "definition": "ONE task-relevant object changes colour, material, finish or how "
                      "strongly it contrasts with its surroundings",
    },
    "contents_and_state": {
        "weight": 3,
        "definition": "what is inside or on ONE object changes - fill level, contents "
                      "colour, emptiness, a coating over the whole object",
    },
    "confusable_objects": {
        "weight": 3,
        "definition": "ONE object comes to resemble something it is not - the target "
                      "looks like the background, or like the other container. Describe "
                      "only the object that changes, never the thing it now resembles",
    },
    "surface_and_spill": {
        "weight": 2,
        "definition": "the work surface changes - spills, stains, wetness, reflectivity, "
                      "texture, debris. Name the surface, not the objects on it",
    },
    "lighting": {
        "weight": 2,
        "definition": "illumination changes - time of day, shadows, colour temperature, "
                      "glare, backlight, a light failing",
    },
    "environment": {
        "weight": 1,
        # The whole point of this category is to keep testing a measured null. A
        # prompt here that names the cup or the bowl is not a scene-only change, so
        # the null stops being tested and nobody notices. Enforced, not requested.
        "max_objects": 0,
        "definition": "the wider workspace changes - wall colour, clutter behind the "
                      "bench, a different room. KEPT AS A NULL CONTROL: this category "
                      "has never moved the policy in 9 attempts, and is included so "
                      "that keeps being tested rather than assumed",
    },
    "sensor_fault": {
        "weight": 1,
        "max_objects": 0,
        "definition": "the scene degrades in a way that mimics a failing camera - haze, "
                      "condensation on the objects, a colour cast over everything. "
                      "Describe it as a property of the SCENE, never of the lens",
    },
}


def _allocate(total: int, categories: dict) -> dict:
    """Split `total` situations across categories by weight, everyone getting at least 1.

    Proportional with largest-remainder, so the counts sum to exactly `total` rather
    than drifting the way per-category rounding does. Every category keeps a floor of
    one: the two low-weight categories are null controls, and a control that rounds away
    at small suite sizes stops being a control.
    """
    names = list(categories)
    if total <= len(names):
        return {n: 1 for n in names[:max(1, total)]}
    weights = {n: max(1, categories[n].get("weight", 1)) for n in names}
    total_w = sum(weights.values())
    exact = {n: total * w / total_w for n, w in weights.items()}
    alloc = {n: max(1, int(exact[n])) for n in names}
    # Largest remainder first, then hand back any overshoot from the biggest allocations,
    # which is where a single scenario costs the least coverage.
    while sum(alloc.values()) < total:
        n = max(names, key=lambda k: exact[k] - alloc[k])
        alloc[n] += 1
    while sum(alloc.values()) > total:
        n = max((k for k in names if alloc[k] > 1), key=lambda k: alloc[k] - exact[k])
        alloc[n] -= 1
    return alloc


def propose_suite(
    task: str,
    frames: dict[str, np.ndarray],
    *,
    total: int | None = None,
    per_category: int = 3,
    categories: dict | None = None,
    brief: str = "",
    on_progress=None,
) -> DirectorResult:
    """Sweep every category, so the suite covers mechanisms rather than variations.

    Each category is a separate call carrying the names already proposed, so the
    director does not rediscover the same idea under a new name. A category that fails
    is skipped rather than aborting the sweep - a partial suite is still a suite.

    `total` splits that many situations across the categories by their measured weight,
    which is what a caller asking for "21 scenarios" means. `per_category` is the older
    flat behaviour and is used only when `total` is not given.
    """
    cats = categories or CATEGORIES
    quota = _allocate(total, cats) if total else {c: per_category for c in cats}
    out: list[Scenario] = []
    errors: list[str] = []
    repairs: list[str] = []
    for category, spec in cats.items():
        n = quota.get(category, 0)
        if n <= 0:
            continue
        definition = spec["definition"] if isinstance(spec, dict) else spec
        result = propose(
            task,
            frames,
            n=n,
            avoid=[s.name for s in out],
            category=(category, definition),
            category_spec=spec if isinstance(spec, dict) else None,
            brief=brief,
        )
        repairs.extend(result.repairs)
        if result.error:
            log.warning("category %s failed: %s", category, result.error)
            errors.append(f"{category}: {result.error}")
            continue
        for scenario in result.scenarios:
            if scenario.name in {s.name for s in out}:
                continue
            scenario.category = category
            out.append(scenario)
        log.info("category %-20s asked %d, +%d (%d total)",
                 category, n, len(result.scenarios), len(out))
        if on_progress:
            # The sweep takes a couple of minutes; without this the dashboard sits
            # blank while the most interesting part - what the director thinks could
            # go wrong - is already being decided.
            try:
                on_progress(category, list(out), len(cats))
            except Exception:  # noqa: BLE001
                log.debug("director progress hook failed", exc_info=True)
    return DirectorResult(scenarios=out, repairs=repairs,
                          error="; ".join(errors) if errors and not out else None)
