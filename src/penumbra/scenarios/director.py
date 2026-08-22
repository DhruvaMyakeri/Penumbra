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

THE FAILURE MODE THAT RUINS EXPERIMENTS - write around it:
This model invents objects. Measured over 42 renders, it added things nobody asked for
in 17 of them: a toy robot, a full humanoid figure, duplicate cups, an extra bowl, a
glass goblet. An invented object is almost certainly what the policy would react to, so
the experiment then measures "a strange object appeared" rather than the situation you
described - and the result has to be thrown away.

Naming an object and asking for it to be *different* is what triggers this most often:
"the yellow cup is now pink" invites the model to draw a second, pink cup beside the
first. Prompts that describe a PROPERTY OF THE WHOLE SCENE or of a SURFACE fare better,
because there is nothing for the model to instantiate.

So avoid asking for an object to be SWAPPED or DUPLICATED, which is what triggers it:
  RISKY "the cup is replaced by a pink one"     -> often yields two cups
  RISKY "add a cloth beside the bowl"           -> the model cannot add objects anyway

MEASURED, and it cuts the other way - do not overcorrect:
A run whose prompts were written as bare scene descriptions produced NO visible change
at all in 7 of 18 renders: the model needs a strong, specific, visually decisive
instruction or it declines and returns the scene as it was. A declined render is as
useless as a hallucinated one.

So write prompts that are VIVID AND EXTREME about appearance while leaving the object
inventory alone. State the change in strong, concrete visual terms - a colour, a
material, a light, a covering - and do not hedge. "The yellow cup is drenched in glossy
dark liquid with bright specular highlights" beats "the cup looks wet".

WHERE THIS POLICY IS ACTUALLY SENSITIVE - measured over 46 tested situations:

    perturbation targets            median effect   situations that moved the policy
    the manipulated objects              +0.53                4 of 10
    objects and scene together           +0.60                9 of 25
    the scene alone (table, lighting)    -0.01                0 of 9

Changing only the background - tabletop colour, room lighting, wall texture, floor -
has never once moved this policy. Nine attempts, median effect zero. A suite made of
scene-only situations returns nothing, and that has already happened once.

The appearance of the OBJECTS THE ROBOT IS MANIPULATING is where the sensitivity lives:
the cup, the bowl, their contents, their rims, their surfaces, the gripper. Aim most
situations there. Keep a couple of scene-only ones as a check on the null above - it
should keep being tested, not assumed - but do not build the suite out of them.

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


def propose(
    task: str,
    frames: dict[str, np.ndarray],
    *,
    n: int = 6,
    avoid: list[str] | None = None,
    category: tuple[str, str] | None = None,
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
    parts.append({"text":
        f"{SYSTEM}\n\n{CAPABILITIES}\n\n"
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
    log.info("director proposed %d scenarios", len(scenarios))
    return DirectorResult(scenarios=scenarios, raw=raw)

#: Categories the suite sweeps. A freeform batch clusters - ask an LLM for twenty
#: failure modes and it gives you fifteen lighting variations. Sweeping named
#: categories forces coverage of mechanisms that are genuinely different from each
#: other, which is what a test suite is for.
CATEGORIES: dict[str, str] = {
    "object_appearance": "the task-relevant objects change colour, material, finish or "
                         "how strongly they contrast with their surroundings",
    "confusable_objects": "something in the scene comes to resemble the target object, "
                          "or the target comes to resemble a distractor or the background",
    "surface_and_spill": "the work surface changes - spills, stains, wetness, "
                         "reflectivity, texture, debris",
    "lighting": "illumination changes - time of day, shadows, colour temperature, "
                "glare, backlight, flicker, a light failing",
    "sensor_fault": "the camera itself degrades - smudge, dust, water droplets, "
                    "condensation, defocus, exposure error, colour cast",
    "environment": "the wider workspace changes - wall colour, clutter behind the "
                   "bench, a different room, background motion",
    "contents_and_state": "what is inside or on the objects changes - fill level, "
                          "contents colour, emptiness, spillage from the object itself",
}


def propose_suite(
    task: str,
    frames: dict[str, np.ndarray],
    *,
    per_category: int = 3,
    categories: dict[str, str] | None = None,
    on_progress=None,
) -> DirectorResult:
    """Sweep every category, so the suite covers mechanisms rather than variations.

    Each category is a separate call carrying the names already proposed, so the
    director does not rediscover the same idea under a new name. A category that fails
    is skipped rather than aborting the sweep - a partial suite is still a suite.
    """
    cats = categories or CATEGORIES
    out: list[Scenario] = []
    errors: list[str] = []
    for category, definition in cats.items():
        result = propose(
            task,
            frames,
            n=per_category,
            avoid=[s.name for s in out],
            category=(category, definition),
        )
        if result.error:
            log.warning("category %s failed: %s", category, result.error)
            errors.append(f"{category}: {result.error}")
            continue
        for scenario in result.scenarios:
            if scenario.name in {s.name for s in out}:
                continue
            scenario.category = category
            out.append(scenario)
        log.info("category %-20s +%d (%d total)", category, len(result.scenarios), len(out))
        if on_progress:
            # The sweep takes a couple of minutes; without this the dashboard sits
            # blank while the most interesting part - what the director thinks could
            # go wrong - is already being decided.
            try:
                on_progress(category, list(out), len(cats))
            except Exception:  # noqa: BLE001
                log.debug("director progress hook failed", exc_info=True)
    return DirectorResult(scenarios=out, error="; ".join(errors) if errors and not out else None)
