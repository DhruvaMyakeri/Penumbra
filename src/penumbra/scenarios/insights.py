"""What the run has learned about the renderer, so far, in this run.

The agents in this pipeline used to be memoryless. The director proposed twenty-one
situations before a single frame was rendered, and the render agent rewrote each prompt
knowing only that scenario's own failures. So the suite made the same mistake twenty-one
times: scenario 19 could not benefit from the fact that scenarios 1-18 had all invented
an extra cup.

This is the missing state. Every render attempt is recorded with the *structural*
features of the prompt that produced it, alongside what the renderer did in response.
That turns a run into an accumulating experiment about the renderer itself, and the
summary is fed back to both agents while the run is still going.

Two disciplines keep this from becoming a rumour mill:

**Only observations go in.** A record is written when a render is gated and judged, from
the gate's own verdict and the judge's own verdict. No agent writes its opinion here.
The ``lessons`` field exists for agent-authored notes and is kept separate and clearly
labelled, so a model can never launder a guess into the evidence table.

**Priors are labelled as priors.** The ledger opens carrying the measured rates from
previous suites (92 renders; see "Prompt structure" in ``docs/METHODOLOGY.md``) so that
early scenarios are not advised by a sample of two. As this run accumulates its own observations they are
reported alongside the prior, never silently merged into it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("penumbra.insights")

#: Nouns for things the robot manipulates. Rule 1 counts *distinct* members of this set
#: in a prompt, because naming a second one doubled the invention rate over 92 renders.
OBJECT_NOUNS = re.compile(
    r"\b(cup|cups|bowl|bowls|mug|mugs|gripper|grippers|jar|jars|bottle|bottles|"
    r"can|cans|box|boxes|container|containers|plate|plates|tool|tools)\b",
    re.I,
)

#: Sub-object features. Ten prompts named one of these; none produced a usable render.
NARROW_FEATURES = re.compile(
    r"\b(rim|rims|edge|edges|finger|fingers|fingertip\w*|inner wall\w*|inside wall\w*|"
    r"lip|lips|handle|handles|seam|seams|chip|chips)\b",
    re.I,
)

#: Prompts about the camera rather than the scene. 0 of 3 usable - suggestive only.
CAMERA_SPACE = re.compile(
    r"\b(camera|lens|frame|viewfinder|image sensor|the shot|the view)\b", re.I
)

#: The prior: n=92 judged renders from previous suites (docs/METHODOLOGY.md).
PRIOR = {
    "objects_named": {
        0: {"n": 15, "usable": 0.533, "invented": 0.400},
        1: {"n": 28, "usable": 0.286, "invented": 0.357},
        2: {"n": 44, "usable": 0.182, "invented": 0.727},
        3: {"n": 5, "usable": 0.200, "invented": 1.000},
    },
    "narrow_feature": {"n": 10, "usable": 0.000},
    "whole_object": {"n": 82, "usable": 0.305},
    "overall_usable": 0.272,
}


def prompt_features(prompt: str) -> dict:
    """The structural properties of a prompt that measurably predict what X2 does.

    Deliberately shallow. These are counted from the text with regexes rather than
    inferred by a model, so the same prompt always yields the same features and the
    feature-to-outcome table stays an observation instead of another model's opinion.
    """
    objects = {m.group(0).lower().rstrip("s") for m in OBJECT_NOUNS.finditer(prompt)}
    return {
        "objects_named": len(objects),
        "object_list": sorted(objects),
        "narrow_feature": bool(NARROW_FEATURES.search(prompt)),
        "camera_space": bool(CAMERA_SPACE.search(prompt)),
        "words": len(prompt.split()),
    }


def rule_violations(prompt: str) -> list[str]:
    """Which measured rules this prompt breaks. Empty is the goal.

    Used two ways: to advise an agent before it commits to a prompt, and to record
    whether the advice was taken. Both matter - an agent that is told the rules and
    ignores them is a different problem from one that was never told.
    """
    f = prompt_features(prompt)
    out = []
    if f["objects_named"] > 1:
        out.append(
            f"names {f['objects_named']} manipulable objects "
            f"({', '.join(f['object_list'])}); measured invention rate rises from 36% "
            f"at one object to 73% at two"
        )
    if f["narrow_feature"]:
        out.append(
            "targets a sub-object feature (rim/edge/finger/chips); 0 of 10 such prompts "
            "produced a usable render"
        )
    if f["camera_space"]:
        out.append(
            "describes the camera or the frame rather than the scene; X2 edits the "
            "scene, and 0 of 3 camera-space prompts were usable (weak evidence, n=3)"
        )
    return out


@dataclass
class RenderObservation:
    """One render attempt and what the renderer did with it. Facts only."""

    scenario: str
    attempt: int
    prompt: str
    features: dict
    violations: list
    #: "declined" | "gate_rejected" | "wrong_thing" | "incoherent" | "usable"
    outcome: str
    judge_verdict: str = ""
    invented: list = field(default_factory=list)
    gate_reasons: list = field(default_factory=list)
    observed: str = ""


@dataclass
class InsightLedger:
    """Accumulating observations about the renderer, for one run."""

    observations: list = field(default_factory=list)
    #: Agent-authored notes. Kept apart from `observations` on purpose - see the
    #: module docstring.
    lessons: list = field(default_factory=list)
    path: Path | None = None

    # -- writing ------------------------------------------------------------------
    def record(self, obs: RenderObservation) -> None:
        self.observations.append(obs)
        self.flush()

    def note(self, author: str, text: str) -> None:
        """Record an agent's own conclusion, labelled with who said it."""
        self.lessons.append({"author": author, "text": str(text)[:400]})
        self.flush()

    def flush(self) -> None:
        if not self.path:
            return
        try:
            self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            log.debug("could not persist insight ledger", exc_info=True)

    # -- reading ------------------------------------------------------------------
    def rate(self, outcome: str, where=None) -> tuple[int, int]:
        """(matching, total) over observations, optionally filtered."""
        pool = [o for o in self.observations if where is None or where(o)]
        return sum(1 for o in pool if o.outcome == outcome), len(pool)

    def invented_objects(self) -> list[str]:
        seen: list[str] = []
        for o in self.observations:
            for x in o.invented:
                if x.lower() not in {s.lower() for s in seen}:
                    seen.append(x)
        return seen

    def by_object_count(self) -> dict:
        out: dict[int, dict] = {}
        for o in self.observations:
            k = min(o.features.get("objects_named", 0), 3)
            b = out.setdefault(k, {"n": 0, "usable": 0, "invented": 0})
            b["n"] += 1
            b["usable"] += o.outcome == "usable"
            b["invented"] += bool(o.invented)
        return out

    def brief(self, *, max_chars: int = 2200) -> str:
        """A compact briefing for an agent prompt: the prior, then this run's evidence.

        Written to be pasted into a system prompt. The prior always appears, because a
        run that is three scenarios old has nothing useful to say on its own; this run's
        numbers appear underneath and are explicitly marked as the smaller sample.
        """
        lines = [
            "MEASURED BEHAVIOUR OF THIS RENDERER (X2), from 92 judged renders in "
            "previous suites:",
            "  objects named in the prompt -> chance the render shows what was asked, "
            "and chance it invents an object",
            "     0 objects   usable 53%   invented 40%   (but the policy ignores "
            "scene-only changes, so these discover nothing)",
            "     1 object    usable 29%   invented 36%   <- the operating point",
            "     2 objects   usable 18%   invented 73%",
            "     3+ objects  usable 20%   invented 100%",
            "  prompts targeting a rim/edge/finger/chips: 0 of 10 usable.",
        ]
        if self.observations:
            n = len(self.observations)
            usable, _ = self.rate("usable")
            declined, _ = self.rate("declined")
            rejected, _ = self.rate("gate_rejected")
            wrong, _ = self.rate("wrong_thing")
            inc, _ = self.rate("incoherent")
            lines += [
                "",
                f"THIS RUN SO FAR ({n} render attempts - a small sample, read it as a "
                f"trend and not as a correction to the table above):",
                f"  usable {usable}   editor declined {declined}   "
                f"gate rejected {rejected}   rendered the wrong thing {wrong}   "
                f"cameras disagreed {inc}",
            ]
            buckets = self.by_object_count()
            if buckets:
                parts = [f"{k}obj: {v['usable']}/{v['n']} usable"
                         for k, v in sorted(buckets.items())]
                lines.append("  by object count -> " + ",  ".join(parts))
            invented = self.invented_objects()
            if invented:
                lines.append(
                    "  objects this renderer has invented unprompted in this run: "
                    + ", ".join(invented[:12])
                )
        if self.lessons:
            lines.append("")
            lines.append("NOTES FROM EARLIER AGENT TURNS (opinions, not measurements):")
            for item in self.lessons[-6:]:
                lines.append(f"  [{item['author']}] {item['text']}")
        return "\n".join(lines)[:max_chars]

    def to_dict(self) -> dict:
        return {
            "prior": PRIOR,
            "observations": [asdict(o) for o in self.observations],
            "lessons": self.lessons,
            "summary": {
                "attempts": len(self.observations),
                "usable": self.rate("usable")[0],
                "declined": self.rate("declined")[0],
                "gate_rejected": self.rate("gate_rejected")[0],
                "wrong_thing": self.rate("wrong_thing")[0],
                "incoherent": self.rate("incoherent")[0],
                "by_object_count": self.by_object_count(),
                "invented_objects": self.invented_objects(),
            },
        }
