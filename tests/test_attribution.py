"""The attribution layer: what a run is allowed to CLAIM, as opposed to what it measured.

These tests exist because of a specific failure. A suite reported 15 vulnerabilities;
an independent judge found 3 of the 15 renders showed the situation they were named
after, and a cross-camera audit found 14 showed *different scenes on the two cameras*.
Every statistic was sound and every name was fiction.

Nothing here tests statistics. It all tests the boundary between "the policy moved" and
"this situation moved the policy", which is the claim that was being made wrongly.
"""
from __future__ import annotations

import json

import pytest

from penumbra.report.run_report import run_report, scenario_report
from penumbra.scenarios.director import CATEGORIES, _allocate
from penumbra.scenarios.garage import ScenarioResult
from penumbra.scenarios.director import Scenario
from penumbra.scenarios.insights import (
    InsightLedger, RenderObservation, prompt_features, rule_violations,
)
from penumbra.scenarios.judge import IntentVerdict, check_coherence
from penumbra.scenarios.render_agent import REPAIRABLE, classify


class _Comparison:
    """Just enough of a GroupComparison for the taxonomy to read."""

    def __init__(self, significant=True):
        self.any_significant = significant
        self.significant = significant
        self.significant_gripper = False
        self.p_value = 0.003
        self.effect_size = 1.6
        self.gripper_p_value = 0.4
        self.at_resolution_floor = True
        self.notes = []

    def to_dict(self):
        return {"p_value": self.p_value, "effect_size_cohens_d": self.effect_size,
                "gripper_p_value": self.gripper_p_value, "any_significant": True,
                "at_resolution_floor": True, "notes": []}


def _result(intent, *, significant=True, status="done"):
    return ScenarioResult(
        scenario=Scenario(name="wet_bowl", situation="s", why_it_might_break="w",
                          prompt="p"),
        status=status,
        vs_control=_Comparison(significant),
        intent=intent,
        passes_fdr=True,
    )


# -- the taxonomy --------------------------------------------------------------

def test_credible_coherent_render_is_confirmed():
    r = _result({"credible": True, "verdict": "applied", "coherent": True})
    assert r.finding_class == "CONFIRMED"
    assert "CONFIRMED" in r.verdict()


def test_wrong_thing_rendered_is_unattributed_not_a_finding():
    r = _result({"credible": False, "verdict": "something_else",
                 "observed": "a new mug appeared", "coherent": True})
    assert r.finding_class == "UNATTRIBUTED"
    # The effect must still be reported - suppressing it would hide the renderer's
    # failure rate - but the situation's name must be disowned in the same breath.
    assert "UNATTRIBUTED" in r.verdict()
    assert "is not what caused it" in r.verdict()


def test_disagreeing_cameras_are_unattributed_even_when_the_judge_approved():
    """The strongest form: a render the judge liked, on cameras that disagree.

    `credible` is True here. If coherence did not override it this would be reported
    as a confirmed finding, which is exactly the 14-of-15 error.
    """
    r = _result({"credible": True, "verdict": "applied", "coherent": False,
                 "coherence_note": "camera 1 shows foam, camera 2 shows chrome"})
    assert r.finding_class == "UNATTRIBUTED"
    assert "CAMERAS DISAGREE" in r.verdict()


def test_missing_adjudication_is_unattributed_not_confirmed():
    """No judge verdict is not evidence of a good render."""
    assert _result(None).finding_class == "UNATTRIBUTED"
    assert _result({"error": "timeout"}).finding_class == "UNATTRIBUTED"


def test_no_significance_is_no_effect_whatever_the_render_showed():
    r = _result({"credible": True, "verdict": "applied", "coherent": True},
                significant=False)
    assert r.finding_class == "NO EFFECT"


@pytest.mark.parametrize("status,expected", [
    ("no_change", "DECLINED"), ("rejected", "REJECTED"),
    ("error", "ERROR"), ("rendered", "PENDING"),
])
def test_non_tested_statuses_map_to_their_own_classes(status, expected):
    assert _result({"credible": True}, status=status).finding_class == expected


def test_is_vulnerability_no_longer_implies_a_nameable_finding():
    """The old flag survives for record compatibility but must not gate the headline."""
    r = _result({"credible": False, "verdict": "something_else", "coherent": False})
    assert r.is_vulnerability is True      # the statistic held
    assert r.finding_class == "UNATTRIBUTED"   # the name did not


# -- the measured prompt rules -------------------------------------------------

def test_object_counting_is_what_the_rule_actually_measures():
    assert prompt_features("The bowl is full of foam.")["objects_named"] == 1
    f = prompt_features("The cup and the bowl are greasy.")
    assert f["objects_named"] == 2
    assert sorted(f["object_list"]) == ["bowl", "cup"]
    # Plural and singular are the same object for this purpose.
    assert prompt_features("The cups and cup")["objects_named"] == 1


def test_each_measured_rule_is_enforced():
    assert rule_violations("The bowl is filled with glossy black liquid.") == []
    assert len(rule_violations("The cup and bowl are both wet.")) == 1
    assert any("sub-object" in v for v in rule_violations("The bowl's rim is chipped."))
    assert any("camera" in v for v in rule_violations("The lens is smudged."))


def test_a_prompt_can_break_several_rules_at_once():
    v = rule_violations("Glare on the rims of both the pink bowl and the yellow cup.")
    assert len(v) == 2


# -- the insight ledger --------------------------------------------------------

def test_ledger_brief_always_carries_the_prior():
    """An empty ledger must still advise, or the first scenarios go in blind."""
    brief = InsightLedger().brief()
    assert "1 object" in brief and "operating point" in brief


def test_ledger_accumulates_this_runs_evidence_separately_from_the_prior():
    led = InsightLedger()
    for i in range(3):
        led.record(RenderObservation(
            scenario=f"s{i}", attempt=1, prompt="The bowl is black.",
            features=prompt_features("The bowl is black."), violations=[],
            outcome="incoherent", invented=["a second bowl"]))
    brief = led.brief()
    assert "THIS RUN SO FAR" in brief
    assert "cameras disagreed 3" in brief
    assert "a second bowl" in brief
    assert led.rate("incoherent") == (3, 3)
    assert led.by_object_count()[1]["n"] == 3


def test_agent_notes_are_labelled_and_kept_out_of_the_evidence():
    led = InsightLedger()
    led.note("director", "I think glare works well")
    assert led.observations == []          # an opinion is not an observation
    assert "opinions, not measurements" in led.brief()


# -- coherence -----------------------------------------------------------------

def test_one_camera_seeing_nothing_is_decided_without_asking_a_model():
    """Disagreement about whether an event happened needs no judgement of wording."""
    coherent, note = check_coherence({
        "cam1": {"verdict": "applied", "observed": "the bowl is full of foam"},
        "cam2": {"verdict": "not_applied", "observed": "no visible differences"},
    })
    assert coherent is False
    assert "disagree about whether anything happened" in note


def test_a_single_camera_reports_unknown_rather_than_agreement():
    coherent, note = check_coherence({"cam1": {"verdict": "applied", "observed": "x"}})
    assert coherent is None
    assert "could not be assessed" in note


def test_incoherence_makes_a_verdict_not_credible():
    v = IntentVerdict(verdict="applied", coherent=False)
    assert v.credible is False
    assert IntentVerdict(verdict="applied", coherent=True).credible is True


def test_classify_reports_disagreeing_cameras_and_it_is_repairable():
    outcome, detail = classify(
        {"valid": True},
        {"coherent": False, "coherence_note": "different scenes",
         "per_view": {"cam1": {"observed": "foam"}, "cam2": {"observed": "chrome"}}},
        False)
    assert outcome == "CAMERAS DISAGREED"
    assert "foam" in detail and "chrome" in detail
    assert outcome in REPAIRABLE


# -- suite allocation ----------------------------------------------------------

@pytest.mark.parametrize("total", [7, 10, 14, 21, 30])
def test_allocation_spends_exactly_the_budget_and_never_starves_a_control(total):
    alloc = _allocate(total, CATEGORIES)
    assert sum(alloc.values()) == total
    # environment and sensor_fault are null controls. A control that rounds away at
    # small suite sizes has stopped being a control.
    assert all(v >= 1 for v in alloc.values())


def test_allocation_favours_the_categories_that_have_produced_findings():
    alloc = _allocate(21, CATEGORIES)
    assert alloc["object_appearance"] > alloc["environment"]
    objecty = alloc["object_appearance"] + alloc["contents_and_state"] \
        + alloc["confusable_objects"]
    assert objecty > total_of_rest(alloc, objecty)


def total_of_rest(alloc, objecty):
    return sum(alloc.values()) - objecty


# -- reports -------------------------------------------------------------------

def _state():
    return {
        "episode": "ep0", "policy": "p", "task": "t", "views": ["a", "b"],
        "views_untouched": [], "cost": {"estimated_usd": 1.0},
        "summary": {"proposed": 3, "tested": 2, "confirmed": ["good"],
                    "unattributed": ["bad"],
                    "unattributed_reasons": {"bad": "cameras rendered different scenes"},
                    "no_effect": [], "rejected_renders": [], "no_change_renders": [],
                    "errors": []},
        "scenarios": [
            {"name": "good", "situation": "s", "why_it_might_break": "w", "prompt": "p",
             "finding_class": "CONFIRMED", "verdict": "CONFIRMED - ...",
             "vs_control": _Comparison().to_dict(),
             "intent": {"verdict": "applied", "observed": "foam", "coherent": True}},
            {"name": "bad", "situation": "s", "why_it_might_break": "w", "prompt": "p",
             "finding_class": "UNATTRIBUTED", "verdict": "UNATTRIBUTED - ...",
             "vs_control": _Comparison().to_dict(),
             "intent": {"verdict": "applied", "observed": "foam", "coherent": False,
                        "coherence_note": "cam1 foam, cam2 chrome",
                        "per_view": {"a": {"verdict": "applied", "observed": "foam"},
                                     "b": {"verdict": "something_else",
                                           "observed": "chrome"}}}},
        ],
    }


def test_run_report_headlines_confirmed_and_never_totals_the_two_together():
    text = run_report(_state())
    assert "**1 confirmed**" in text
    assert "cause is not established" in text
    # The failure this guards: a reader must never see "2 findings".
    assert "**2 confirmed**" not in text


def test_run_report_states_what_it_does_not_establish():
    text = run_report(_state())
    assert "No robot failed" in text
    assert "One episode" in text


def test_scenario_report_of_an_unattributed_finding_disowns_its_own_name():
    text = scenario_report(_state()["scenarios"][1], run_name="r")
    assert "UNATTRIBUTED" in text
    assert "cause is not established" in text
    assert "Do not quote this situation by name" in text
    # Both cameras' descriptions have to appear, or the reader cannot check the claim.
    assert "foam" in text and "chrome" in text


def test_scenario_report_shows_the_camera_disagreement_table():
    text = scenario_report(_state()["scenarios"][1], run_name="r")
    assert "Camera by camera" in text
    assert "The cameras disagree" in text


# -- the null-control categories -----------------------------------------------

def test_a_null_control_category_rejects_object_prompts():
    """The reason this exists, from a live run.

    The director put "the bowl is filled with purple yogurt" into `environment` - a
    category whose entire job is to keep testing the measured null that scene-only
    changes never move this policy. An object prompt there removes the control while
    the suite still reports having one, and nothing would have flagged it.
    """
    from penumbra.scenarios.director import _category_violation

    env = CATEGORIES["environment"]
    assert _category_violation("The bowl is filled with purple yogurt.", env)
    assert _category_violation("The cup is rough grey concrete.", env)
    assert _category_violation("The back wall is bright orange.", env) is None


def test_categories_without_a_budget_are_unconstrained():
    from penumbra.scenarios.director import _category_violation

    assert _category_violation("The bowl is filled with foam.",
                               CATEGORIES["object_appearance"]) is None
    assert _category_violation("anything at all", None) is None


def test_both_null_controls_carry_an_object_budget():
    """If either loses its budget, the null silently stops being tested."""
    for name in ("environment", "sensor_fault"):
        assert CATEGORIES[name].get("max_objects") == 0, name


# -- transport resilience ------------------------------------------------------

def test_a_dropped_session_is_retryable_but_a_bad_request_is_not():
    """Measured three times: a session goes ready, drops, and every later push_frame
    raises INVALID_STATE. That killed a head-to-head and a 21-situation suite. It is a
    transport failure wearing a state error's clothes, so it must be retried - while a
    genuine mistake like a malformed request must still abort.
    """
    from reactor_sdk.errors import (
        BadRequestError, DisconnectedError, InvalidStateError, UnauthorizedError,
    )
    from penumbra.perturbation.x2 import TRANSPORT_ERRORS

    assert issubclass(InvalidStateError, TRANSPORT_ERRORS)
    assert issubclass(DisconnectedError, TRANSPORT_ERRORS)
    assert not issubclass(BadRequestError, TRANSPORT_ERRORS)
    assert not issubclass(UnauthorizedError, TRANSPORT_ERRORS)


def test_the_category_budget_actually_reaches_the_repair_pass():
    """The checker worked; the wiring did not, and only a live run revealed it.

    `propose_suite` passed the category's human-readable definition STRING down to
    `propose`, which handed it to the repair pass as if it were the spec dict. The
    isinstance(dict) guard then matched nothing and every object prompt sailed through
    a null-control category - silently, with the run reporting "no repairs needed".

    Testing `_category_violation` in isolation could never catch that. This asserts on
    the seam instead: whatever `propose_suite` hands down must be the thing that carries
    `max_objects`.
    """
    import inspect

    from penumbra.scenarios import director

    # propose() must take the spec as its own argument, not smuggle it inside `category`.
    assert "category_spec" in inspect.signature(director.propose).parameters

    src = inspect.getsource(director.propose_suite)
    assert "category_spec=spec" in src, \
        "propose_suite must forward the spec dict, not the definition string"

    # And the object that gets forwarded must be one the checker can actually read.
    spec = director.CATEGORIES["environment"]
    assert isinstance(spec, dict) and "max_objects" in spec
    assert director._category_violation("The bowl is full of yogurt.", spec) is not None


def test_reactor_transport_failures_are_retryable_and_real_mistakes_are_not():
    """The bug this pins down was invisible for the life of the project.

    `ExperimentRunner._rollout` caught `(TimeoutError, RuntimeError)` and its docstring
    said it existed to survive wedged Reactor sessions. It could not: every reactor_sdk
    error derives from ReactorError -> Exception, never from RuntimeError. The retry
    read correctly, logged nothing, and caught none of the failures it was for. A run
    died on NETWORK_ERROR during connect - precisely the case it was meant to survive.

    Catching the shared ReactorError base would over-correct: it also covers genuine
    mistakes like a malformed request, which must abort rather than be repeated.
    """
    import reactor_sdk.errors as E

    from penumbra.reactor.session import TRANSPORT_ERRORS

    for name in ("NetworkError", "DisconnectedError", "InvalidStateError",
                 "ServerError", "RequestTimeoutError", "SessionTerminalError"):
        assert issubclass(getattr(E, name), TRANSPORT_ERRORS), name
        assert not issubclass(getattr(E, name), RuntimeError), \
            f"{name} is not a RuntimeError - that is the whole bug"

    for name in ("BadRequestError", "UnauthorizedError", "NotFoundError"):
        assert not issubclass(getattr(E, name), TRANSPORT_ERRORS), \
            f"{name} is a real mistake and must abort, not retry"


def test_both_reactor_callers_use_the_shared_retry_tuple():
    """The renderer and the policy both talk to Reactor; one having the fix is not enough."""
    import inspect

    from penumbra.experiments import runner
    from penumbra.perturbation import x2

    assert "TRANSPORT_ERRORS" in inspect.getsource(runner._rollout_source_probe
                                                   if hasattr(runner, "_rollout_source_probe")
                                                   else runner.ExperimentRunner._rollout)
    assert "TRANSPORT_ERRORS" in inspect.getsource(x2.X2Perturbation._apply_one_resilient)
