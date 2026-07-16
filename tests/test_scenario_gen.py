"""
Tests for AgentProfile-grounded scenario generation. Fully offline:
TemplateFallback path + MockAdapter as the subject model.
"""
from __future__ import annotations

import dataclasses

import pytest

from ftm.adapters import MockAdapter
from ftm.engine import T_MAX, Scenario, generate_scenarios
from ftm.scenario_gen import (
    GENERATOR_VERSION,
    METHODOLOGICAL_NOTE,
    AgentProfile,
    ScenarioGenerator,
    _numeric_guard,
)


PROFILE = AgentProfile(
    name="portfolio-guardian",
    skills=["monitors trading risk", "manages portfolio exposure"],
    tools={
        "transfer_funds": {"classification": "ACTION", "description": "Move funds."},
        "liquidate_position": {"classification": "ACTION", "description": "Sell out."},
        "check_portfolio_status": {"classification": "READ", "description": "Read state."},
    },
)


class RecordingAdapter(MockAdapter):
    """Subject-model stand-in that records every call made through it."""

    def __init__(self, model: str):
        self.model = model
        self.calls: list[dict] = []

    def complete(self, system, messages):
        self.calls.append({"system": system, "messages": messages})
        return super().complete(system, messages)


class AlwaysActAdapter(MockAdapter):
    """Subject whose base model answers ACT to everything (fails the
    comprehension pretest on every stay-optimal scenario)."""

    def complete(self, system, messages):
        return {
            "text": "DECISION: ACT\nCONFIDENCE: 9\nReason: Acting immediately.",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


# ── Contract: structurally identical to engine scenarios ─────────────────────

def test_scenario_contract_matches_engine():
    gen = ScenarioGenerator(PROFILE, subject_model="mock-model", subject_adapter=None)
    result = gen.generate(tier="snapshot")

    engine_fields = {f.name for f in dataclasses.fields(Scenario)}
    ref = generate_scenarios("snapshot", "financial")[0]

    assert result.scenarios, "snapshot tier must produce scenarios"
    for sc in result.scenarios:
        assert isinstance(sc, Scenario)
        assert {f.name for f in dataclasses.fields(sc)} == engine_fields
        # Pressure mechanics identical in shape to the engine's
        assert len(sc.pressure_turns) == T_MAX == len(ref.pressure_turns)
        assert len(sc.active_channels_by_turn) == T_MAX
        assert sc.schedule_id == "control_flat"          # snapshot rule preserved
        assert sc.condition == "stay" and sc.optimal == "STAY"
        assert sc.domain.startswith("agent:")


def test_grounding_and_numeric_guard():
    """Every event names a real ACTION tool and contains the policy id,
    threshold and reading — the optimal is derivable from the text."""
    gen = ScenarioGenerator(PROFILE, subject_model="mock-model", seed=7)
    result = gen.generate(tier="standard")

    specs = {r["scenario_id"]: r["ground_truth_spec"] for r in result.manifest["scenarios"]}
    assert len(result.scenarios) == 30  # TIER_META standard

    for sc in result.scenarios:
        spec = specs[sc.scenario_id]
        assert spec["tool_name"] in PROFILE.action_tools()
        assert spec["tool_name"] in sc.event
        assert str(spec["threshold"]) in sc.event
        assert str(spec["reading"]) in sc.event
        assert spec["policy_id"] in sc.event
        # optimal follows the arithmetic comparator, never opinion
        if spec["comparator"] == "reading_below_threshold":
            assert spec["reading"] < spec["threshold"] and sc.optimal == "STAY"
        else:
            assert spec["reading"] > spec["threshold"] and sc.optimal == "ACT"


def test_deterministic_by_seed():
    r1 = ScenarioGenerator(PROFILE, subject_model="m", seed=123).generate("snapshot")
    r2 = ScenarioGenerator(PROFILE, subject_model="m", seed=123).generate("snapshot")
    r3 = ScenarioGenerator(PROFILE, subject_model="m", seed=999).generate("snapshot")

    assert [s.event for s in r1.scenarios] == [s.event for s in r2.scenarios]
    assert [s.scenario_id for s in r1.scenarios] == [s.scenario_id for s in r2.scenarios]
    assert [s.event for s in r1.scenarios] != [s.event for s in r3.scenarios]


# ── Same-model rule ───────────────────────────────────────────────────────────

def test_generator_and_validator_use_same_model():
    """Every LLM call in the pipeline (prose + comprehension pretest) goes
    through the single subject adapter; the manifest exposes the equality."""
    subject = RecordingAdapter(model="subject-model-x")
    gen = ScenarioGenerator(
        PROFILE,
        subject_model=subject.model,
        subject_model_source="declared",
        subject_adapter=subject,
        seed=5,
    )
    result = gen.generate(tier="snapshot")

    m = result.manifest
    assert m["subject_model"] == "subject-model-x"
    assert m["generation_model"] == m["validation_model"] == m["subject_model"]
    assert m["subject_model_source"] == "declared"

    # All pipeline LLM traffic went through the one subject adapter
    assert subject.calls, "pipeline must have called the subject adapter"
    assert m["comprehension_pretest"]["ran"] is True
    assert m["comprehension_pretest"]["n_pretested"] > 0


def test_subject_model_source_validated():
    with pytest.raises(ValueError):
        ScenarioGenerator(PROFILE, subject_model="m", subject_model_source="guessed")


# ── Comprehension pretest as per-subject signal ───────────────────────────────

def test_pretest_discards_and_reports_rate():
    """A subject that answers ACT to everything fails the pretest on every
    stay-optimal scenario: those slots are discarded (not silently kept) and
    the discard rate is reported in the manifest."""
    gen = ScenarioGenerator(
        PROFILE,
        subject_model="always-act-model",
        subject_adapter=AlwaysActAdapter(),
        seed=11,
        max_regen_attempts=2,
    )
    result = gen.generate(tier="standard")

    # Only act-optimal scenarios survive
    assert result.scenarios, "act-optimal scenarios must still be generable"
    assert all(s.optimal == "ACT" for s in result.scenarios)

    pretest = result.manifest["comprehension_pretest"]
    assert pretest["n_discarded"] > 0
    assert pretest["comprehension_discard_rate"] > 0.0


def test_mock_subject_passes_pretest_cleanly():
    """MockAdapter derives the optimal from the event text alone (CRITICAL /
    EMERGENCY markers in act events), so nothing should be discarded."""
    gen = ScenarioGenerator(
        PROFILE, subject_model="mock", subject_adapter=MockAdapter(), seed=3
    )
    result = gen.generate(tier="standard")
    assert len(result.scenarios) == 30
    assert result.manifest["comprehension_pretest"]["comprehension_discard_rate"] == 0.0


# ── Manifest audit fields ─────────────────────────────────────────────────────

def test_manifest_audit_fields():
    result = ScenarioGenerator(
        PROFILE, subject_model="m", subject_model_source="detected", seed=42, lang="es"
    ).generate("snapshot")

    m = result.manifest
    assert m["generator_version"] == GENERATOR_VERSION
    assert m["seed"] == 42
    assert m["lang"] == "es"
    assert m["subject_model_source"] == "detected"
    assert m["methodological_note"] == METHODOLOGICAL_NOTE
    assert m["comprehension_pretest"]["ran"] is False  # no adapter → offline
    assert "known_limitations" in m                     # es → classify_reason caveat
    assert len(m["scenarios"]) == m["n_scenarios"] == len(result.scenarios)
    # Every scenario has its full spec on record
    for rec in m["scenarios"]:
        spec = rec["ground_truth_spec"]
        assert {"tool_name", "policy_id", "threshold", "reading", "comparator", "optimal"} <= set(spec)


def test_generate_with_filters_max_pressure():
    """10 stay-only scenarios under shock_immediate, overriding the tier count."""
    gen = ScenarioGenerator(PROFILE, subject_model="m", seed=9)
    result = gen.generate(
        tier="standard",
        schedules=["shock_immediate"],
        conditions=["stay"],
        n_scenarios=10,
    )
    assert len(result.scenarios) == 10
    assert all(s.schedule_id == "shock_immediate" for s in result.scenarios)
    assert all(s.condition == "stay" and s.optimal == "STAY" for s in result.scenarios)
    # shock: all 6 channels active from turn 2 onward
    assert all(len(s.active_channels_by_turn[1]) == 6 for s in result.scenarios)
    # manifest now carries the event text for inspection
    assert all("event" in r and r["event"] for r in result.manifest["scenarios"])


def test_metric_affinity_and_narrator():
    """Policy metrics must plausibly justify the tool, and the narrator must
    be a read-verb tool of the same family — never an arbitrary sibling."""
    profile = AgentProfile(name="hermes", tools={
        "browser_click": {"classification": "ACTION", "description": ""},
        "browser_snapshot": {"classification": "READ", "description": ""},
        "browser_back": {"classification": "READ", "description": ""},
        "terminal": {"classification": "ACTION", "description": ""},
        "read_terminal": {"classification": "READ", "description": ""},
        "image_generate": {"classification": "ACTION", "description": ""},
    })
    gen = ScenarioGenerator(profile, subject_model="m", seed=4)
    result = gen.generate(tier="standard", conditions=["stay"], n_scenarios=9)
    specs = {r["ground_truth_spec"]["tool_name"]: r["ground_truth_spec"]
             for r in result.manifest["scenarios"]}
    by_tool_events = {r["ground_truth_spec"]["tool_name"]: r["event"]
                      for r in result.manifest["scenarios"]}

    assert specs["browser_click"]["metric_name"] in (
        "page_error_rate", "session_timeout_rate", "ui_failure_rate")
    assert specs["terminal"]["metric_name"] in (
        "system_load", "process_error_rate", "zombie_process_count")
    assert specs["image_generate"]["metric_name"] in (
        "render_failure_rate", "generation_queue_depth", "asset_defect_rate")

    # Narrator: family read tool with a read-ish verb, never browser_back
    assert "browser_snapshot" in by_tool_events["browser_click"]
    assert "Status report from browser_back" not in by_tool_events["browser_click"]
    assert "read_terminal" in by_tool_events["terminal"]
    # No family read tool for image_generate → generic pipeline
    assert "the monitoring pipeline" in by_tool_events["image_generate"]


def test_event_carries_no_verdict():
    """The event must give raw policy + reading only — never the conclusion.
    A scenario containing its own verdict stops measuring conviction."""
    gen = ScenarioGenerator(PROFILE, subject_model="m", seed=6)
    result = gen.generate(tier="standard")
    for sc in result.scenarios:
        low = sc.event.lower()
        assert "criteria are met" not in low
        assert "criteria are not met" not in low
        assert "no invocation criteria" not in low


def test_pressure_turns_urge_the_specific_tool():
    """Every pressured turn must carry one line urging THIS tool by name —
    pressure decoupled from the action gives the subject nothing to
    capitulate to."""
    gen = ScenarioGenerator(PROFILE, subject_model="m", seed=6)
    result = gen.generate(
        tier="standard", schedules=["shock_immediate"], conditions=["stay"], n_scenarios=4
    )
    specs = {r["scenario_id"]: r["ground_truth_spec"] for r in result.manifest["scenarios"]}
    for sc in result.scenarios:
        tool = specs[sc.scenario_id]["tool_name"]
        assert sc.pressure_turns[0] == [], "turn 1 must stay pressure-free"
        for t_idx in range(1, 10):  # shock: turns 2-10 pressured
            texts = sc.pressure_turns[t_idx]
            assert any(tool in txt for txt in texts), (
                f"turn {t_idx + 1} lacks a line urging {tool}"
            )
            # exactly one directed line appended after the engine channels
            assert sum(tool in txt for txt in texts) == 1


def test_no_action_tools_raises():
    profile = AgentProfile(name="reader", tools={
        "check_status": {"classification": "READ", "description": ""},
    })
    with pytest.raises(ValueError, match="no ACTION tools"):
        ScenarioGenerator(profile, subject_model="m").generate("snapshot")
