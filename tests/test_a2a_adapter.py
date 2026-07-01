"""
Acceptance tests for the A2A + OTel agent adapter slice.

Covers:
- End-to-end FTM scenario against the fake agent: ACT is derived from the
  correlated execute_tool span, and lands on the exact turn that produced it.
- Late-arriving spans (async telemetry): polling with timeout picks them up.
- Zero spans in the window → UNKNOWN (PARSE_FAIL), never STAY.
- Stateful resumption: an interrupted A2A scenario is REDONE from turn 1 with
  a fresh contextId, not continued mid-way.
- Tool classification: LLM judge prioritized, cached, override-able,
  exportable as an audit artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ftm.a2a import A2AAgentAdapter, InProcessSpanCollector, ToolClassifier
from ftm.engine import TurnResult, generate_scenarios
from ftm.runner import RunConfig, run
from tests.fakes import FakeA2AAgent


def _language_agnostic_judge(name: str, description: str) -> str:
    """Stand-in for the LLM judge: decides from a Spanish description,
    which the English verb heuristic could never parse."""
    return "READ" if "sin modificarlo" in description else "ACTION"


def _make_adapter(agent_kwargs: dict, span_timeout_s: float = 3.0):
    collector = InProcessSpanCollector()
    agent = FakeA2AAgent(collector, **agent_kwargs)
    adapter = A2AAgentAdapter(
        transport=agent,
        collector=collector,
        classifier=ToolClassifier(judge=_language_agnostic_judge),
        span_timeout_s=span_timeout_s,
    )
    return agent, adapter


def _run_in(tmpdir: Path, config: RunConfig, adapter) -> dict:
    old = os.getcwd()
    os.chdir(tmpdir)
    try:
        return run(config, adapter)
    finally:
        os.chdir(old)


def _checkpoint_turns(tmpdir: Path, run_id: str) -> list[TurnResult]:
    path = tmpdir / "checkpoints" / f"{run_id}.jsonl"
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(TurnResult(**json.loads(line)))
    return out


def _cfg(run_id: str, max_turns: int) -> RunConfig:
    return RunConfig(
        models=["fake-a2a-agent"],
        tier="snapshot",
        domain="financial",
        run_id=run_id,
        adapter="mock",
        max_turns=max_turns,
    )


# ── 1. End-to-end + span→turn correlation ─────────────────────────────────────

def test_e2e_act_derived_from_span_on_correct_turn(tmp_path):
    """Tool call injected at turn 3 → ACT lands on turn 3, not any other."""
    agent, adapter = _make_adapter({"act_turns": {3}})
    report = _run_in(tmp_path, _cfg("a2a_e2e", max_turns=4), adapter)

    sid = generate_scenarios("snapshot", "financial")[0].scenario_id
    turns = sorted(
        (t for t in _checkpoint_turns(tmp_path, "a2a_e2e") if t.scenario_id == sid),
        key=lambda t: t.turn,
    )
    assert [t.turn for t in turns] == [1, 2, 3, 4]
    assert [t.decision for t in turns] == ["STAY", "STAY", "ACT", "STAY"]

    # The ACT turn carries the correlated tool call in its audit payload
    payload = json.loads(turns[2].raw_response)
    assert payload["tool_calls"][0]["name"] == "transfer_funds"
    assert payload["tool_calls"][0]["classification"] == "ACTION"

    # Scenario optimal is STAY → the injected ACT must register as incorrect
    assert turns[2].is_correct is False
    assert all(t.is_correct for t in turns if t.turn != 3)
    assert report["n_scenarios"] == 1


# ── 2a. Late span: polling picks it up (must-fix 1) ───────────────────────────

def test_late_span_detected_via_polling(tmp_path):
    agent, adapter = _make_adapter(
        {"act_turns": {2}, "span_delay_s": 0.3}, span_timeout_s=3.0
    )
    _run_in(tmp_path, _cfg("a2a_late", max_turns=2), adapter)

    turns = sorted(_checkpoint_turns(tmp_path, "a2a_late"), key=lambda t: t.turn)
    assert [t.decision for t in turns] == ["STAY", "ACT"], (
        "Late-exported span (0.3s) must be captured by the polling window"
    )


# ── 2b. No spans in window: UNKNOWN, never STAY (must-fix 1) ──────────────────

def test_no_spans_yields_unknown_never_stay(tmp_path):
    agent, adapter = _make_adapter({"drop_spans": True}, span_timeout_s=0.3)
    _run_in(tmp_path, _cfg("a2a_unknown", max_turns=2), adapter)

    turns = _checkpoint_turns(tmp_path, "a2a_unknown")
    assert turns, "Turns must still be checkpointed"
    for t in turns:
        assert t.decision == "PARSE_FAIL", (
            f"Silent telemetry must yield UNKNOWN/PARSE_FAIL, got {t.decision!r}"
        )
        assert t.decision != "STAY"
        assert t.reason.startswith("UNKNOWN:")


# ── 3. Stateful resumption: redo from turn 1, fresh contextId (must-fix 2) ────

def test_a2a_interrupted_scenario_is_redone_not_continued(tmp_path):
    collector = InProcessSpanCollector()
    agent = FakeA2AAgent(collector, act_turns=set())
    adapter = A2AAgentAdapter(
        transport=agent, collector=collector,
        classifier=ToolClassifier(judge=_language_agnostic_judge),
        span_timeout_s=3.0,
    )

    # Phase 1: interrupted after turn 2
    _run_in(tmp_path, _cfg("a2a_resume", max_turns=2), adapter)
    assert len(agent.context_ids_seen) == 1
    first_ctx = agent.context_ids_seen[0]
    assert agent.turn_counter[first_ctx] == 2

    # Phase 2: resume with 3 turns — must REDO from turn 1 in a NEW context
    _run_in(tmp_path, _cfg("a2a_resume", max_turns=3), adapter)

    assert len(agent.context_ids_seen) == 2, (
        "Resume must open a fresh A2A contextId, not reuse the interrupted one"
    )
    second_ctx = agent.context_ids_seen[1]
    assert agent.turn_counter[second_ctx] == 3, (
        "Redo must replay ALL turns (1..3) in the new context, not continue at 3"
    )

    # Checkpoint: turns 1-2 appear twice (partial + redo); keep-last dedup
    # must leave exactly turns {1,2,3} for the scenario.
    raw = _checkpoint_turns(tmp_path, "a2a_resume")
    assert sorted(t.turn for t in raw) == [1, 1, 2, 2, 3]
    deduped = {t.turn: t for t in raw}  # same keep-last rule as the runner
    assert sorted(deduped) == [1, 2, 3]
    assert all(t.decision == "STAY" for t in deduped.values())


# ── 4. Classification: judge priority, cache, overrides, audit artifact ───────

def test_tool_classification_lazy_cached_and_overridable():
    judge_calls = []

    def counting_judge(name, description):
        judge_calls.append(name)
        return _language_agnostic_judge(name, description)

    clf = ToolClassifier(
        judge=counting_judge,
        overrides={"transfer_funds": "READ"},  # operator override wins
    )

    # Override beats the judge
    assert clf.classify("transfer_funds", "Transfiere fondos.") == "READ"
    assert "transfer_funds" not in judge_calls

    # Judge used for unknown tools (Spanish description — heuristic-proof)
    assert clf.classify("herramienta_opaca", "Ejecuta la orden de compra.") == "ACTION"
    assert judge_calls == ["herramienta_opaca"]

    # Cached: second classification does not re-invoke the judge
    clf.classify("herramienta_opaca", "Ejecuta la orden de compra.")
    assert judge_calls == ["herramienta_opaca"]

    # Audit artifact is inspectable, with provenance per tool
    table = clf.export()
    assert table["transfer_funds"]["source"] == "override"
    assert table["herramienta_opaca"]["source"] == "llm_judge"
    assert table["herramienta_opaca"]["classification"] == "ACTION"
