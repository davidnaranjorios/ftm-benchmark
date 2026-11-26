"""Offline tests for the Inspect task (mockllm, zero tokens)."""
from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from ftm.adapters import MockAdapter
from ftm.engine import compute_metrics, generate_scenarios
from ftm_inspect import DATA_ANCHORING_PROMPT, ftm, ftm_dataset
from ftm_inspect.scorer import _trajectory_summary


def _out(text: str) -> ModelOutput:
    # Explicit usage keeps mockllm from loading a tokenizer (fully offline).
    out = ModelOutput.from_content(model="mockllm", content=text)
    out.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    return out


def _policy(capitulate_at=None, parse_fail_at=None, recover_at=None):
    """Mock model: answers the scenario's optimal decision (keyword rule of
    the engine's MockAdapter), but on STAY scenarios switches to ACT from
    `capitulate_at` (and back to STAY from `recover_at`)."""
    oracle = MockAdapter()

    def generate(messages, tools, tool_choice, config):
        users = [m for m in messages if m.role == "user"]
        turn = len(users)
        optimal = oracle.complete("", [{"role": "user", "content": users[0].text}])["text"]
        decision = "ACT" if "DECISION: ACT" in optimal else "STAY"
        if decision == "STAY" and capitulate_at and turn >= capitulate_at:
            if not (recover_at and turn >= recover_at):
                decision = "ACT"
        if parse_fail_at and turn == parse_fail_at:
            text = "I need more information."
        else:
            text = f"DECISION: {decision}\nCONFIDENCE: 8\nReason: metrics within threshold."
        return _out(text)

    return generate


import tempfile
from pathlib import Path

_LOG_DIR = Path(tempfile.mkdtemp(prefix="ftm-inspect-test-"))


def _run(policy, **task_args):
    model = get_model("mockllm/model", custom_outputs=policy)
    [log] = inspect_eval(ftm(**task_args), model=model, display="none", log_dir=str(_LOG_DIR))
    assert log.status == "success", log.error
    return log


def _metrics(log):
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


def test_dataset_matches_engine():
    ds = ftm_dataset("standard")
    engine = generate_scenarios("standard")
    assert len(ds) == len(engine) == 30
    assert [s.id for s in ds] == [s.scenario_id for s in engine]
    assert {s.target for s in ds} == {"STAY", "ACT"}
    assert len(ftm_dataset("standard", condition="stay", schedule="control_flat")) == 5


def test_pressure_insensitive_model_scores_zero_farp():
    log = _run(_policy(), tier="standard")
    m = _metrics(log)
    assert m["farp_strict"] == 0.0
    assert m["inaction_rate"] == 0.0
    assert m["accuracy"] == 1.0


def test_capitulation_turn_is_recorded():
    log = _run(_policy(capitulate_at=3), tier="standard", condition="stay")
    m = _metrics(log)
    assert m["farp_strict"] == 1.0
    for sample in log.samples:
        md = sample.scores["decision_persistence"].metadata
        if md["initial_decision"] == "STAY":
            assert md["first_error_turn"] == 3
            assert md["first_flip_turn"] == 3
            assert md["recovered"] is False
            assert sample.scores["decision_persistence"].value == "I"
            assert len(md["turns"]) == 10


def test_metrics_match_engine_on_same_turns():
    log = _run(_policy(capitulate_at=4, recover_at=8), tier="standard")
    from ftm_inspect.scorer import turn_results_from_scores
    from inspect_ai.scorer import SampleScore

    scores = [SampleScore(score=s.scores["decision_persistence"], sample_id=s.id) for s in log.samples]
    eng = compute_metrics(turn_results_from_scores(scores))
    m = _metrics(log)
    assert m["farp_strict"] == pytest.approx(eng.v10["farp_strict"])
    assert m["crs"] == pytest.approx(eng.composite)
    assert m["inaction_rate"] == pytest.approx(eng.inaction_rate)
    recovered = [s for s in log.samples
                 if s.scores["decision_persistence"].metadata["recovered"]]
    assert recovered, "recover_at=8 should produce recovered trajectories"


def test_parse_fail_is_not_a_decision():
    log = _run(_policy(parse_fail_at=2), tier="snapshot")
    m = _metrics(log)
    assert m["parse_fail_rate"] == pytest.approx(0.1)
    for sample in log.samples:
        md = sample.scores["decision_persistence"].metadata
        assert md["n_parse_fail"] == 1
        assert md["first_flip_turn"] is None


def test_data_anchoring_changes_only_the_system_prompt():
    seen = {}

    def spy(messages, tools, tool_choice, config):
        seen.setdefault("system", messages[0].text)
        return _out("DECISION: STAY\nCONFIDENCE: 5\nReason: x")

    _run(spy, tier="snapshot", max_turns=1, data_anchoring=True)
    assert seen["system"].endswith(DATA_ANCHORING_PROMPT)
    seen.clear()
    _run(spy, tier="snapshot", max_turns=1)
    assert DATA_ANCHORING_PROMPT not in seen["system"]


def test_pressure_turns_follow_schedule():
    prompts = []

    def spy(messages, tools, tool_choice, config):
        prompts.append(messages[-1].text)
        return _out("DECISION: STAY\nCONFIDENCE: 5\nReason: x")

    _run(spy, tier="standard", condition="stay", schedule="shock_immediate", domain="medical")
    assert len(prompts) == 10
    assert prompts[0].startswith("Situation:")
    assert "No change to objective data" in prompts[1]
    assert "[emotional, temporal, hierarchical, peer, reputational, ambiguity]" in prompts[1]


def test_trajectory_summary_ignores_parse_fail():
    turns = [{"turn": i + 1, "decision": d} for i, d in
             enumerate(["STAY", "PARSE_FAIL", "STAY", "ACT", "STAY"])]
    s = _trajectory_summary(turns, "STAY")
    assert s["first_error_turn"] == 4 and s["first_flip_turn"] == 4
    assert s["recovered"] is True and s["n_parse_fail"] == 1
