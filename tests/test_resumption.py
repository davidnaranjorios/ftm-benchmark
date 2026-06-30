"""
Resumption acceptance test.

Verifies that:
1. A 3-turn run interrupted after turn 2 continues at turn 3 on resume.
2. No turn is duplicated in the checkpoint after resumption.
3. Final metrics (and per-turn decisions) are bit-for-bit identical to an
   uninterrupted 3-turn run with the same MockAdapter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ftm.adapters import MockAdapter
from ftm.engine import TurnResult, compute_metrics, generate_scenarios
from ftm.runner import RunConfig, load_checkpoint, run


# ── Helper: run inside an isolated temp directory ────────────────────────────

def _run_in(tmpdir: Path, config: RunConfig, adapter: MockAdapter) -> dict:
    old = os.getcwd()
    os.chdir(tmpdir)
    try:
        return run(config, adapter)
    finally:
        os.chdir(old)


def _load_checkpoint_in(tmpdir: Path, run_id: str) -> list[TurnResult]:
    path = tmpdir / "checkpoints" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(TurnResult(**json.loads(line)))
    return results


# ── Fixture: single devops snapshot scenario ──────────────────────────────────

@pytest.fixture(scope="module")
def target_scenario():
    scenarios = generate_scenarios(tier="snapshot", domain="devops_server")
    assert scenarios, "generate_scenarios returned empty list"
    return scenarios[0]


# ── Acceptance test ───────────────────────────────────────────────────────────

def test_resumption_no_duplication_and_metrics_identical(tmp_path, target_scenario):
    """
    Simulates: run 2 turns → crash → resume → turn 3 runs exactly once.
    Metrics must match an uninterrupted 3-turn run.
    """
    adapter = MockAdapter()
    sid = target_scenario.scenario_id

    # ── Phase 1: run turns 1-2 (simulates process killed after turn 2) ───────
    cfg_p1 = RunConfig(
        models=["mock"],
        tier="snapshot",
        domain="devops_server",
        run_id="test_interrupted",
        adapter="mock",
        max_turns=2,
    )
    _run_in(tmp_path, cfg_p1, adapter)

    cp_after_p1 = _load_checkpoint_in(tmp_path, "test_interrupted")
    our_turns_p1 = [r for r in cp_after_p1 if r.scenario_id == sid]

    assert len(our_turns_p1) == 2, (
        f"Expected 2 checkpoint records after phase 1, got {len(our_turns_p1)}"
    )
    assert {r.turn for r in our_turns_p1} == {1, 2}

    # ── Phase 2: resume — continue to turn 3 ─────────────────────────────────
    cfg_p2 = RunConfig(
        models=["mock"],
        tier="snapshot",
        domain="devops_server",
        run_id="test_interrupted",   # same run_id → picks up checkpoint
        adapter="mock",
        max_turns=3,
    )
    _run_in(tmp_path, cfg_p2, adapter)

    cp_after_p2 = _load_checkpoint_in(tmp_path, "test_interrupted")
    our_turns_p2 = [r for r in cp_after_p2 if r.scenario_id == sid]
    turn_numbers = sorted(r.turn for r in our_turns_p2)

    # No duplicates, exactly 3 turns
    assert turn_numbers == [1, 2, 3], (
        f"Expected checkpoint turns [1,2,3], got {turn_numbers}"
    )
    assert len(turn_numbers) == len(set(turn_numbers)), "Duplicate turns in checkpoint"

    # Turn 1 and 2 records must be byte-identical to what was written in phase 1
    p1_by_turn = {r.turn: r for r in our_turns_p1}
    p2_by_turn = {r.turn: r for r in our_turns_p2}
    for t in (1, 2):
        assert p1_by_turn[t].decision == p2_by_turn[t].decision, (
            f"Turn {t} decision changed on resume"
        )
        assert p1_by_turn[t].raw_response == p2_by_turn[t].raw_response, (
            f"Turn {t} raw_response changed on resume"
        )

    # ── Phase 3: uninterrupted 3-turn run for comparison ─────────────────────
    cfg_fresh = RunConfig(
        models=["mock"],
        tier="snapshot",
        domain="devops_server",
        run_id="test_fresh",
        adapter="mock",
        max_turns=3,
    )
    _run_in(tmp_path, cfg_fresh, adapter)

    cp_fresh = _load_checkpoint_in(tmp_path, "test_fresh")
    fresh_turns = sorted(
        [r for r in cp_fresh if r.scenario_id == sid], key=lambda r: r.turn
    )
    assert len(fresh_turns) == 3, f"Expected 3 fresh turns, got {len(fresh_turns)}"

    # ── Compare decisions turn-by-turn ────────────────────────────────────────
    interrupted_turns = sorted(our_turns_p2, key=lambda r: r.turn)
    for t_int, t_fresh in zip(interrupted_turns, fresh_turns):
        assert t_int.decision == t_fresh.decision, (
            f"Turn {t_int.turn}: interrupted={t_int.decision}, fresh={t_fresh.decision}"
        )
        assert t_int.is_correct == t_fresh.is_correct, (
            f"Turn {t_int.turn}: is_correct differs"
        )

    # ── Compare computed metrics ──────────────────────────────────────────────
    m_int = compute_metrics(interrupted_turns)
    m_fresh = compute_metrics(fresh_turns)

    for field_name in ("stay_acc", "act_acc", "dis", "abi", "composite", "farp_rate", "overall_accuracy"):
        v_int = getattr(m_int, field_name)
        v_fresh = getattr(m_fresh, field_name)
        assert v_int == v_fresh, (
            f"Metric '{field_name}': interrupted={v_int}, fresh={v_fresh}"
        )
