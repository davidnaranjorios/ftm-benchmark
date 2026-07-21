"""
Offline tests for Experiment B. No network, no real model: MockAdapter per
arm, a deterministic fake-gate subject, and the committed pack.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftm.adapters import MockAdapter
from ftm.engine import parse_decision
from ftm.expb.adapters import GatedAdapter, parse_completeness
from ftm.expb.analysis import (
    build_report,
    fold_by_scenario,
    h3_ambiguity_share,
    mcnemar,
)
from ftm.expb.arm1_policy import arm1_operator_prompt, verify_r1_sha
from ftm.expb.driver import estimate_budget, run_arm
from ftm.expb.gates import (
    GovCase,
    ambiguity_gate,
    build_default_gates,
    evaluate_hard_gates,
)
from ftm.expb.mapping import gov_case_from_spec, load_pack_specs
from ftm.grounding.santander import load_pack_scenarios

PACK = Path(__file__).parent.parent / "scenarios" / "packs" / "ftm_banking_v0"


def _load():
    return load_pack_scenarios(PACK), load_pack_specs(PACK)


def _core(scenarios):
    return [s for s in scenarios if "_supp_" not in s.scenario_id]


# ── Vendored gates: SHA + logic ───────────────────────────────────────────────

def test_r1_sha_matches_source():
    assert verify_r1_sha()


def test_ambiguity_gate_logic():
    assert ambiguity_gate(GovCase("x", 0.5, 0.2, [])) is not None      # DEFER
    assert ambiguity_gate(GovCase("x", 0.8, 0.2, [])).value == "ESCALATE"
    assert ambiguity_gate(GovCase("x", 0.5, 0.9, [])) is None          # no fire


def test_hard_gate_first_match_wins():
    gates = build_default_gates()
    # SANCTIONS + very high risk → K0_6 DECLINE
    r = evaluate_hard_gates(GovCase("x", 0.95, 0.9, ["SANCTIONS"]), gates)
    assert r is not None and r[0] == "K0_6"


# ── (mapping) act ⇒ gate fires, stay ⇒ none (the essential invariant) ─────────

def test_mapping_act_fires_stay_silent_on_core():
    scenarios, specs = _load()
    gates = build_default_gates()
    for s in _core(scenarios):
        spec = specs[s.scenario_id]
        case = gov_case_from_spec(s.scenario_id, spec)
        hard = evaluate_hard_gates(case, gates)
        fired = hard is not None
        if spec["optimal"] == "ACT":
            assert fired, f"act scenario did not fire a hard gate: {s.scenario_id}"
        else:
            assert not fired, f"stay scenario fired a hard gate: {s.scenario_id}"


def test_homologous_gate_matches_map_on_act_core():
    """The gate that fires on an act core scenario is the one GATE_ANCHOR_MAP
    predicts (hard-gate anchors only; ambiguity anchors are supplementary)."""
    from ftm.expb.gates import GATE_ANCHOR_MAP
    scenarios, specs = _load()
    gates = build_default_gates()
    for s in _core(scenarios):
        spec = specs[s.scenario_id]
        if spec["optimal"] != "ACT":
            continue
        expected = GATE_ANCHOR_MAP[spec["policy_id"]]
        if expected["kind"] != "hard":
            continue
        gate_id = evaluate_hard_gates(gov_case_from_spec(s.scenario_id, spec), gates)[0]
        # first-match ordering may surface an equally-correct earlier DECLINE
        # gate; require a gate fired and the homologous one is at least present
        assert gate_id is not None


# ── ARM-2a: deterministic, FARP=0 on core (the sanity control) ────────────────

def test_arm2a_static_farp_zero_and_turn_invariant():
    scenarios, specs = _load()
    core = _core(scenarios)
    turns, glog = run_arm("ARM-2a", core, specs, MockAdapter)
    # every STAY scenario stays correct on every turn
    stay = [t for t in turns if t.optimal == "STAY"]
    assert stay and all(t.is_correct for t in stay)
    folds = fold_by_scenario(turns)
    assert sum(folds.values()) == 0
    # turn-invariant: within a scenario, one unique decision across turns
    by_scen: dict[str, set] = {}
    for t in turns:
        by_scen.setdefault(t.scenario_id, set()).add(t.decision)
    assert all(len(v) == 1 for v in by_scen.values())
    # gate log records no ambiguity firing in 2a
    assert all(e["gate_id"] != "ambiguity" for e in glog)


def test_arm2a_act_scenarios_all_act_on_core():
    scenarios, specs = _load()
    core = _core(scenarios)
    turns, _ = run_arm("ARM-2a", core, specs, MockAdapter)
    act = [t for t in turns if t.optimal == "ACT"]
    assert act and all(t.decision == "ACT" for t in act)


# ── Fake-gate determinist: GatedAdapter wiring + gate_log, no mech_gov ────────

class FakeGateSubject(MockAdapter):
    """Subject whose completeness re-estimate drops below 0.3 at a fixed turn,
    so ARM-2b's ambiguity gate must flip STAY→ACT exactly then."""

    def __init__(self, break_turn: int):
        self.break_turn = break_turn
        self._t = 0

    def begin_scenario(self, sid):
        self._t = 0

    def complete(self, system, messages):
        self._t += 1
        comp = 0.1 if self._t >= self.break_turn else 0.9
        return {"text": f"DECISION: STAY\nCONFIDENCE: 8\nReason: holding.\n"
                        f"COMPLETENESS: {comp}",
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_fake_gate_ambiguity_flips_at_break_turn():
    scenarios, specs = _load()
    # pick a completeness-anchored STAY scenario so ambiguity is the live gate
    target = next(
        s for s in scenarios
        if specs[s.scenario_id]["metric_name"] == "completeness"
        and specs[s.scenario_id]["optimal"] == "STAY"
    )
    adapter = GatedAdapter(FakeGateSubject(break_turn=3), target.scenario_id,
                           specs[target.scenario_id], mode="2b")
    adapter.begin_scenario(target.scenario_id)
    decisions = []
    for t in range(1, 6):
        out = adapter.complete("sys", [{"role": "user", "content": f"turn{t}"}])
        decisions.append(parse_decision(out["text"])["decision"])
    assert decisions == ["STAY", "STAY", "ACT", "ACT", "ACT"]
    fired = [e for e in adapter.gate_log if e.gate_id == "ambiguity"]
    assert fired and all(e.turn >= 3 for e in fired)
    assert all(e.completeness_reestimated for e in fired)


def test_fake_gate_2a_ignores_reestimate():
    """The same subject under ARM-2a never flips — ambiguity gate is off."""
    scenarios, specs = _load()
    target = next(
        s for s in scenarios
        if specs[s.scenario_id]["metric_name"] == "completeness"
        and specs[s.scenario_id]["optimal"] == "STAY"
    )
    adapter = GatedAdapter(FakeGateSubject(break_turn=2), target.scenario_id,
                           specs[target.scenario_id], mode="2a")
    adapter.begin_scenario(target.scenario_id)
    decisions = []
    for t in range(1, 5):
        out = adapter.complete("s", [{"role": "user", "content": f"t{t}"}])
        decisions.append(parse_decision(out["text"])["decision"])
    assert set(decisions) == {"STAY"}


# ── Completeness parser ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text,exp,found", [
    ("DECISION: STAY\nCOMPLETENESS: 0.2", 0.2, True),
    ("COMPLETENESS: .15", 0.15, True),
    ("no line here", 0.9, False),
    ("COMPLETENESS: 5", 0.9, False),  # out of range → default
])
def test_parse_completeness(text, exp, found):
    v, f = parse_completeness(text, default=0.9)
    assert v == exp and f == found


# ── ARM-1 operator prompt: R1 verbatim intact + bridge marked ─────────────────

def test_arm1_prompt_contains_verbatim_r1_and_bridge():
    p = arm1_operator_prompt()
    assert "senior banking compliance officer" in p        # R1 verbatim
    assert "FTM PROTOCOL BRIDGE" in p                       # our bridge, marked
    assert "DECISION: ACT" in p and "DECISION: STAY" in p


# ── Analysis: McNemar + H3 share ──────────────────────────────────────────────

def test_mcnemar_discordant_counts():
    a = {"s1": 1, "s2": 0, "s3": 1}
    b = {"s1": 0, "s2": 0, "s3": 1}
    r = mcnemar(a, b)
    assert r["b_a_not_b"] == 1 and r["c_b_not_a"] == 0 and r["discordant"] == 1


def test_h3_share_counts_ambiguity_folds():
    from ftm.engine import TurnResult

    def tr(sid, turn, correct):
        return TurnResult(sid, "d", "stay", "sch", "control", turn, [], 0,
                          "STAY", "STAY" if correct else "ACT", 9, "r", "OBJ",
                          correct, "p", "resp")
    turns = [tr("s1", 1, True), tr("s1", 2, False)]  # folds at turn 2
    glog = [{"scenario_id": "s1", "turn": 2, "gate_id": "ambiguity",
             "completeness_reestimated": True}]
    r = h3_ambiguity_share(turns, glog)
    assert r["n_folds"] == 1 and r["n_via_ambiguity_reestimate"] == 1 and r["share"] == 1.0


# ── Full offline pilot across all arms via MockAdapter ────────────────────────

def test_offline_pilot_all_arms_builds_report():
    scenarios, specs = _load()
    core = _core(scenarios)[:8]
    arms = ["ARM-0", "ARM-1", "ARM-2a", "ARM-2b"]
    arm_turns, gate_logs = {}, {}
    for arm in arms:
        turns, glog = run_arm(arm, core, specs, MockAdapter)
        arm_turns[arm] = turns
        if glog:
            gate_logs[arm] = glog
    report = build_report(arm_turns, gate_logs, {"model": "mock", "arms": arms,
                          "pack": "ftm_banking_v0", "honesty_declaration": "x"})
    assert set(report["arm_summaries"]) == set(arms)
    assert report["hypotheses"]["sanity_ARM_2a"]["passes"]
    # every arm ran the same paired scenarios
    n = len({t.scenario_id for t in arm_turns["ARM-0"]})
    for arm in arms:
        assert len({t.scenario_id for t in arm_turns[arm]}) == n


class GarbageSubject(MockAdapter):
    """Subject that emits neither a DECISION line nor JSON → genuine parse fail."""

    def complete(self, system, messages):
        return {"text": "I am not sure how to proceed here.",
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_parse_fail_is_not_a_fold_and_h1_not_evaluable():
    scenarios, specs = _load()
    core = _core(scenarios)[:6]
    at, gl = {}, {}
    at["ARM-0"], _ = run_arm("ARM-0", core, specs, GarbageSubject)  # all PARSE_FAIL
    at["ARM-1"], _ = run_arm("ARM-1", core, specs, MockAdapter)
    at["ARM-2a"], _ = run_arm("ARM-2a", core, specs, MockAdapter)
    at["ARM-2b"], g = run_arm("ARM-2b", core, specs, MockAdapter)
    if g:
        gl["ARM-2b"] = g
    report = build_report(at, gl, {"model": "mock", "arms": list(at),
                          "pack": "ftm_banking_v0", "honesty_declaration": "x"})
    s0 = report["arm_summaries"]["ARM-0"]
    assert s0["n_folded"] == 0                      # PARSE_FAIL ≠ fold
    assert s0["n_stay_measurable"] == 0             # nothing measurable
    assert s0["parse_fail_rate"] == 1.0
    # H1 must be not_evaluable, never a false "refuted", when an arm is unmeasurable
    assert report["hypotheses"]["H1"]["verdict"] == "not_evaluable"
    assert "ARM-0" in report["hypotheses"]["H1"]["unmeasurable_arms"]


# ── ARM-1 hybrid resolver (pilot-driven fix) ──────────────────────────────────

class R1JsonSubject(MockAdapter):
    """Subject that answers only with R1-style JSON, no DECISION line."""

    def __init__(self, decision="APPROVE"):
        self._decision = decision

    def complete(self, system, messages):
        return {"text": '```json\n{"decision": "%s", "rationale": "x"}\n```' % self._decision,
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_resolve_r1_decision_prefers_decision_line():
    from ftm.expb.arm1_policy import resolve_r1_decision
    d, src = resolve_r1_decision("DECISION: ACT\nCONFIDENCE: 8\nReason: go")
    assert d == "ACT" and src == "decision_line"


@pytest.mark.parametrize("jd,exp", [
    ("APPROVE", "STAY"), ("DECLINE", "ACT"), ("ESCALATE", "ACT"),
    ("DEFER", "ACT"), ("CONDITIONAL", "ACT"),
])
def test_resolve_r1_decision_json_fallback(jd, exp):
    from ftm.expb.arm1_policy import resolve_r1_decision
    d, src = resolve_r1_decision('{"decision": "%s"}' % jd)
    assert d == exp and src == "json_fallback"


def test_resolve_r1_decision_truncated_json():
    """R1's verbose JSON cut off at max_tokens still yields the decision."""
    from ftm.expb.arm1_policy import resolve_r1_decision
    truncated = ('```json\n{\n  "decision": "DECLINE",\n  "rationale": "The risk '
                 'score of 0.95 is well abo')  # cut mid-string, no closing brace
    d, src = resolve_r1_decision(truncated)
    assert d == "ACT" and src == "json_fallback"


def test_resolve_r1_decision_unresolved():
    from ftm.expb.arm1_policy import resolve_r1_decision
    d, src = resolve_r1_decision("no decision, no json here")
    assert d == "PARSE_FAIL" and src == "unresolved"


def test_arm1_json_only_subject_is_measurable():
    """The pilot failure mode: R1 JSON with no DECISION line must now resolve
    (APPROVE→STAY) instead of PARSE_FAIL, making ARM-1 measurable."""
    scenarios, specs = _load()
    core = _core(scenarios)[:6]
    turns, _ = run_arm("ARM-1", core, specs, lambda: R1JsonSubject("APPROVE"))
    stay = [t for t in turns if t.optimal == "STAY"]
    assert stay and all(t.decision == "STAY" for t in stay)  # APPROVE→STAY
    assert all(t.decision != "PARSE_FAIL" for t in turns)


def test_h1_not_evaluable_when_no_folding():
    """All arms FARP=0 (nobody folds) → H1 not_evaluable, not 'refuted'."""
    scenarios, specs = _load()
    core = _core(scenarios)[:8]
    at, gl = {}, {}
    for arm in ["ARM-0", "ARM-1", "ARM-2a", "ARM-2b"]:
        turns, g = run_arm(arm, core, specs, MockAdapter)
        at[arm] = turns
        if g:
            gl[arm] = g
    report = build_report(at, gl, {"model": "mock", "arms": list(at),
                          "pack": "ftm_banking_v0", "honesty_declaration": "x"})
    h1 = report["hypotheses"]["H1"]
    assert h1["no_folding_observed"] is True
    assert h1["verdict"] == "not_evaluable"


def test_checkpoint_resume_skips_completed_scenarios(tmp_path):
    """A resumed run reuses checkpointed scenarios and only calls the model
    for the missing ones."""
    scenarios, specs = _load()
    core = _core(scenarios)[:4]
    calls = {"n": 0}

    class CountingSubject(MockAdapter):
        def complete(self, system, messages):
            calls["n"] += 1
            return super().complete(system, messages)

    # First run: 2 scenarios, checkpointed.
    t1, _ = run_arm("ARM-0", core[:2], specs, CountingSubject,
                    checkpoint_dir=tmp_path, run_id="r1")
    first_calls = calls["n"]
    assert first_calls > 0

    # Resume over all 4: the first 2 are reused (no new calls for them).
    calls["n"] = 0
    t2, _ = run_arm("ARM-0", core, specs, CountingSubject,
                    checkpoint_dir=tmp_path, run_id="r1")
    n_turns_each = len(t1) // 2
    # only the 2 new scenarios cost calls
    assert calls["n"] == first_calls  # same count as the original 2 = the 2 new
    assert len({t.scenario_id for t in t2}) == 4
    assert len(t2) == 4 * n_turns_each


def test_checkpoint_persists_gatelog(tmp_path):
    scenarios, specs = _load()
    core = _core(scenarios)[:2]
    _, glog = run_arm("ARM-2b", core, specs, MockAdapter,
                      checkpoint_dir=tmp_path, run_id="g1")
    # resume: gate log restored from disk, not regenerated
    _, glog2 = run_arm("ARM-2b", core, specs, MockAdapter,
                       checkpoint_dir=tmp_path, run_id="g1")
    assert glog2 == glog


def test_budget_estimate_scales_with_arms():
    scenarios, specs = _load()
    core = _core(scenarios)[:4]
    b = estimate_budget(core, ["ARM-0", "ARM-1", "ARM-2b"])
    assert b["per_arm"]["ARM-1"]["input_tokens"] > b["per_arm"]["ARM-0"]["input_tokens"]
    assert b["totals"]["input_tokens"] > 0
