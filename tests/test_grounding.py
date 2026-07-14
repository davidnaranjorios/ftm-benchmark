"""
Offline tests for the Santander grounding source and the ftm_banking_v0
pack. No network, no LLM: templates only plus MockAdapter for the pilot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftm.adapters import MockAdapter
from ftm.engine import (
    Scenario,
    build_ftm_system_prompt,
    build_turn_user_message,
    parse_decision,
)
from ftm.grounding.santander import (
    CELLS,
    CORE_PAIRS,
    DETERMINISTIC_ANCHORS,
    SUPPLEMENTARY_STRESSED_ONLY_PAIRS,
    PolicyAnchor,
    SantanderGroundedGenerator,
    build_pack,
    load_pack_scenarios,
    load_sgcd_subset,
    subset_sha256,
)
from ftm.scenario_gen import GroundTruthSpec, _numeric_guard


def _gen(seed: int = 42) -> SantanderGroundedGenerator:
    return SantanderGroundedGenerator(seed=seed)


def _spec_from_record(rec: dict) -> GroundTruthSpec:
    gts = dict(rec["ground_truth_spec"])
    gts.pop("source")
    return GroundTruthSpec(**gts)


# ── (1) Numeric guard ─────────────────────────────────────────────────────────

def test_all_scenarios_pass_numeric_guard():
    result = _gen().generate(n_per_cell=8)
    by_id = {s.scenario_id: s for s in result.scenarios}
    assert len(by_id) == len(result.scenarios)  # unique ids
    for rec in result.manifest["scenarios"]:
        scenario = by_id[rec["scenario_id"]]
        spec = _spec_from_record(rec)
        assert _numeric_guard(scenario.event, spec), scenario.scenario_id


def test_reading_threshold_comparator_consistency():
    result = _gen().generate(n_per_cell=8)
    for rec in result.manifest["scenarios"]:
        spec = _spec_from_record(rec)
        if spec.comparator == "reading_above_threshold":
            assert spec.reading > spec.threshold
        else:
            assert spec.reading < spec.threshold


# ── (2) Provenance ────────────────────────────────────────────────────────────

def test_provenance_recorded_per_scenario():
    result = _gen().generate(n_per_cell=8)
    subset_ids = {r["id"] for r in load_sgcd_subset()}
    for rec in result.manifest["scenarios"]:
        src = rec["ground_truth_spec"]["source"]
        policy, reading = src["policy"], src["reading"]
        assert policy["repo"] == "SantanderAI/mech-gov-framework"
        assert policy["license"] == "Apache-2.0"
        assert policy["rule_id"] == rec["ground_truth_spec"]["policy_id"]
        assert policy["layer"] == "deterministic"
        assert reading["repo"] == "SantanderAI/sota-stressed-datasets"
        assert reading["license"] == "CC-BY-4.0"
        assert reading["record_id"] in subset_ids
        assert reading["derivation"] in (
            "band_sampled_from_label", "band_sampled_from_shock"
        )
        assert reading["derivation_seed"] == 42
        assert reading["verbatim"] is False


def test_manifest_declares_grounding_and_exclusions():
    m = _gen().generate(n_per_cell=8).manifest
    g = m["grounding"]
    assert g["ambiguous_layer_excluded"] is True
    assert "credit_amount" in g["reading_pool_exclusions"]
    assert "DET-05" in g["excluded_anchors"]
    assert g["reading_source"]["subset_sha256"] == subset_sha256()
    assert g["reading_source"]["mother_dataset_sha256"]
    assert m["lang"] == "en"


# ── (3) 2x2 design ────────────────────────────────────────────────────────────

def _core_records(manifest: dict) -> list[dict]:
    return [r for r in manifest["scenarios"] if r["grounding_cell"] is not None]


def _supp_records(manifest: dict) -> list[dict]:
    return [r for r in manifest["scenarios"] if r["grounding_cell"] is None]


def test_2x2_labels_all_four_cells():
    result = _gen().generate(n_per_cell=8)
    counts = result.manifest["design_2x2"]["cell_counts"]
    assert set(counts) == set("ABCD")
    assert all(v == 8 for v in counts.values())
    for rec in _core_records(result.manifest):
        cell = rec["grounding_cell"]
        assert rec["grounding_axes"] == CELLS[cell]
        assert rec["supplementary_stressed_only"] is False
        # data axis consistent with the SGCD shock of the source record
        if rec["grounding_axes"]["data_condition"] == "clean":
            assert rec["sgcd_shock"] == "F0"
        else:
            assert rec["sgcd_shock"] != "F0"


def test_pressure_axis_maps_to_schedule_category():
    result = _gen().generate(n_per_cell=8)
    by_id = {s.scenario_id: s for s in result.scenarios}
    for rec in _core_records(result.manifest):
        scenario = by_id[rec["scenario_id"]]
        if rec["grounding_axes"]["social_pressure"] == "low":
            assert scenario.schedule_category == "control"
        else:
            assert scenario.schedule_category != "control"


def test_anchor_composition_identical_across_cells():
    """The (anchor, condition) multiset must be IDENTICAL in A, B, C, D —
    otherwise the clean/stressed axis is confounded with anchor type."""
    result = _gen().generate(n_per_cell=8)
    multisets: dict[str, list[tuple[str, str]]] = {c: [] for c in CELLS}
    for rec in _core_records(result.manifest):
        spec = rec["ground_truth_spec"]
        multisets[rec["grounding_cell"]].append(
            (spec["policy_id"], "act" if spec["optimal"] == "ACT" else "stay")
        )
    reference = sorted(multisets["A"])
    assert reference == sorted(CORE_PAIRS)
    for cell in "BCD":
        assert sorted(multisets[cell]) == reference, cell
    # 50/50 stay/act within every cell
    for cell, pairs in multisets.items():
        n_act = sum(1 for _, cond in pairs if cond == "act")
        assert n_act * 2 == len(pairs), cell


def test_core_pairs_feasible_in_all_cells():
    """No completeness-ACT pair (infeasible on clean F0 records) may be in
    the 2x2 core pool."""
    anchors = {a.rule_id: a for a in DETERMINISTIC_ANCHORS}
    for rule_id, condition in CORE_PAIRS:
        anchor = anchors[rule_id]
        assert not (anchor.metric == "completeness" and condition == "act"), (
            rule_id, condition,
        )
    assert ("DET-08", "stay") not in CORE_PAIRS
    assert ("DET-08", "act") not in CORE_PAIRS


def test_supplementary_set_labelled_and_outside_2x2():
    result = _gen().generate(n_per_cell=8)
    supp = _supp_records(result.manifest)
    assert len(supp) == 2 * len(SUPPLEMENTARY_STRESSED_ONLY_PAIRS)
    for rec in supp:
        assert rec["supplementary_stressed_only"] is True
        assert rec["grounding_cell"] is None
        assert rec["grounding_axes"] is None
        assert rec["sgcd_shock"] != "F0"  # stressed records only
    block = result.manifest["supplementary_stressed_only"]
    assert block["n_scenarios"] == len(supp)
    assert sorted(map(tuple, block["pairs"])) == sorted(
        SUPPLEMENTARY_STRESSED_ONLY_PAIRS
    )
    rule = result.manifest["design_2x2"]["anchor_composition_rule"]
    assert "held constant across cells" in rule


def test_n_per_cell_must_preserve_composition():
    with pytest.raises(ValueError, match="multiple"):
        _gen().generate(n_per_cell=6)


# ── (4) Determinism ───────────────────────────────────────────────────────────

def test_same_seed_same_specs():
    a = _gen(seed=7).generate(n_per_cell=8)
    b = _gen(seed=7).generate(n_per_cell=8)
    assert a.manifest["scenarios"] == b.manifest["scenarios"]
    assert [s.event for s in a.scenarios] == [s.event for s in b.scenarios]


def test_different_seed_different_readings():
    a = _gen(seed=1).generate(n_per_cell=8)
    b = _gen(seed=2).generate(n_per_cell=8)
    ra = [r["ground_truth_spec"]["reading"] for r in a.manifest["scenarios"]]
    rb = [r["ground_truth_spec"]["reading"] for r in b.manifest["scenarios"]]
    assert ra != rb


# ── (5) AMBIGUOUS-layer rejection ─────────────────────────────────────────────

def test_ambiguous_anchor_rejected():
    ambiguous = PolicyAnchor(
        rule_id="AMB-01",
        metric="risk_score",
        threshold=0.6,
        trigger="above",
        decision="ESCALATE",
        description="ambiguous-layer anchor (must be rejected)",
        layer="ambiguous",
    )
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        SantanderGroundedGenerator(
            anchors=DETERMINISTIC_ANCHORS + [ambiguous]
        )


def test_default_anchors_are_all_deterministic():
    assert all(a.layer == "deterministic" for a in DETERMINISTIC_ANCHORS)


# ── Tool vocabulary override ──────────────────────────────────────────────────

def test_agent_profile_tools_replace_default_vocabulary():
    tools = ["freeze_account", "notify_compliance"]
    result = SantanderGroundedGenerator(tool_names=tools).generate(n_per_cell=8)
    used = {r["ground_truth_spec"]["tool_name"] for r in result.manifest["scenarios"]}
    assert used <= set(tools)
    assert result.manifest["tool_vocabulary"] == tools


# ── Committed pack integrity ──────────────────────────────────────────────────

PACK_DIR = Path(__file__).parent.parent / "scenarios" / "packs" / "ftm_banking_v0"


def test_committed_pack_matches_regeneration(tmp_path):
    assert PACK_DIR.exists(), "committed pack missing"
    rebuilt = build_pack(out_dir=tmp_path / "pack")
    for name in ("scenarios.jsonl", "manifest.json", "ATTRIBUTION.md"):
        assert (rebuilt / name).read_text() == (PACK_DIR / name).read_text(), name


def test_committed_pack_attribution_mentions_both_sources():
    text = (PACK_DIR / "ATTRIBUTION.md").read_text()
    assert "Santander AI Lab" in text
    assert "UCI Machine Learning Repository" in text
    assert "CC BY 4.0" in text
    assert "Apache License 2.0" in text


# ── Pilot: full pipeline over the pack with MockAdapter, zero tokens ─────────

def test_pilot_snapshot_run_with_mock_adapter():
    scenarios = load_pack_scenarios(PACK_DIR)
    assert scenarios and all(isinstance(s, Scenario) for s in scenarios)
    adapter = MockAdapter()
    snapshot = scenarios[:5]  # tier-snapshot-sized slice
    for scenario in snapshot:
        system = build_ftm_system_prompt(scenario)
        messages: list[dict] = []
        decisions = []
        for t in range(1, len(scenario.pressure_turns) + 2):
            messages.append(
                {"role": "user", "content": build_turn_user_message(scenario, t)}
            )
            result = adapter.complete(system, messages)
            parsed = parse_decision(result["text"])
            assert parsed["decision"] in ("ACT", "STAY")
            decisions.append(parsed["decision"])
            messages.append({"role": "assistant", "content": result["text"]})
        # MockAdapter keys off the turn-1 event text, so its decision must
        # equal the constructed optimal on every turn — pipeline end-to-end.
        assert decisions == [scenario.optimal] * len(decisions), (
            scenario.scenario_id
        )
