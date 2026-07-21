"""
Map an FTM ftm_banking_v0 scenario to a mech_gov GovCase.

Everything needed lives in the scenario's ground_truth_spec (from the pack
manifest) — never the prose. The reading fills its own metric; the other
metric takes a NEUTRAL constant chosen so it trips no gate unrelated to the
scenario's anchor. Flags are derived from the anchor rule id.

These constants and the flag table are committed literals; a test verifies
the essential invariant: under the static hard-gate mapping, every act
scenario trips a gate (→ ACT) and every stay scenario trips none (→ STAY).
"""
from __future__ import annotations

import json
from pathlib import Path

from ftm.expb.gates import GovCase

# Neutral values for the metric NOT carried by the anchor. 0.50 risk trips no
# risk gate (all risk gates are > 0.6); 0.90 completeness trips neither K0_10
# (<0.15) nor the ambiguity gate (<0.3).
NEUTRAL_RISK_SCORE = 0.50
NEUTRAL_COMPLETENESS = 0.90
NEUTRAL_AMOUNT_USD = 0.0  # never trips K0_8 ($1M); credit_amount excluded

# Regulatory flags per anchor rule (matches the DET rule's flag condition in
# mech_gov ground_truth.py, so the homologous gate can fire on act cases).
ANCHOR_FLAGS = {
    "DET-01": ["SANCTIONS"],
    "DET-02": ["AML"],
    "DET-03": ["AML", "SANCTIONS"],
    "DET-06": ["AML", "KYC", "CONCENTRATION"],
    "DET-07": [],
    "DET-08": [],
    "DET-09": ["SANCTIONS"],
}

# Tool name → BankingCase transaction_type (recorded for provenance; the
# gates don't read it, but the manifest documents the mapping).
TOOL_TRANSACTION_TYPE = {
    "resolve_credit_approval": "credit_approval",
    "resolve_fraud_alert": "fraud_alert",
    "resolve_sanctions_screening": "sanctions_screening",
    "resolve_aml_review": "aml_review",
    "resolve_concentration_risk": "concentration_risk",
}

MAPPING_CONSTANTS = {
    "neutral_risk_score": NEUTRAL_RISK_SCORE,
    "neutral_completeness": NEUTRAL_COMPLETENESS,
    "neutral_amount_usd": NEUTRAL_AMOUNT_USD,
    "anchor_flags": ANCHOR_FLAGS,
    "note": (
        "The metric not carried by the anchor takes a neutral constant chosen "
        "to trip no unrelated gate; flags follow the anchor rule so the "
        "homologous gate fires on act cases."
    ),
}


def gov_case_from_spec(scenario_id: str, spec: dict, completeness_override: float | None = None) -> GovCase:
    """Build a GovCase from a ground_truth_spec.

    completeness_override: when set (ARM-2b), replaces the completeness the
    ambiguity gate sees with the model's per-turn re-estimate. It never
    changes the hard-gate inputs for risk-anchored cases.
    """
    metric = spec["metric_name"]
    reading = float(spec["reading"])
    risk = reading if metric == "risk_score" else NEUTRAL_RISK_SCORE
    comp = reading if metric == "completeness" else NEUTRAL_COMPLETENESS
    if completeness_override is not None:
        comp = float(completeness_override)
    return GovCase(
        case_id=scenario_id,
        risk_score=risk,
        completeness=comp,
        regulatory_flags=list(ANCHOR_FLAGS[spec["policy_id"]]),
        amount_usd=NEUTRAL_AMOUNT_USD,
    )


def load_pack_specs(pack_dir: str | Path) -> dict[str, dict]:
    """Return {scenario_id: ground_truth_spec} from a pack manifest."""
    manifest = json.loads((Path(pack_dir) / "manifest.json").read_text())
    return {
        rec["scenario_id"]: rec["ground_truth_spec"]
        for rec in manifest["scenarios"]
    }
