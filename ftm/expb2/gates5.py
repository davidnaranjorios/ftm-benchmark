"""
Per-turn 5-way governance pipeline for R2a/R2b (Exp B v2).

Reuses the vendored mech_gov primitives (ftm.expb.gates) on the case's REAL
field vector (no neutral-constant mapping needed in v2 — cases are native).

Deviation (declared): mech_gov applies hard gates pre-LLM; v2 consults the
model first so model_decision is always recorded, then applies:
  1. hard gates on the static case            → forced decision
  2. ambiguity_gate on static (R2a) or model-re-estimated (R2b) completeness
  3. otherwise final = model_decision
"""
from __future__ import annotations

from ftm.expb.gates import GovCase, ambiguity_gate, build_default_gates, evaluate_hard_gates

_GATES = build_default_gates()


def gov_case(case: dict, completeness_override: float | None = None) -> GovCase:
    return GovCase(
        case_id=case["case_id"],
        risk_score=float(case["risk_score"]),
        completeness=float(
            completeness_override if completeness_override is not None
            else case["completeness"]
        ),
        regulatory_flags=list(case["regulatory_flags"]),
        amount_usd=float(case["amount_usd"]),
    )


def apply_pipeline(
    case: dict, model_decision: str, mode: str,
    completeness_reported: float | None = None,
) -> dict:
    """mode: '2a' (static completeness) | '2b' (re-estimated when reported).

    Returns {final_decision, gate_id, gate_intervened, forced_decision,
             completeness_used, ambiguity_static_would_fire}."""
    if mode not in ("2a", "2b"):
        raise ValueError("mode must be '2a' or '2b'")

    hard = evaluate_hard_gates(gov_case(case), _GATES)
    static_amb = ambiguity_gate(gov_case(case))

    if mode == "2b" and completeness_reported is not None:
        comp_used = float(completeness_reported)
    else:
        comp_used = float(case["completeness"])

    gate_id = forced = None
    if hard is not None:
        gate_id, forced_dec, _ = hard
        forced = forced_dec.value
    else:
        amb = (static_amb if mode == "2a"
               else ambiguity_gate(gov_case(case, completeness_override=comp_used)))
        if amb is not None:
            gate_id, forced = "ambiguity", amb.value

    final = forced if forced is not None else model_decision
    return {
        "final_decision": final,
        "gate_id": gate_id,
        "gate_intervened": final != model_decision,
        "forced_decision": forced,
        "completeness_used": round(comp_used, 3),
        # H3 counterfactual: would the static ambiguity gate have fired?
        "ambiguity_static_would_fire": (
            hard is None and static_amb is not None
        ),
    }
