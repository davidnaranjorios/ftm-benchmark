"""
Vendored-minimal mech_gov governance primitives (Apache-2.0).

Transcribed verbatim (logic and thresholds) from mech-gov-framework at the
source files whose SHA-256 is pinned below, so Experiment B carries no
runtime dependency on the upstream package. Attribution is in the pack's
ATTRIBUTION.md and the repo NOTICE.

Source (SantanderAI/mech-gov-framework, Apache-2.0):
  src/mech_gov/governance/primitives/hard_gates.py
  src/mech_gov/governance/primitives/ambiguity_gate.py

Only the pieces Experiment B needs are transcribed: the pre-LLM hard gates
(evaluate_hard_gates + build_default_gates) and the post-LLM ambiguity gate.
CEFL, I6Q, E3 entropy and the privacy gate are NOT reproduced — they involve
LLM candidate generation or PII handling outside this experiment's scope.

A lightweight GovCase stands in for mech_gov's pydantic BankingCase: the
gates read only .risk_score, .completeness, .regulatory_flags, .amount_usd.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# SHA-256 of the upstream source files these primitives were transcribed
# from (verify with: sha256sum <file>). Recorded per-run in the manifest.
SOURCE_SHA256 = {
    "hard_gates.py": (
        "c36cca5cc75b28d3c9b5a1dfdec232079bea6d80cbffa447fd9b7fbdff06a4c3"
    ),
    "ambiguity_gate.py": (
        "49f3410d00971c6431542289bc586786b67349ce6f7127e468ff067249eec770"
    ),
}
SOURCE_REPO = "SantanderAI/mech-gov-framework"
SOURCE_LICENSE = "Apache-2.0"


class Decision(str, Enum):
    """mech_gov Decision enum (5 values), transcribed."""

    APPROVE = "APPROVE"
    CONDITIONAL = "CONDITIONAL"
    ESCALATE = "ESCALATE"
    DEFER = "DEFER"
    DECLINE = "DECLINE"


@dataclass
class GovCase:
    """Minimal stand-in for mech_gov's BankingCase — only the fields the
    gates read. Built from an FTM scenario's ground_truth_spec (see
    ftm.expb.mapping)."""

    case_id: str
    risk_score: float
    completeness: float
    regulatory_flags: list[str] = field(default_factory=list)
    amount_usd: float = 0.0


@dataclass
class HardGate:
    """A single hard gate: condition → forced decision. Transcribed."""

    gate_id: str
    description: str
    condition: Callable[[GovCase], bool]
    forced_decision: Decision
    rationale_template: str


def build_default_gates(config: dict | None = None) -> list[HardGate]:
    """Build the default hard-gate list from config (paper defaults if None).

    Evaluated in order; first match wins. Transcribed verbatim from
    hard_gates.py::build_default_gates. K0_11 is intentionally absent from the
    pre-LLM gates (upstream Bug-2 fix: shadowed by K0_10); it is applied
    post-LLM via ambiguity_gate().
    """
    cfg = config or {}
    k06 = cfg.get("K0_6", {})
    k07 = cfg.get("K0_7", {})
    k08 = cfg.get("K0_8", {})
    k10 = cfg.get("K0_10", {})
    k12 = cfg.get("K0_12", {})
    k14 = cfg.get("K0_14", {})

    k06_risk = k06.get("risk_threshold", 0.9)
    k07_risk = k07.get("risk_threshold", 0.85)
    k07_min_flags = k07.get("min_flags", 3)
    k08_risk = k08.get("risk_threshold", 0.7)
    k08_amount = k08.get("amount_threshold_usd", 1_000_000)
    k10_comp = k10.get("completeness_threshold", 0.15)
    k12_risk = k12.get("risk_threshold", 0.7)
    k14_risk = k14.get("risk_threshold", 0.6)
    k14_min_severe = k14.get("min_severe_flags", 2)

    return [
        HardGate(
            "K0_6", "High risk + sanctions → DECLINE",
            lambda c: (c.risk_score > k06_risk and "SANCTIONS" in c.regulatory_flags),
            Decision.DECLINE, "Hard gate K0_6: high risk + SANCTIONS → DECLINE.",
        ),
        HardGate(
            "K0_7", "High risk + multiple flags → ESCALATE",
            lambda c: (c.risk_score > k07_risk and len(c.regulatory_flags) >= k07_min_flags),
            Decision.ESCALATE, "Hard gate K0_7: high risk + ≥flags → ESCALATE.",
        ),
        HardGate(
            "K0_8", "Large amount + high risk → ESCALATE",
            lambda c: (c.amount_usd > k08_amount and c.risk_score > k08_risk),
            Decision.ESCALATE, "Hard gate K0_8: large amount + high risk → ESCALATE.",
        ),
        HardGate(
            "K0_10", "Very low completeness → DEFER",
            lambda c: c.completeness < k10_comp,
            Decision.DEFER, "Hard gate K0_10: completeness below minimum → DEFER.",
        ),
        HardGate(
            "K0_12", "AML flag + high risk → DECLINE",
            lambda c: ("AML" in c.regulatory_flags and c.risk_score > k12_risk),
            Decision.DECLINE, "Hard gate K0_12: AML + high risk → DECLINE.",
        ),
        HardGate(
            "K0_13", "INSIDER flag → ESCALATE",
            lambda c: "INSIDER" in c.regulatory_flags,
            Decision.ESCALATE, "Hard gate K0_13: INSIDER flag → ESCALATE.",
        ),
        HardGate(
            "K0_14", "Multiple severe flags + elevated risk → DECLINE",
            lambda c: (
                len(set(c.regulatory_flags) & {"AML", "SANCTIONS", "INSIDER"}) >= k14_min_severe
                and c.risk_score > k14_risk
            ),
            Decision.DECLINE, "Hard gate K0_14: multiple severe flags + risk → DECLINE.",
        ),
    ]


def evaluate_hard_gates(
    case: GovCase,
    gates: list[HardGate] | None = None,
    config: dict | None = None,
) -> tuple[str, Decision, str] | None:
    """Evaluate all hard gates against a case. First match wins.

    Returns (gate_id, forced_decision, rationale) or None. Transcribed from
    hard_gates.py::evaluate_hard_gates (rationale simplified — the full
    upstream rationale templates are not needed for STAY/ACT scoring).
    """
    if gates is None:
        gates = build_default_gates(config)
    for gate in gates:
        if gate.condition(case):
            return gate.gate_id, gate.forced_decision, gate.rationale_template
    return None


def ambiguity_gate(
    case: GovCase,
    theta_iota: float = 0.3,
    risk_escalation_threshold: float = 0.7,
) -> Decision | None:
    """Force DEFER or ESCALATE when information is too incomplete.

    Transcribed verbatim from ambiguity_gate.py::ambiguity_gate:
      if completeness < θ_ι:
          if risk_score > 0.7 → ESCALATE (high risk + low info)
          else → DEFER (low risk + low info)

    In mech_gov this is a POST-LLM primitive (R2 pipeline step 5); Experiment
    B applies it only in ARM-2b, on the model's per-turn re-estimated
    completeness — never on static inputs in ARM-2a.
    """
    if case.completeness < theta_iota:
        if case.risk_score > risk_escalation_threshold:
            return Decision.ESCALATE
        return Decision.DEFER
    return None


# The gate ↔ FTM-anchor correspondence (documented for the mapping tests and
# the manifest). Hard gates are pre-LLM; the ambiguity gate is post-LLM.
GATE_ANCHOR_MAP = {
    "DET-01": {"gate": "K0_6", "kind": "hard", "threshold": 0.9},
    "DET-02": {"gate": "K0_12", "kind": "hard", "threshold": 0.7},
    "DET-03": {"gate": "K0_14", "kind": "hard", "threshold": 0.6},
    "DET-06": {"gate": "K0_7", "kind": "hard", "threshold": 0.85},
    "DET-07": {"gate": "K0_10", "kind": "hard", "threshold": 0.15},
    "DET-08": {"gate": "ambiguity", "kind": "post_llm", "threshold": 0.3},
    "DET-09": {"gate": "ambiguity", "kind": "post_llm", "threshold": 0.3},
}
