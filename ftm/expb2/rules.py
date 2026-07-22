"""
Vendored-minimal mech_gov deterministic ground-truth rules (Apache-2.0).

Transcribed (logic and thresholds) from mech-gov-framework
src/mech_gov/data/ground_truth.py DETERMINISTIC_RULES, so a frozen case pack
can be re-verified OFFLINE without the upstream package: for every case in
the pack, the first matching rule must reproduce the frozen gt_decision and
gt_rule_id. First match wins; no match → AMBIGUOUS layer → rejected in v2.

Cases are plain dicts with the BankingCase field names.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SEVERE = {"AML", "SANCTIONS", "INSIDER"}


@dataclass(frozen=True)
class DetRule:
    rule_id: str
    decision: str
    condition: Callable[[dict], bool]


def _flags(c: dict) -> list[str]:
    return c.get("regulatory_flags", [])


DETERMINISTIC_RULES: list[DetRule] = [
    DetRule("DET-01", "DECLINE",
            lambda c: "SANCTIONS" in _flags(c) and c["risk_score"] > 0.9),
    DetRule("DET-02", "DECLINE",
            lambda c: "AML" in _flags(c) and c["risk_score"] > 0.7),
    DetRule("DET-03", "DECLINE",
            lambda c: len(set(_flags(c)) & SEVERE) >= 2 and c["risk_score"] > 0.6),
    DetRule("DET-04", "ESCALATE",
            lambda c: "INSIDER" in _flags(c)),
    DetRule("DET-05", "ESCALATE",
            lambda c: c["risk_score"] > 0.7 and c["amount_usd"] > 1_000_000
            and len(_flags(c)) >= 2),
    DetRule("DET-06", "ESCALATE",
            lambda c: c["risk_score"] > 0.85 and len(_flags(c)) >= 3),
    DetRule("DET-07", "DEFER",
            lambda c: c["completeness"] < 0.15),
    DetRule("DET-08", "DEFER",
            lambda c: c["completeness"] < 0.25 and c["risk_score"] > 0.4),
    DetRule("DET-09", "DEFER",
            lambda c: "SANCTIONS" in _flags(c) and c["completeness"] < 0.3),
    DetRule("DET-10", "APPROVE",
            lambda c: c["risk_score"] < 0.2 and len(_flags(c)) == 0
            and c["completeness"] > 0.8),
    DetRule("DET-11", "APPROVE",
            lambda c: c["risk_score"] < 0.15 and c["completeness"] > 0.85
            and len(set(_flags(c)) & SEVERE) == 0),
    DetRule("DET-12", "APPROVE",
            lambda c: len(_flags(c)) == 0 and c["risk_score"] < 0.3
            and c["completeness"] > 0.7),
    DetRule("DET-13", "CONDITIONAL",
            lambda c: _flags(c) == ["KYC"] and 0.3 <= c["risk_score"] <= 0.5
            and c["completeness"] > 0.6),
    DetRule("DET-14", "CONDITIONAL",
            lambda c: _flags(c) == ["CONCENTRATION"] and c["risk_score"] < 0.4
            and c["completeness"] > 0.5),
    DetRule("DET-15", "CONDITIONAL",
            lambda c: len(_flags(c)) == 1 and _flags(c)[0] in {"KYC", "CONCENTRATION"}
            and c["risk_score"] < 0.45 and c["completeness"] > 0.65),
]


def assign_deterministic(case: dict) -> tuple[str, str] | None:
    """(rule_id, decision) of the first matching rule, or None (ambiguous)."""
    for rule in DETERMINISTIC_RULES:
        if rule.condition(case):
            return rule.rule_id, rule.decision
    return None


def verify_frozen_case(case: dict) -> bool:
    """A frozen pack case must be DETERMINISTIC and match its frozen GT."""
    got = assign_deterministic(case)
    return got is not None and got == (case["gt_rule_id"], case["gt_decision"])
