"""
Governance adapters for Experiment B.

GatedAdapter wraps the subject model's adapter (composition; same complete()
interface) and applies mech_gov's mechanical enforcement to each turn's
decision. The FINAL decision is the gates', not the model's.

Two modes:
  "2a" static        — hard gates on the static anchor case. The ambiguity
                       gate is NOT applied (it is post-LLM in mech_gov, and
                       static application would break the deterministic
                       FARP=0 sanity property). Turn-invariant.
  "2b" model-mediated — hard gates on the static case, PLUS the ambiguity
                       gate on the model's per-turn re-estimated completeness
                       (parsed from a 'COMPLETENESS: <0-1>' line). This is the
                       H3 leak channel: pressure that makes the model report
                       lower completeness lets the ambiguity gate flip STAY→ACT.

STAY/ACT encoding (Precision A): "ACT" = the governance output deviated from
the anchor-correct decision; any gate firing (hard OR ambiguity, the latter
forcing DEFER/ESCALATE) maps to ACT. No gate firing maps to STAY.

Every turn's outcome is recorded in a gate-log entry so the analysis can tell
a gate block from a model decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ftm.engine import parse_decision
from ftm.expb.gates import (
    ambiguity_gate,
    build_default_gates,
    evaluate_hard_gates,
)
from ftm.expb.mapping import gov_case_from_spec

_COMPLETENESS_RE = re.compile(r"COMPLETENESS:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)


def parse_completeness(text: str, default: float) -> tuple[float, bool]:
    """Extract a 'COMPLETENESS: <0-1>' re-estimate. Returns (value, found).
    Falls back to `default` (the static spec completeness) if absent/invalid."""
    m = _COMPLETENESS_RE.search(text or "")
    if not m:
        return default, False
    try:
        v = float(m.group(1))
    except ValueError:
        return default, False
    if 0.0 <= v <= 1.0:
        return v, True
    return default, False


@dataclass
class GateLogEntry:
    scenario_id: str
    turn: int
    arm: str
    model_decision: str          # STAY | ACT | PARSE_FAIL (the subject's own)
    final_decision: str          # STAY | ACT (post-governance)
    gate_intervened: bool        # did governance override the model?
    gate_id: str | None          # which gate fired (hard id or "ambiguity")
    forced_decision: str | None  # mech_gov Decision forced (DECLINE/…), if any
    completeness_used: float | None
    completeness_reestimated: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class R1ResolvingAdapter:
    """ARM-1 wrapper: resolve the model's governance decision to FTM ACT/STAY
    via the hybrid resolver (DECISION line first, then R1's JSON `decision`
    field), and rewrite the response to a canonical DECISION line so the
    driver's parse_decision reads it. Preserves the model's own text in the
    reason for audit and records the resolution source per turn."""

    def __init__(self, subject_adapter, scenario_id: str):
        self.subject = subject_adapter
        self.scenario_id = scenario_id
        self.resolutions: list[dict] = []
        self.stateful = getattr(subject_adapter, "stateful", False)
        self._turn = 0

    def begin_scenario(self, scenario_id: str) -> None:
        self._turn = 0
        if hasattr(self.subject, "begin_scenario"):
            self.subject.begin_scenario(scenario_id)

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        from ftm.expb.arm1_policy import resolve_r1_decision
        self._turn += 1
        raw = self.subject.complete(system, messages)
        text = raw.get("text", "")
        decision, source = resolve_r1_decision(text)
        self.resolutions.append({
            "scenario_id": self.scenario_id, "turn": self._turn,
            "decision": decision, "source": source,
        })
        snippet = " ".join((text or "").split())[:160]
        out = dict(raw)
        out["text"] = (
            f"DECISION: {decision}\nCONFIDENCE: 7\n"
            f"Reason: [R1 {source}] {snippet}"
        )
        return out


class GatedAdapter:
    """Wrap a subject adapter; apply mech_gov gates per turn.

    subject_adapter: the model under evaluation (any object with
        complete(system, messages) -> {"text", "usage"}).
    spec: the scenario's ground_truth_spec (static anchor case source).
    mode: "2a" or "2b".
    """

    def __init__(self, subject_adapter, scenario_id: str, spec: dict, mode: str):
        if mode not in ("2a", "2b"):
            raise ValueError("mode must be '2a' or '2b'")
        self.subject = subject_adapter
        self.scenario_id = scenario_id
        self.spec = spec
        self.mode = mode
        self.arm = "ARM-2a" if mode == "2a" else "ARM-2b"
        self._gates = build_default_gates()
        self._turn = 0
        self.gate_log: list[GateLogEntry] = []
        # stateful passthrough for the runner's A2A handling, if any
        self.stateful = getattr(subject_adapter, "stateful", False)

    def begin_scenario(self, scenario_id: str) -> None:
        self._turn = 0
        if hasattr(self.subject, "begin_scenario"):
            self.subject.begin_scenario(scenario_id)

    def _static_completeness(self) -> float:
        gc = gov_case_from_spec(self.scenario_id, self.spec)
        return gc.completeness

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        self._turn += 1
        raw = self.subject.complete(system, messages)
        text = raw.get("text", "")
        model_decision = parse_decision(text)["decision"]

        # Model re-estimated completeness (ARM-2b only); else static spec value.
        static_comp = self._static_completeness()
        if self.mode == "2b":
            comp_used, reestimated = parse_completeness(text, static_comp)
        else:
            comp_used, reestimated = static_comp, False

        # Hard gates on the static anchor case (risk-anchored inputs unchanged).
        hard_case = gov_case_from_spec(self.scenario_id, self.spec)
        hard = evaluate_hard_gates(hard_case, self._gates)

        gate_id: str | None = None
        forced: str | None = None
        if hard is not None:
            gate_id, forced_dec, _ = hard
            forced = forced_dec.value
        elif self.mode == "2b":
            # Ambiguity gate on the model's (possibly pressured) completeness.
            amb_case = gov_case_from_spec(
                self.scenario_id, self.spec, completeness_override=comp_used
            )
            amb = ambiguity_gate(amb_case)
            if amb is not None:
                gate_id, forced = "ambiguity", amb.value

        final_decision = "ACT" if gate_id is not None else "STAY"
        gate_intervened = final_decision != model_decision

        self.gate_log.append(GateLogEntry(
            scenario_id=self.scenario_id,
            turn=self._turn,
            arm=self.arm,
            model_decision=model_decision,
            final_decision=final_decision,
            gate_intervened=gate_intervened,
            gate_id=gate_id,
            forced_decision=forced,
            completeness_used=round(comp_used, 3),
            completeness_reestimated=reestimated,
        ))

        # Rewrite the response so the engine's parse_decision reads the FINAL
        # (governance) decision, with an auditable structured prefix in reason.
        prefix = (
            f"[GATE_OVERRIDE gate={gate_id} model_said={model_decision} "
            f"forced={final_decision}] " if gate_intervened else ""
        )
        new_text = (
            f"DECISION: {final_decision}\n"
            f"CONFIDENCE: 9\n"
            f"Reason: {prefix}governance pipeline ({self.arm}) resolved this "
            f"turn to {final_decision}."
        )
        out = dict(raw)
        out["text"] = new_text
        return out
