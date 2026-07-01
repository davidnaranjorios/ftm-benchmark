"""
Normalized per-turn observation returned by every adapter.

The decision is derived INSIDE the adapter (parse_decision for text models,
OTel tool-call spans for A2A agents). The runner consumes the observation
without knowing which strategy produced it.

decision semantics:
    "STAY"       — adapter affirmatively observed a hold decision
    "ACT"        — adapter affirmatively observed an action
    "PARSE_FAIL" — the model/agent RESPONDED but the decision could not be
                   derived from its behavior (unparseable text). Reserved for
                   behavioral failures only.

Telemetry loss is NOT a decision: when an agent adapter cannot observe the
turn at all (e.g. zero OTel spans within the polling window), it raises
TelemetryLostError instead of returning an observation. The runner treats it
as a permanent turn failure: no TurnResult, the scenario is left incomplete
and unscored, and it is redone on the next run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class TelemetryLostError(Exception):
    """The turn ran but its behavior could not be observed (infra failure,
    not agent behavior). Never mapped to STAY or PARSE_FAIL."""


@dataclass
class TurnObservation:
    decision: str                       # "STAY" | "ACT" | "PARSE_FAIL"
    confidence: int                     # 0-10
    reason: str
    response_text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, span_id, classification}]
    usage: dict = field(default_factory=dict)
