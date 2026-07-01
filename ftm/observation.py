"""
Normalized per-turn observation returned by every adapter.

The decision is derived INSIDE the adapter (parse_decision for text models,
OTel tool-call spans for A2A agents). The runner consumes the observation
without knowing which strategy produced it.

decision semantics:
    "STAY"       — adapter affirmatively observed a hold decision
    "ACT"        — adapter affirmatively observed an action
    "PARSE_FAIL" — no decision could be derived. For A2A agents this covers
                   the UNKNOWN case (zero telemetry spans within the polling
                   window); the reason field carries the "UNKNOWN: ..." marker.
                   Never mapped to STAY by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnObservation:
    decision: str                       # "STAY" | "ACT" | "PARSE_FAIL"
    confidence: int                     # 0-10
    reason: str
    response_text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, span_id, classification}]
    usage: dict = field(default_factory=dict)
