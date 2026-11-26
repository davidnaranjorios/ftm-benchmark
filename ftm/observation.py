"""
Normalized per-turn observation returned by every adapter.

The decision is derived INSIDE the adapter (parse_decision on the model's
text). The runner consumes the observation without knowing how it was made.

decision semantics:
    "STAY"       — the model answered with a hold decision
    "ACT"        — the model answered with an action decision
    "PARSE_FAIL" — the model responded but no decision could be parsed
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnObservation:
    decision: str                       # "STAY" | "ACT" | "PARSE_FAIL"
    confidence: int                     # 0-10
    reason: str
    response_text: str
    usage: dict = field(default_factory=dict)
