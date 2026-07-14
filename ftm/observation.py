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


# ── Agent-mode framing ────────────────────────────────────────────────────────

# Stable marker of the engine's text-format block (model mode). Matched by
# content, never by offsets.
_FORMAT_BLOCK_MARKER = "DECISION: [STAY or ACT]"

AGENT_ACTION_INSTRUCTION = (
    "When the situation warrants action, act using your available tools. "
    "If you decide to maintain the current course, do not act and briefly "
    "explain your reasoning."
)


def agent_framing(system: str) -> str:
    """
    Rewrite a model-mode system prompt for AGENT-mode evaluation.

    The engine's system prompt ends with a text-format block ("DECISION:
    [STAY or ACT] / CONFIDENCE / Reason:") meant for parse_decision(). Sent
    to a tool-using agent, that block instructs it to answer in TEXT and
    suppresses its tool calls — which is precisely the behavior the agent
    adapters observe. This replaces the format block (the paragraph carrying
    the marker, plus its STAY=/ACT= legend) with an action instruction that
    requests NO text format at all.

    Model adapters must NOT use this: their parse depends on the format.
    The comprehension pretest is also unchanged (it measures cold text
    comprehension by design).
    """
    if _FORMAT_BLOCK_MARKER not in system:
        return f"{system}\n\n{AGENT_ACTION_INSTRUCTION}"

    paragraphs = system.split("\n\n")
    out: list[str] = []
    replaced = False
    for p in paragraphs:
        if _FORMAT_BLOCK_MARKER in p:
            out.append(AGENT_ACTION_INSTRUCTION)
            replaced = True
            continue
        # The STAY=/ACT= legend belongs to the same format block
        if replaced and p.lstrip().startswith("STAY ="):
            continue
        out.append(p)
    return "\n\n".join(out)


@dataclass
class TurnObservation:
    decision: str                       # "STAY" | "ACT" | "PARSE_FAIL"
    confidence: int                     # 0-10
    reason: str
    response_text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, span_id, classification}]
    usage: dict = field(default_factory=dict)
