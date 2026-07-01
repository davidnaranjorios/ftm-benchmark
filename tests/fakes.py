"""
In-process fakes for the A2A + OTel slice. No network, no external infra,
no tokens — same philosophy as MockAdapter.

FakeA2AAgent implements the A2ATransport protocol. On each send() it emits
OTel-shaped spans straight into the collector (synchronous export, i.e.
SimpleSpanProcessor semantics), or delayed / dropped when configured, so the
adapter's polling and UNKNOWN paths are exercisable deterministically.
"""
from __future__ import annotations

import threading
import time
import uuid

from ftm.a2a import (
    EXECUTE_TOOL,
    GENAI_OPERATION_KEY,
    GENAI_TOOL_DESC_KEY,
    GENAI_TOOL_NAME_KEY,
    InProcessSpanCollector,
)


class FakeA2AAgent:
    """
    act_turns   : FTM turn numbers (1-based, per context) on which the agent
                  "acts" by calling an ACTION tool. On other turns it calls a
                  READ tool (telemetry stays alive → STAY is provable).
    span_delay_s: export spans from a background thread after this delay
                  (late-span scenario; 0 = synchronous export).
    drop_spans  : never export spans (UNKNOWN scenario).
    """

    def __init__(
        self,
        collector: InProcessSpanCollector,
        act_turns: set[int] = frozenset(),
        action_tool: str = "transfer_funds",
        read_tool: str = "check_portfolio_status",
        span_delay_s: float = 0.0,
        drop_spans: bool = False,
    ):
        self.collector = collector
        self.act_turns = set(act_turns)
        self.action_tool = action_tool
        self.read_tool = read_tool
        self.span_delay_s = span_delay_s
        self.drop_spans = drop_spans
        # Observability for tests:
        self.turn_counter: dict[str, int] = {}      # context_id -> turns seen
        self.messages_seen: list[tuple[str, str]] = []  # (context_id, message)

    @property
    def context_ids_seen(self) -> list[str]:
        return list(self.turn_counter)

    def send(self, context_id: str, trace_id: str, system: str, message: str) -> dict:
        self.turn_counter[context_id] = self.turn_counter.get(context_id, 0) + 1
        turn = self.turn_counter[context_id]
        self.messages_seen.append((context_id, message))

        acting = turn in self.act_turns
        tool = self.action_tool if acting else self.read_tool
        spans = [
            {
                "trace_id": trace_id,
                "span_id": uuid.uuid4().hex[:16],
                "name": "invoke_agent fake-agent",
                "attributes": {GENAI_OPERATION_KEY: "invoke_agent"},
            },
            {
                "trace_id": trace_id,
                "span_id": uuid.uuid4().hex[:16],
                # Deliberately unhelpful span.name: matching must go through
                # GenAI attributes, never span.name substrings.
                "name": "opaque-internal-op-7f3a",
                "attributes": {
                    GENAI_OPERATION_KEY: EXECUTE_TOOL,
                    GENAI_TOOL_NAME_KEY: tool,
                    GENAI_TOOL_DESC_KEY: (
                        "Transfiere fondos entre cuentas de inmediato."
                        if acting else
                        "Consulta el estado actual del portafolio sin modificarlo."
                    ),
                },
            },
        ]

        if not self.drop_spans:
            if self.span_delay_s > 0:
                def _late_export():
                    time.sleep(self.span_delay_s)
                    for s in spans:
                        self.collector.export(s)
                threading.Thread(target=_late_export, daemon=True).start()
            else:
                for s in spans:  # synchronous export (SimpleSpanProcessor)
                    self.collector.export(s)

        text = (
            f"Executed {tool} due to observed conditions."
            if acting else
            "Current data does not warrant intervention; maintaining state."
        )
        return {"text": text, "usage": {"input_tokens": 0, "output_tokens": 0}}
