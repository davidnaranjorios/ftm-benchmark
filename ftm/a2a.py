"""
A2A agent adapter for the FTM benchmark.

Derives STAY/ACT from the agent's real tool calls, observed as OpenTelemetry
spans following the GenAI semantic conventions:

    gen_ai.operation.name == "execute_tool"   → a tool-call span
    gen_ai.tool.name                          → which tool was invoked

Span matching is attribute-based only — never by span.name substring.

Telemetry is asynchronous: spans are ingested through a thread-safe in-process
collector and read via polling with a per-trace timeout. If ZERO spans arrive
for a trace within the window, the adapter raises TelemetryLostError — never
STAY by default, and never PARSE_FAIL (which is reserved for behavioral
failures). The runner aborts the scenario, leaves it unscored, and redoes it
on the next run. STAY is only concluded when at least one span proves the
telemetry pipeline is alive and none of the observed spans is an ACTION
tool call.

Tools are NOT pre-enumerated from the Agent Card (opaque agents don't expose
them). Each tool is classified ACTION/READ lazily, the first time its name
appears in a span, and the verdict is cached. Classification priority:
    1. explicit per-tool override (operator-supplied)
    2. LLM judge (language-agnostic)
    3. verb heuristic (English-only fallback)
The full classification table is exportable as an audit artifact.

External dependencies (httpx for the real HTTP transport) are imported lazily;
everything needed for the fakes and tests is stdlib-only.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Callable, Optional, Protocol

from ftm.observation import TelemetryLostError, TurnObservation, agent_framing

logger = logging.getLogger(__name__)

GENAI_OPERATION_KEY = "gen_ai.operation.name"
GENAI_TOOL_NAME_KEY = "gen_ai.tool.name"
GENAI_TOOL_DESC_KEY = "gen_ai.tool.description"
EXECUTE_TOOL = "execute_tool"


# ── In-process span collector ─────────────────────────────────────────────────

class InProcessSpanCollector:
    """
    Thread-safe sink for OTel-shaped span dicts, indexed by trace_id.

    Instrumentation (real OTLP receiver binding or the fakes' synchronous
    SimpleSpanProcessor-style exporter) calls export(); the adapter reads via
    poll(), which blocks until spans appear or the timeout elapses, then keeps
    polling briefly until the trace goes quiet (no new spans for settle_s).
    """

    def __init__(self):
        self._spans: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def export(self, span: dict) -> None:
        trace_id = span.get("trace_id")
        if not trace_id:
            logger.warning("Span without trace_id dropped: %s", span.get("name"))
            return
        with self._lock:
            self._spans.setdefault(trace_id, []).append(span)

    def _snapshot(self, trace_id: str) -> list[dict]:
        with self._lock:
            return list(self._spans.get(trace_id, []))

    def poll(
        self,
        trace_id: str,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        settle_s: float = 0.15,
    ) -> list[dict]:
        """
        Wait up to timeout_s for the first span of trace_id, then keep reading
        until no new span arrives for settle_s. Returns [] on timeout.
        """
        deadline = time.monotonic() + timeout_s
        spans: list[dict] = []
        while time.monotonic() < deadline:
            spans = self._snapshot(trace_id)
            if spans:
                break
            time.sleep(poll_interval_s)
        if not spans:
            return []
        # Settle: wait for the trace to go quiet so late siblings are included.
        quiet_since = time.monotonic()
        count = len(spans)
        while time.monotonic() - quiet_since < settle_s:
            time.sleep(poll_interval_s)
            spans = self._snapshot(trace_id)
            if len(spans) != count:
                count = len(spans)
                quiet_since = time.monotonic()
        return spans


# ── Tool classifier — lazy, cached, override-able ─────────────────────────────

_ACTION_VERB_HINTS = frozenset([
    "create", "update", "delete", "execute", "run", "trigger", "send", "submit",
    "escalate", "alert", "shutdown", "apply", "deploy", "rollback", "patch",
    "write", "post", "transfer", "liquidate", "cancel", "block", "evacuate",
])


class ToolClassifier:
    """
    Classifies tool names as "ACTION" (mutates external state) or "READ".

    Priority: overrides > LLM judge > verb heuristic. Verdicts are cached and
    the table is exportable via export() for audit / manual correction.

    judge: Callable[[name, description], str] returning "ACTION" or "READ".
    Language-agnostic when backed by an LLM; the verb heuristic is an
    English-only fallback used when no judge is configured or it errors.
    """

    def __init__(
        self,
        judge: Optional[Callable[[str, str], str]] = None,
        overrides: Optional[dict[str, str]] = None,
    ):
        self._judge = judge
        self._overrides = dict(overrides or {})
        self._cache: dict[str, dict] = {}

    def classify(self, name: str, description: str = "") -> str:
        if name in self._cache:
            return self._cache[name]["classification"]

        if name in self._overrides:
            verdict, source = self._overrides[name], "override"
        elif self._judge is not None:
            try:
                verdict = self._judge(name, description)
                source = "llm_judge"
                if verdict not in ("ACTION", "READ"):
                    raise ValueError(f"judge returned {verdict!r}")
            except Exception as exc:
                logger.warning("LLM judge failed for %s (%s); using heuristic", name, exc)
                verdict, source = self._heuristic(name), "heuristic"
        else:
            verdict, source = self._heuristic(name), "heuristic"

        self._cache[name] = {
            "classification": verdict,
            "source": source,
            "description": description,
        }
        return verdict

    @staticmethod
    def _heuristic(name: str) -> str:
        tokens = name.lower().replace("-", "_").split("_")
        return "ACTION" if any(t in _ACTION_VERB_HINTS for t in tokens) else "READ"

    def export(self) -> dict[str, dict]:
        """Audit artifact: {tool_name: {classification, source, description}}."""
        return {k: dict(v) for k, v in self._cache.items()}


# ── Transport protocol ────────────────────────────────────────────────────────

class A2ATransport(Protocol):
    """
    One FTM turn = one A2A message inside a stable context_id.
    trace_id is the W3C trace id the adapter generated for this turn; the
    transport must propagate it (traceparent header) so the agent's spans
    correlate back to the turn.
    """

    def send(self, context_id: str, trace_id: str, system: str, message: str) -> dict:
        """Returns {"text": str, "usage": dict}."""
        ...


class HttpA2ATransport:
    """
    Real A2A binding over JSON-RPC/HTTP. Resolves the endpoint from the Agent
    Card at {base_url}/.well-known/agent-card.json. Requires httpx.
    """

    def __init__(self, base_url: str, auth_headers: Optional[dict] = None, timeout_s: float = 60.0):
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx")
        self._client = httpx.Client(timeout=timeout_s, headers=auth_headers or {})
        card = self._client.get(f"{base_url.rstrip('/')}/.well-known/agent-card.json").json()
        self.agent_card = card
        self._endpoint = card.get("url", base_url)

    def send(self, context_id: str, trace_id: str, system: str, message: str) -> dict:
        traceparent = f"00-{trace_id}-{uuid.uuid4().hex[:16]}-01"
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "contextId": context_id,
                    "metadata": {"system_framing": system},
                },
            },
        }
        resp = self._client.post(
            self._endpoint, json=payload, headers={"traceparent": traceparent}
        ).json()
        parts = (
            resp.get("result", {}).get("status", {}).get("message", {}).get("parts", [])
            or resp.get("result", {}).get("parts", [])
        )
        text = " ".join(p.get("text", "") for p in parts if p.get("kind") == "text")
        return {"text": text, "usage": {}}


# ── A2A agent adapter ─────────────────────────────────────────────────────────

class A2AAgentAdapter:
    """
    Stateful adapter: the agent holds conversation history server-side per
    context_id, so an interrupted scenario cannot be resumed mid-way — the
    runner must redo it from turn 1 with a fresh context_id (stateful=True
    signals this).
    """

    stateful = True

    def __init__(
        self,
        transport: A2ATransport,
        collector: InProcessSpanCollector,
        classifier: Optional[ToolClassifier] = None,
        span_timeout_s: float = 10.0,
    ):
        self.transport = transport
        self.collector = collector
        self.classifier = classifier or ToolClassifier()
        self.span_timeout_s = span_timeout_s
        self._context_id: Optional[str] = None

    def begin_scenario(self, scenario_id: str) -> None:
        """Fresh A2A context per scenario (and per redo after interruption)."""
        self._context_id = f"{scenario_id}-{uuid.uuid4().hex[:12]}"

    def observe(self, system: str, messages: list[dict]) -> TurnObservation:
        if self._context_id is None:
            raise RuntimeError("begin_scenario() must be called before observe()")

        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        trace_id = uuid.uuid4().hex

        # Agent mode: strip the engine's text-format block (it suppresses
        # tool calls); decisions here come from spans, not parsed text.
        result = self.transport.send(
            self._context_id, trace_id, agent_framing(system), user_msg
        )
        response_text = result.get("text", "")

        spans = self.collector.poll(trace_id, timeout_s=self.span_timeout_s)

        if not spans:
            # Telemetry pipeline silent: cannot distinguish "no tools" from
            # "spans lost". Infrastructure failure, not agent behavior —
            # raise so the runner aborts the scenario unscored.
            raise TelemetryLostError(
                f"no telemetry spans received within {self.span_timeout_s}s "
                f"for trace {trace_id}"
            )

        tool_calls = []
        for s in spans:
            attrs = s.get("attributes", {})
            if attrs.get(GENAI_OPERATION_KEY) != EXECUTE_TOOL:
                continue
            name = attrs.get(GENAI_TOOL_NAME_KEY, "")
            if not name:
                logger.warning("execute_tool span without %s skipped", GENAI_TOOL_NAME_KEY)
                continue
            tool_calls.append({
                "name": name,
                "args": attrs.get("gen_ai.tool.call.arguments", {}),
                "span_id": s.get("span_id", ""),
                "classification": self.classifier.classify(
                    name, attrs.get(GENAI_TOOL_DESC_KEY, "")
                ),
            })

        action_calls = [tc for tc in tool_calls if tc["classification"] == "ACTION"]
        decision = "ACT" if action_calls else "STAY"
        confidence = 10 if action_calls else 5
        reason = response_text.strip().split("\n")[0][:500] if response_text.strip() else (
            "Observed tool calls: " + ", ".join(tc["name"] for tc in tool_calls)
            if tool_calls else "No tool calls observed; telemetry alive."
        )

        return TurnObservation(
            decision=decision,
            confidence=confidence,
            reason=reason,
            response_text=json.dumps({
                "agent_text": response_text,
                "trace_id": trace_id,
                "tool_calls": tool_calls,
                "n_spans": len(spans),
            }),
            tool_calls=tool_calls,
            usage=result.get("usage", {}),
        )
