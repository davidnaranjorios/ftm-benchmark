"""
HermesAdapter — evaluate a stock Hermes agent through its session API.

Pure client of the stock Hermes HTTP surface (audited):
  POST /api/sessions                  → create session, returns {id}
  POST /api/sessions/{id}/chat        → SYNCHRONOUS; blocks, returns {message, usage}
  GET  /api/sessions/{id}/messages    → full history; assistant messages with
                                        finish_reason == "tool_calls" carry
                                        tool_calls[].function.{name, arguments}
  GET  /v1/capabilities               → toolsets (agent profile source)
Auth: Authorization: Bearer $API_SERVER_KEY.

System prompt handling: the session API takes no separate system field, so the
scenario's system prompt is PREPENDED to the first user message (turn 1 only).
Subsequent turns send ONLY that turn's user message — the session is stateful
server-side; history is never resent.

Turn attribution (the watermark): before sending, the adapter records the
current length of GET /messages; after the synchronous chat returns, the
messages added past that watermark are the turn's messages, and their
tool_calls are the turn's tool calls. decision = ACT iff any of them
classifies as ACTION (ToolClassifier: lazy, cached, override-able).

stateful = True: an interrupted scenario is redone from turn 1 with a NEW
session (begin_scenario creates one); the runner's keep-last checkpoint dedup
already handles superseding partial turns.

Known limitation (recorded in the manifest by the CLI): classification is at
TOOL granularity. A generic tool like "terminal" is one ACTION tool — an
`ls` and an `rm -rf` inside it are indistinguishable at this level.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ftm.a2a import ToolClassifier
from ftm.adapters import ModelAdapter, _retry
from ftm.observation import TurnObservation, agent_framing
from ftm.scenario_gen import AgentProfile

logger = logging.getLogger(__name__)

HERMES_OBSERVATION_MODE = "hermes-session-api"

GRANULARITY_LIMITATION = (
    "Tool-call observation via the Hermes session API classifies at TOOL "
    "granularity: a generic tool (e.g. 'terminal') is a single ACTION tool, "
    "so a harmless command and a destructive one inside it are "
    "indistinguishable to the benchmark."
)


class _RetryableHermesError(Exception):
    """5xx or transport error — retried with the shared backoff helper."""


class HermesAdapter(ModelAdapter):
    stateful = True

    def __init__(
        self,
        base_url: str = "http://localhost:8642",
        api_key: str = "",
        classifier: Optional[ToolClassifier] = None,
        client=None,               # injectable httpx.Client (tests use MockTransport)
        timeout_s: float = 120.0,
    ):
        if client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("pip install 'ftm-benchmark[a2a]' (httpx required)")
            client = httpx.Client(timeout=timeout_s)
        self._client = client
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.classifier = classifier or ToolClassifier()
        self._session_id: Optional[str] = None
        self._turn = 0

    # ── HTTP with retry on transport errors / 5xx ─────────────────────────────

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> Any:
        import httpx

        def _call():
            try:
                resp = self._client.request(
                    method, f"{self.base_url}{path}",
                    json=json_body, headers=self._headers,
                )
            except httpx.TransportError as exc:
                raise _RetryableHermesError(str(exc)) from exc
            if resp.status_code >= 500:
                raise _RetryableHermesError(f"{resp.status_code} on {path}")
            resp.raise_for_status()
            return resp.json()

        return _retry(_call, (_RetryableHermesError,))

    # ── ModelAdapter surface ──────────────────────────────────────────────────

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        raise NotImplementedError(
            "HermesAdapter derives decisions from session tool calls; use observe()"
        )

    def begin_scenario(self, scenario_id: str) -> None:
        resp = self._request("POST", "/api/sessions", {})
        self._session_id = str(resp["id"])
        self._turn = 0
        logger.info("Hermes session %s for scenario %s", self._session_id, scenario_id)

    def observe(self, system: str, messages: list[dict]) -> TurnObservation:
        if self._session_id is None:
            raise RuntimeError("begin_scenario() must be called before observe()")
        self._turn += 1

        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        # Turn 1 only: the session API has no separate system field. Agent
        # mode: the engine's text-format block would suppress tool calls, so
        # it is replaced by the action instruction (agent_framing).
        text = f"{agent_framing(system)}\n\n{user_msg}" if self._turn == 1 else user_msg

        sid = self._session_id
        watermark = len(self._request("GET", f"/api/sessions/{sid}/messages"))

        chat = self._request("POST", f"/api/sessions/{sid}/chat", {"message": text})

        history = self._request("GET", f"/api/sessions/{sid}/messages")
        turn_msgs = history[watermark:]

        tool_calls: list[dict] = []
        response_text = ""
        for m in turn_msgs:
            if m.get("role") != "assistant":
                continue
            if m.get("finish_reason") == "tool_calls":
                for tc in m.get("tool_calls", []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if not name:
                        continue
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            pass
                    tool_calls.append({
                        "name": name,
                        "args": args,
                        "classification": self.classifier.classify(name),
                    })
            elif m.get("content"):
                response_text = m["content"]  # final assistant of the turn

        if not response_text:
            response_text = (chat.get("message") or {}).get("content", "") \
                if isinstance(chat.get("message"), dict) else str(chat.get("message", ""))

        action_calls = [tc for tc in tool_calls if tc["classification"] == "ACTION"]
        decision = "ACT" if action_calls else "STAY"

        return TurnObservation(
            decision=decision,
            confidence=10 if action_calls else 5,
            reason=response_text.strip().split("\n")[0][:500],
            response_text=response_text,
            tool_calls=tool_calls,
            usage=chat.get("usage", {}) or {},
        )


# Re-export: the scenario-injection helper now lives with the runner.
from ftm.runner import run_with_scenarios  # noqa: F401  (backward compat)


# ── Agent profile from Hermes capabilities ────────────────────────────────────

def fetch_hermes_profile(
    base_url: str,
    api_key: str = "",
    classifier: Optional[ToolClassifier] = None,
    tools_file: Optional[str] = None,
    client=None,
    agent_name: str = "hermes",
) -> AgentProfile:
    """
    Build an AgentProfile for scenario generation.

    Primary source: GET /v1/capabilities (toolsets). If it yields nothing
    usable — or the operator wants explicit control — a tools_file (JSON list
    of names, or of {name, description} objects) declares the surface.
    """
    classifier = classifier or ToolClassifier()
    tools: dict[str, dict] = {}

    if tools_file:
        entries = json.loads(open(tools_file).read())
        for e in entries:
            name = e if isinstance(e, str) else e["name"]
            desc = "" if isinstance(e, str) else e.get("description", "")
            tools[name] = {
                "classification": classifier.classify(name, desc),
                "description": desc,
            }
    else:
        adapter = HermesAdapter(base_url, api_key, classifier=classifier, client=client)
        try:
            caps = adapter._request("GET", "/v1/capabilities")
        except Exception as exc:
            raise RuntimeError(
                f"GET /v1/capabilities failed ({exc}); pass --tools-file to declare "
                "the tool surface explicitly"
            ) from exc
        tools = _parse_toolsets(caps.get("toolsets"), classifier)

        # Fallback: some Hermes builds (e.g. Nous Research's Hermes Agent)
        # expose tools at /v1/toolsets instead of inside /v1/capabilities.
        if not tools:
            try:
                raw = adapter._request("GET", "/v1/toolsets")
            except Exception as exc:
                raise RuntimeError(
                    "/v1/capabilities returned no tool granularity and "
                    f"GET /v1/toolsets failed ({exc}); pass --tools-file"
                ) from exc
            # Shapes seen in the wild: Nous Hermes Agent wraps the list as
            # {"object": "list", "data": [...]}; others use {"toolsets": [...]}
            # or a bare list.
            if isinstance(raw, dict):
                toolsets = raw.get("toolsets") or raw.get("data")
            else:
                toolsets = raw
            tools = _parse_toolsets(toolsets, classifier)

        if not tools:
            raise RuntimeError(
                "neither /v1/capabilities nor /v1/toolsets returned tool "
                "granularity; pass --tools-file"
            )

    return AgentProfile(name=agent_name, tools=tools)


def _parse_toolsets(toolsets, classifier: ToolClassifier) -> dict[str, dict]:
    """Normalize a toolsets payload into {name: {classification, description}}.

    Tolerated shapes: a list of toolset objects with "tools" (each tool a
    string or a {name, description} object), or a flat list of tools.
    Toolsets carrying "enabled": false are skipped — their tools cannot be
    invoked, so they must not ground scenarios. When a tool is a bare string,
    it is classified by NAME ONLY: the toolset's group description (e.g.
    "read, write, patch, search") would contaminate every sibling — a
    read_file misclassified ACTION turns ordinary reads into false FARP.
    The group description is kept as metadata.
    """
    tools: dict[str, dict] = {}
    for ts in toolsets or []:
        if isinstance(ts, dict) and ts.get("enabled") is False:
            continue
        ts_desc = ts.get("description", "") if isinstance(ts, dict) else ""
        entries = ts.get("tools", []) if isinstance(ts, dict) and "tools" in ts else [ts]
        for t in entries or []:
            if isinstance(t, str):
                name, desc, classify_desc = t, ts_desc, ""
            else:
                name = t.get("name", "")
                desc = t.get("description", "")
                classify_desc = desc  # per-tool description: safe context
            if not name:
                continue
            tools[name] = {
                "classification": classifier.classify(name, classify_desc),
                "description": desc,
            }
    return tools
