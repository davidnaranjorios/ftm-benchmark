"""
HermesAdapter tests — fully offline via httpx.MockTransport reproducing the
REAL JSON shapes from the Hermes session-API audit:

  POST /api/sessions              → {"id": "..."}
  POST /api/sessions/{id}/chat    → synchronous {"message": {...}, "usage": {...}}
  GET  /api/sessions/{id}/messages→ history; assistant msgs with
                                    finish_reason=="tool_calls" carry
                                    tool_calls[].function.{name,arguments};
                                    tool msgs carry tool_name/tool_call_id
  GET  /v1/capabilities           → {"toolsets": [...]}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx", reason="httpx required (ftm-benchmark[a2a])")

from ftm.a2a import ToolClassifier
from ftm.bridges.hermes import HermesAdapter, fetch_hermes_profile, run_with_scenarios
from ftm.engine import TurnResult
from ftm.runner import RunConfig
from ftm.scenario_gen import AgentProfile, ScenarioGenerator


class FakeHermesServer:
    """In-process stock-Hermes stand-in with real response shapes.

    act_turns: user-turn numbers (per session) on which the agent calls the
    ACTION tool 'terminal'; other turns call the READ tool 'read_file'.
    """

    def __init__(self, act_turns: set[int] = frozenset()):
        self.act_turns = set(act_turns)
        self.sessions: dict[str, list[dict]] = {}   # id -> messages
        self.user_turns: dict[str, int] = {}
        self.chat_bodies: list[dict] = []           # every /chat body, for asserts
        self._next_id = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/api/sessions":
            self._next_id += 1
            sid = f"sess-{self._next_id}"
            self.sessions[sid] = []
            self.user_turns[sid] = 0
            return httpx.Response(200, json={"id": sid})

        if request.method == "GET" and path == "/v1/capabilities":
            return httpx.Response(200, json={"toolsets": [{
                "name": "core",
                "tools": [
                    {"name": "terminal", "description": "Run a shell command."},
                    {"name": "read_file", "description": "Read a file's contents."},
                ],
            }]})

        if path.startswith("/api/sessions/"):
            sid = path.split("/")[3]
            if sid not in self.sessions:
                return httpx.Response(404, json={"error": "unknown session"})
            msgs = self.sessions[sid]

            if request.method == "GET" and path.endswith("/messages"):
                return httpx.Response(200, json=msgs)

            if request.method == "POST" and path.endswith("/chat"):
                body = json.loads(request.content)
                self.chat_bodies.append({"session": sid, **body})
                self.user_turns[sid] += 1
                turn = self.user_turns[sid]
                acting = turn in self.act_turns
                tool = "terminal" if acting else "read_file"
                call_id = f"call_{sid}_{turn}"

                msgs.append({"role": "user", "content": body["message"]})
                msgs.append({
                    "role": "assistant", "content": None,
                    "finish_reason": "tool_calls",
                    "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(
                                {"command": "systemctl restart svc"} if acting
                                else {"path": "/var/log/status"}
                            ),
                        },
                    }],
                })
                msgs.append({
                    "role": "tool", "tool_name": tool,
                    "tool_call_id": call_id, "content": "ok",
                })
                final = {
                    "role": "assistant",
                    "content": (
                        "I executed the restart because conditions demanded action."
                        if acting else
                        "Data remains within thresholds; no intervention warranted."
                    ),
                    "finish_reason": "stop",
                }
                msgs.append(final)
                return httpx.Response(200, json={
                    "message": final,
                    "usage": {"input_tokens": 120, "output_tokens": 40},
                })

        return httpx.Response(404, json={"error": f"no route {path}"})


def _judge(name: str, description: str) -> str:
    return "ACTION" if name == "terminal" else "READ"


def _make(server: FakeHermesServer) -> HermesAdapter:
    return HermesAdapter(
        "http://hermes.test", api_key="test-key",
        classifier=ToolClassifier(judge=_judge),
        client=httpx.Client(transport=server.transport()),
    )


def _scenarios(n_turns_irrelevant=None):
    profile = AgentProfile(name="hermes", tools={
        "terminal": {"classification": "ACTION", "description": "Run a shell command."},
        "read_file": {"classification": "READ", "description": "Read a file."},
    })
    return ScenarioGenerator(profile, subject_model="declared-model", seed=1).generate("snapshot").scenarios


def _run_in(tmpdir: Path, config: RunConfig, adapter, scenarios) -> dict:
    old = os.getcwd()
    os.chdir(tmpdir)
    try:
        return run_with_scenarios(config, adapter, scenarios)
    finally:
        os.chdir(old)


def _checkpoint_turns(tmpdir: Path, run_id: str, sid: str) -> list[TurnResult]:
    path = tmpdir / "checkpoints" / f"{run_id}.jsonl"
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if not data.get("_failure") and data["scenario_id"] == sid:
                    out.append(TurnResult(**data))
    return out


def _cfg(run_id: str, max_turns: int) -> RunConfig:
    return RunConfig(models=["hermes"], tier="snapshot", run_id=run_id,
                     adapter="hermes", max_turns=max_turns)


# ── 1. e2e: ACTION tool at turn 3 → ACT exactly at turn 3 (watermark) ────────

def test_action_tool_call_lands_on_exact_turn(tmp_path):
    server = FakeHermesServer(act_turns={3})
    adapter = _make(server)
    scenarios = _scenarios()[:1]
    _run_in(tmp_path, _cfg("hermes_e2e", max_turns=4), adapter, scenarios)

    turns = sorted(_checkpoint_turns(tmp_path, "hermes_e2e", scenarios[0].scenario_id),
                   key=lambda t: t.turn)
    assert [t.turn for t in turns] == [1, 2, 3, 4]
    assert [t.decision for t in turns] == ["STAY", "STAY", "ACT", "STAY"]

    # The ACT turn's TurnObservation carried the terminal tool call
    # (persisted via raw_response only as text; assert via decision+reason)
    assert "restart" in turns[2].raw_response


# ── 2. Turn with no ACTION tool → STAY ────────────────────────────────────────

def test_read_only_turns_are_stay(tmp_path):
    server = FakeHermesServer(act_turns=set())
    adapter = _make(server)
    scenarios = _scenarios()[:1]
    _run_in(tmp_path, _cfg("hermes_stay", max_turns=3), adapter, scenarios)

    turns = _checkpoint_turns(tmp_path, "hermes_stay", scenarios[0].scenario_id)
    assert len(turns) == 3
    assert all(t.decision == "STAY" for t in turns)


# ── 3. Interruption mid-scenario → redo with NEW session, no duplication ─────

def test_interrupted_scenario_redone_with_new_session(tmp_path):
    server = FakeHermesServer(act_turns=set())
    adapter = _make(server)
    scenarios = _scenarios()[:1]
    sid = scenarios[0].scenario_id

    _run_in(tmp_path, _cfg("hermes_resume", max_turns=2), adapter, scenarios)
    assert len(server.sessions) == 1
    first_session = next(iter(server.sessions))
    assert server.user_turns[first_session] == 2

    _run_in(tmp_path, _cfg("hermes_resume", max_turns=3), adapter, scenarios)

    assert len(server.sessions) == 2, "redo must create a NEW Hermes session"
    second_session = [s for s in server.sessions if s != first_session][0]
    assert server.user_turns[second_session] == 3, (
        "redo must replay turns 1..3 in the new session, not continue at 3"
    )

    raw = _checkpoint_turns(tmp_path, "hermes_resume", sid)
    assert sorted(t.turn for t in raw) == [1, 1, 2, 2, 3]  # partial + redo
    deduped = {t.turn: t for t in raw}                     # runner's keep-last
    assert sorted(deduped) == [1, 2, 3]


# ── 4. System prompt prepended on turn 1 only, with agent framing ─────────────

def test_system_prompt_prepended_only_on_turn_1(tmp_path):
    server = FakeHermesServer()
    adapter = _make(server)
    scenarios = _scenarios()[:1]
    _run_in(tmp_path, _cfg("hermes_sys", max_turns=3), adapter, scenarios)

    bodies = [b["message"] for b in server.chat_bodies]
    assert len(bodies) == 3
    role_marker = scenarios[0].role[:40]
    assert role_marker in bodies[0], "turn 1 must carry the system prompt"
    assert role_marker not in bodies[1] and role_marker not in bodies[2]
    # And no history resent: each later body is just that turn's user message
    assert bodies[1].startswith("Turn 2") and bodies[2].startswith("Turn 3")


def test_agent_framing_replaces_format_block_in_hermes_system(tmp_path):
    """Agent mode must NOT instruct a text DECISION format (it suppresses
    tool calls); it must carry the action instruction instead."""
    from ftm.observation import AGENT_ACTION_INSTRUCTION

    server = FakeHermesServer()
    adapter = _make(server)
    scenarios = _scenarios()[:1]
    _run_in(tmp_path, _cfg("hermes_framing", max_turns=2), adapter, scenarios)

    turn1_body = server.chat_bodies[0]["message"]
    assert "DECISION:" not in turn1_body
    assert "CONFIDENCE:" not in turn1_body
    assert AGENT_ACTION_INSTRUCTION in turn1_body


# ── Profile from capabilities ─────────────────────────────────────────────────

def test_fetch_profile_from_capabilities():
    server = FakeHermesServer()
    profile = fetch_hermes_profile(
        "http://hermes.test", "k",
        classifier=ToolClassifier(judge=_judge),
        client=httpx.Client(transport=server.transport()),
    )
    assert profile.tools["terminal"]["classification"] == "ACTION"
    assert profile.tools["read_file"]["classification"] == "READ"
    assert profile.action_tools() == ["terminal"]
