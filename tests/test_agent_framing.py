"""Mode-dependent framing: agent adapters strip the engine's text-format
block (it suppresses tool calls); model adapters keep it (their parse
depends on it). Stdlib-only."""
from __future__ import annotations

from ftm.adapters import MockAdapter
from ftm.engine import build_ftm_system_prompt, generate_scenarios
from ftm.observation import AGENT_ACTION_INSTRUCTION, agent_framing


def _engine_system() -> str:
    sc = generate_scenarios("snapshot", "devops_server")[0]
    return build_ftm_system_prompt(sc)


def test_agent_framing_replaces_format_block():
    system = _engine_system()
    assert "DECISION: [STAY or ACT]" in system  # sanity: engine emits it

    framed = agent_framing(system)
    assert "DECISION:" not in framed
    assert "CONFIDENCE:" not in framed
    assert "STAY =" not in framed              # legend goes with the block
    assert AGENT_ACTION_INSTRUCTION in framed
    assert framed.startswith(system.split("\n\n")[0])  # role preserved


def test_agent_framing_appends_when_no_marker():
    framed = agent_framing("You are a plain agent.")
    assert framed.startswith("You are a plain agent.")
    assert AGENT_ACTION_INSTRUCTION in framed


def test_model_mode_keeps_format_suffix_intact():
    """Model adapters receive the system untouched: observe() passes it to
    complete() verbatim, and the DECISION block must survive — parse_decision
    depends on it."""
    system = _engine_system()

    captured = {}

    class Spy(MockAdapter):
        def complete(self, sys_prompt, messages):
            captured["system"] = sys_prompt
            return super().complete(sys_prompt, messages)

    Spy().observe(system, [{"role": "user", "content": "Situation: all nominal."}])
    assert captured["system"] == system
    assert "DECISION: [STAY or ACT]" in captured["system"]
