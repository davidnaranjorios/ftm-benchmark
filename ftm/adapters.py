"""
Model adapters for the FTM benchmark runner.

External dependencies (openai, anthropic) are imported lazily inside each
adapter — the core runner works with MockAdapter using only stdlib.

Interface:
    complete(system, messages) -> {"text": str, "usage": dict}
"""
from __future__ import annotations

import abc
import os
import time
from typing import Any

from ftm.engine import parse_decision
from ftm.observation import TurnObservation


# ── Abstract interface ────────────────────────────────────────────────────────

class ModelAdapter(abc.ABC):
    # Stateless adapters resend the full message history each turn, so the
    # runner can resume an interrupted scenario mid-way. Stateful adapters
    # (e.g. A2A agents holding history server-side) set this True and the
    # runner redoes the scenario from turn 1 instead.
    stateful: bool = False

    @abc.abstractmethod
    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        """
        Call the model and return {"text": str, "usage": {"input_tokens": int, "output_tokens": int}}.
        Raises on unrecoverable failure; retries are the adapter's responsibility.
        """

    def begin_scenario(self, scenario_id: str) -> None:
        """Per-scenario setup hook (fresh context for stateful adapters)."""

    def observe(self, system: str, messages: list[dict]) -> TurnObservation:
        """
        Run one turn and return a normalized observation with the decision
        already derived. Text-model default: complete() + parse_decision().
        """
        result = self.complete(system, messages)
        text = result.get("text", "")
        d = parse_decision(text)
        return TurnObservation(
            decision=d["decision"],
            confidence=d["confidence"],
            reason=d["reason"],
            response_text=text,
            usage=result.get("usage", {}),
        )


# ── Retry helper ──────────────────────────────────────────────────────────────

def _retry(fn, retryable_types: tuple, max_attempts: int = 5, base_delay: float = 1.0):
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable_types as exc:
            last_exc = exc
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIAdapter(ModelAdapter):
    """OpenAI client — also serves any OpenAI-compatible endpoint (xAI,
    OpenRouter, ...) via base_url + api_key."""

    def __init__(self, model: str = "gpt-4o", base_url: str | None = None,
                 api_key: str | None = None):
        try:
            import openai as _oai
        except ImportError:
            raise ImportError("pip install openai")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self._client = _oai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        import openai

        msgs = [{"role": "system", "content": system}] + messages

        def _call():
            return self._client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=256
            )

        resp = _retry(
            _call,
            (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        )
        return {
            "text": resp.choices[0].message.content or "",
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic as _ant
        except ImportError:
            raise ImportError("pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        self._client = _ant.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        import anthropic

        def _call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system,
                messages=messages,
            )

        resp = _retry(
            _call,
            (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError),
        )
        return {
            "text": resp.content[0].text if resp.content else "",
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }


# ── Mock (stdlib-only, deterministic, zero API cost) ─────────────────────────

# Keywords that appear exclusively in ACT events (not in any STAY event).
_ACT_KEYWORDS = frozenset([
    "CRITICAL",
    "EMERGENCY",
    "STOP-LOSS",
    "MARGIN CALL",
    "SAFETY ALERT",
    "EVACUATION",
    "IMMEDIATE INTERVENTION",
])


class MockAdapter(ModelAdapter):
    """
    Deterministic adapter for offline testing.

    Decision rule: inspect the FIRST user message in the conversation
    (turn 1 — always contains the scenario event verbatim) for keywords
    that appear only in ACT events. This makes the decision consistent
    across all T turns of the same scenario and independent of pressure text.
    """

    def complete(self, system: str, messages: list[dict]) -> dict[str, Any]:
        first_user = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        upper = first_user.upper()
        decision = "ACT" if any(kw in upper for kw in _ACT_KEYWORDS) else "STAY"
        return {
            "text": (
                f"DECISION: {decision}\n"
                "CONFIDENCE: 9\n"
                "Reason: Objective data confirms current status."
            ),
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def get_adapter(
    model: str,
    adapter: str = "auto",
    base_url: str | None = None,
    api_key: str | None = None,
) -> ModelAdapter:
    """
    Resolve a ModelAdapter from a model name and an optional adapter hint.

    adapter values: "auto" | "openai" | "anthropic" | "mock"
    base_url: any OpenAI-compatible endpoint (xAI, OpenRouter, ...). With
    "auto", grok / "x-ai/..." models resolve to xAI (XAI_API_KEY) or, failing
    that, OpenRouter (OPENROUTER_API_KEY).
    """
    if adapter == "mock":
        return MockAdapter()
    if base_url:
        return OpenAIAdapter(model=model, base_url=base_url, api_key=api_key)
    if adapter == "openai" or (
        adapter == "auto"
        and (model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"))
    ):
        return OpenAIAdapter(model=model)
    if adapter == "anthropic" or (adapter == "auto" and model.startswith("claude-")):
        return AnthropicAdapter(model=model)
    if adapter == "auto" and (model.startswith("x-ai/") or model.startswith("grok")):
        xai_key = api_key or os.environ.get("XAI_API_KEY")
        if xai_key:
            return OpenAIAdapter(
                model=model.split("/", 1)[-1],           # xAI ids have no vendor prefix
                base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
                api_key=xai_key,
            )
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            return OpenAIAdapter(
                model=model, base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
        raise EnvironmentError(
            f"model {model!r} needs XAI_API_KEY or OPENROUTER_API_KEY "
            "(or an explicit base_url/api_key)"
        )
    raise ValueError(
        f"Cannot infer adapter for model {model!r}. "
        "Pass --adapter openai|anthropic|mock explicitly."
    )
