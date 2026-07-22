"""
5-way decision resolver for Experiment B v2 (adapts v1's hybrid resolver).

The model's native output is the R1/R2 JSON. Resolution order:
  1. json        — parse the JSON object, read `decision`
  2. truncation  — regex on '"decision": "<WORD>"' (v1 lesson: verbose JSON
                   gets cut at max_tokens; `decision` is the first field)
  3. keyword     — first standalone occurrence of exactly one decision word
  4. unresolved  — PARSE_FAIL

Also extracts R2b's `completeness_estimate` field when present.
"""
from __future__ import annotations

import json
import re

from ftm.expb2 import DRIFT_RANK

DECISIONS = set(DRIFT_RANK)

_DEC_RE = re.compile(r'"decision"\s*:\s*"([A-Za-z]+)"')
_COMP_RE = re.compile(
    r'"completeness_estimate"\s*:\s*"?([01](?:\.\d+)?|\.\d+)"?'
)
_WORD_RES = {d: re.compile(rf"\b{d}\b") for d in DECISIONS}


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t


def resolve_decision(text: str) -> tuple[str, str]:
    """Returns (decision, source). decision ∈ 5-way ∪ {PARSE_FAIL}."""
    t = _strip_fences(text)
    candidates = [t]
    block = re.search(r"\{.*\}", t, re.DOTALL)
    if block:
        candidates.append(block.group())
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        d = str(obj.get("decision", "")).strip().upper()
        if d in DECISIONS:
            return d, "json"
    m = _DEC_RE.search(t)
    if m and m.group(1).upper() in DECISIONS:
        return m.group(1).upper(), "truncation_regex"
    found = [d for d, rx in _WORD_RES.items() if rx.search(t.upper())]
    if len(found) == 1:
        return found[0], "keyword"
    return "PARSE_FAIL", "unresolved"


def resolve_completeness(text: str) -> float | None:
    """R2b's per-turn completeness re-estimate, or None if absent/invalid."""
    m = _COMP_RE.search(_strip_fences(text))
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None
