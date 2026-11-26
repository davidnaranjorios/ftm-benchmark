"""ftm — FTM evaluation engine (stdlib only).

Scenarios, pressure schedules, decision parsing, metrics and archetypes.
The Inspect task that drives real models lives in `ftm_inspect`.
"""

from .engine import (
    DOMAINS,
    PRESSURE_CHANNELS,
    PRESSURE_TEXTS,
    PRESSURE_SCHEDULES,
    DOMAIN_ROLES,
    STAY_EVENTS,
    ACT_EVENTS,
    TIER_META,
    DEFAULT_SYSTEM_PROMPT,
    Scenario,
    TurnResult,
    MetricsResult,
    ArchetypeResult,
    generate_scenarios,
    classify_reason,
    parse_decision,
    compute_metrics,
    detect_archetype,
    build_ftm_system_prompt,
    build_turn_user_message,
)

__all__ = [
    "DOMAINS",
    "PRESSURE_CHANNELS",
    "PRESSURE_TEXTS",
    "PRESSURE_SCHEDULES",
    "DOMAIN_ROLES",
    "STAY_EVENTS",
    "ACT_EVENTS",
    "TIER_META",
    "DEFAULT_SYSTEM_PROMPT",
    "Scenario",
    "TurnResult",
    "MetricsResult",
    "ArchetypeResult",
    "generate_scenarios",
    "classify_reason",
    "parse_decision",
    "compute_metrics",
    "detect_archetype",
    "build_ftm_system_prompt",
    "build_turn_user_message",
]
