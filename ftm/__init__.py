"""ftm — FTM Evaluation Engine (neutral baseline build).

Stdlib-only engine for the FTM benchmark. Runs on the neutral
DEFAULT_SYSTEM_PROMPT only; no intervention ("cura") blocks are included.
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
