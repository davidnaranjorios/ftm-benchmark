"""FTM scenarios as an Inspect dataset.

Each Sample is one engine Scenario. The turn-1 situation is the sample input,
the optimal decision ("STAY" | "ACT") is the target, and the full Scenario
(pressure texts per turn, channels, schedule) travels in the metadata so the
solver can rebuild every turn exactly as the engine defines it.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from inspect_ai.dataset import MemoryDataset, Sample

from ftm.engine import (
    PRESSURE_SCHEDULES,
    Scenario,
    build_turn_user_message,
    generate_scenarios,
)

SCHEDULE_IDS = [s["id"] for s in PRESSURE_SCHEDULES]


def scenario_to_sample(scenario: Scenario) -> Sample:
    return Sample(
        id=scenario.scenario_id,
        input=build_turn_user_message(scenario, 1),
        target=scenario.optimal,
        metadata={"scenario": dataclasses.asdict(scenario)},
    )


def scenario_from_metadata(metadata: dict) -> Scenario:
    return Scenario(**metadata["scenario"])


def ftm_dataset(
    tier: str = "extended",
    domain: Optional[str] = None,
    condition: Optional[str] = None,
    schedule: Optional[str] = None,
) -> MemoryDataset:
    """Engine scenarios for `tier`, optionally filtered.

    condition: "stay" | "act"; schedule: one of SCHEDULE_IDS.
    """
    if condition not in (None, "stay", "act"):
        raise ValueError("condition must be 'stay', 'act' or None")
    if schedule is not None and schedule not in SCHEDULE_IDS:
        raise ValueError(f"schedule must be one of {SCHEDULE_IDS}")
    scenarios = [
        s for s in generate_scenarios(tier, domain)
        if (condition is None or s.condition == condition)
        and (schedule is None or s.schedule_id == schedule)
    ]
    return MemoryDataset(
        [scenario_to_sample(s) for s in scenarios],
        name=f"ftm_{tier}",
    )
