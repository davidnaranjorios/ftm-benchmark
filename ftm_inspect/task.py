"""The FTM task for Inspect.

    inspect eval ftm_inspect/ftm --model openrouter/openai/gpt-5.2
    inspect eval ftm_inspect/ftm --model openrouter/openai/gpt-5.2 -T data_anchoring=true
"""
from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from ftm.engine import T_MAX

from .dataset import ftm_dataset
from .scorer import FTM_METRICS, decision_persistence
from .solver import pressure_dialogue


@task
def ftm(
    tier: str = "extended",
    data_anchoring: bool = False,
    domain: Optional[str] = None,
    condition: Optional[str] = None,
    schedule: Optional[str] = None,
    max_turns: int = T_MAX,
    operator_prompt: Optional[str] = None,
) -> Task:
    """FTM: does the model keep a correct decision under social pressure?

    Args:
      tier: engine scenario set — snapshot (5), standard (30), extended (90),
        research (300 — reuses the same 30 events with rotated pressure text).
      data_anchoring: add the Data Anchoring rule to the system prompt.
      domain: restrict to one of devops_server, medical, financial, legal, industrial.
      condition: restrict to "stay" or "act" scenarios.
      schedule: restrict to control_flat, ramp_emot_lead or shock_immediate.
      max_turns: turns per scenario (default 10).
      operator_prompt: extra operator instruction appended to the system prompt.
    """
    return Task(
        dataset=ftm_dataset(tier, domain, condition, schedule),
        solver=pressure_dialogue(
            data_anchoring=data_anchoring,
            max_turns=max_turns,
            operator_prompt=operator_prompt,
        ),
        scorer=decision_persistence(),
        metrics=FTM_METRICS,
        config=GenerateConfig(temperature=0.0),
        metadata={"data_anchoring": data_anchoring, "tier": tier},
    )
