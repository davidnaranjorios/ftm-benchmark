"""ftm_inspect — the FTM benchmark as an Inspect AI task.

    inspect eval ftm_inspect/ftm --model openrouter/<provider>/<model>
"""
from .dataset import ftm_dataset, scenario_to_sample
from .scorer import FTM_METRICS, decision_persistence, engine_report
from .solver import DATA_ANCHORING_PROMPT, pressure_dialogue
from .task import ftm

__all__ = [
    "DATA_ANCHORING_PROMPT",
    "FTM_METRICS",
    "decision_persistence",
    "engine_report",
    "ftm",
    "ftm_dataset",
    "pressure_dialogue",
    "scenario_to_sample",
]
