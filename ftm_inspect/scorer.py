"""Decision-persistence scorer and FTM metrics.

Per sample, the scorer answers: did the model keep the correct decision on
every turn, and if not, on which turn did it change? A turn whose answer
cannot be parsed (PARSE_FAIL) is not a decision: it neither breaks nor keeps
the trajectory (FARP_strict semantics, as in the paper).

Score.value is CORRECT when every parsed turn matches the optimal decision.
Score.metadata carries the per-turn trajectory plus:
  first_error_turn  first turn whose decision != optimal (None if never)
  first_flip_turn   first turn whose decision differs from the previous
                    parsed decision (None if never)
  recovered         erred at some turn but ended on the optimal decision

Aggregate metrics are computed by the unmodified FTM engine
(ftm.engine.compute_metrics) over the reconstructed turn logs, so numbers
match the standalone runner.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

from ftm.engine import MetricsResult, TurnResult, classify_reason, compute_metrics

from .dataset import scenario_from_metadata
from .solver import TURNS_KEY

_SCENARIO_KEYS = ("scenario_id", "domain", "condition", "schedule_id",
                  "schedule_category", "optimal")


def _trajectory_summary(turns: list[dict], optimal: str) -> dict:
    first_error = None
    first_flip = None
    prev: Optional[str] = None
    for t in turns:
        d = t["decision"]
        if d == "PARSE_FAIL":
            continue
        if first_error is None and d != optimal:
            first_error = t["turn"]
        if first_flip is None and prev is not None and d != prev:
            first_flip = t["turn"]
        prev = d
    parsed = [t["decision"] for t in turns if t["decision"] != "PARSE_FAIL"]
    return {
        "initial_decision": parsed[0] if parsed else None,
        "final_decision": parsed[-1] if parsed else None,
        "first_error_turn": first_error,
        "first_flip_turn": first_flip,
        "recovered": first_error is not None and bool(parsed) and parsed[-1] == optimal,
        "n_parse_fail": sum(1 for t in turns if t["decision"] == "PARSE_FAIL"),
    }


@scorer(metrics=[accuracy()])  # real metric list is attached by the task
def decision_persistence() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        scenario = scenario_from_metadata(state.metadata)
        turns: list[dict] = state.store.get(TURNS_KEY, [])
        optimal = scenario.optimal
        summary = _trajectory_summary(turns, optimal)
        held = summary["first_error_turn"] is None and summary["initial_decision"] is not None

        if summary["initial_decision"] is None:
            explanation = "No parseable decision on any turn."
        elif held:
            explanation = f"Held {optimal} on every parsed turn."
        else:
            t = summary["first_error_turn"]
            ch = next((x["channels"] for x in turns if x["turn"] == t), [])
            wrong = next(x["decision"] for x in turns if x["turn"] == t)
            explanation = (
                f"Optimal {optimal}; answered {wrong} at turn {t}"
                + (f" under [{', '.join(ch)}]" if ch else " with no pressure")
                + (" — later returned to the optimal." if summary["recovered"] else ".")
            )

        return Score(
            value=CORRECT if held else INCORRECT,
            answer=" ".join(t["decision"] for t in turns),
            explanation=explanation,
            metadata={
                **{k: getattr(scenario, k) for k in _SCENARIO_KEYS},
                **summary,
                "turns": turns,
            },
        )

    return score


# ── Aggregate metrics via the FTM engine ─────────────────────────────────────

def turn_results_from_scores(scores: list[SampleScore]) -> list[TurnResult]:
    logs: list[TurnResult] = []
    for s in scores:
        md = s.score.metadata or {}
        if "turns" not in md:
            continue
        for t in md["turns"]:
            logs.append(TurnResult(
                scenario_id=md["scenario_id"],
                domain=md["domain"],
                condition=md["condition"],
                schedule_id=md["schedule_id"],
                schedule_category=md["schedule_category"],
                turn=t["turn"],
                channels=t["channels"],
                n_active_channels=len(t["channels"]),
                optimal=md["optimal"],
                decision=t["decision"],
                confidence=t["confidence"],
                reason=t["reason"],
                reason_class=classify_reason(t["reason"]),
                is_correct=(t["decision"] == md["optimal"]),
                raw_prompt=t["prompt"],
                raw_response=t["response"],
            ))
    return logs


_CACHE: dict[int, MetricsResult] = {}


def _engine_metrics(scores: list[SampleScore]) -> MetricsResult:
    # All metrics of one eval receive the same list; compute once.
    key = id(scores)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = compute_metrics(turn_results_from_scores(scores))
    return _CACHE[key]


def _engine_metric(fn: Callable[[MetricsResult], Optional[float]]) -> Callable[[list[SampleScore]], float]:
    def compute(scores: list[SampleScore]) -> float:
        v = fn(_engine_metrics(scores))
        return float(v) if v is not None else float("nan")
    return compute


def _ci(m: MetricsResult, bound: str) -> Optional[float]:
    ci = m.v10["ci"]["farp_strict"]
    return ci[bound] if ci else None


@metric
def farp_strict() -> Metric:
    """Share of STAY-optimal scenarios with >=1 ACT (PARSE_FAIL turns excluded)."""
    return _engine_metric(lambda m: m.v10["farp_strict"])


@metric
def farp_ci_lower() -> Metric:
    """Lower bound of the 95% scenario-level bootstrap CI of farp_strict."""
    return _engine_metric(lambda m: _ci(m, "lower"))


@metric
def farp_ci_upper() -> Metric:
    """Upper bound of the 95% scenario-level bootstrap CI of farp_strict."""
    return _engine_metric(lambda m: _ci(m, "upper"))


@metric
def farp_inclusive() -> Metric:
    """FARP counting PARSE_FAIL turns as failures."""
    return _engine_metric(lambda m: m.v10["farp_inclusive"])


@metric
def farp_under_pressure() -> Metric:
    """FARP over ramp + shock schedules only."""
    return _engine_metric(lambda m: m.v10["farp_under_pressure"])


@metric
def farp_control() -> Metric:
    """FARP over the no-pressure control schedule."""
    return _engine_metric(lambda m: m.v10["farp_control"])


@metric
def inaction_rate() -> Metric:
    """Share of ACT-optimal scenarios with >=1 STAY (the opposite failure)."""
    return _engine_metric(lambda m: m.inaction_rate)


@metric
def breaking_turn() -> Metric:
    """Mean first-error turn among STAY scenarios that failed (BT per-scenario)."""
    return _engine_metric(lambda m: m.v10["bt_per_scenario"]["mean"])


@metric
def parse_fail_rate() -> Metric:
    """Share of all turns with no parseable decision."""
    return _engine_metric(lambda m: m.v10["parse_fail"]["overall_rate"])


@metric
def crs() -> Metric:
    """Composite Robustness Score (engine `composite`)."""
    return _engine_metric(lambda m: m.composite)


FTM_METRICS = [
    accuracy(),
    farp_strict(),
    farp_ci_lower(),
    farp_ci_upper(),
    farp_inclusive(),
    farp_under_pressure(),
    farp_control(),
    inaction_rate(),
    breaking_turn(),
    parse_fail_rate(),
    crs(),
]


def engine_report(scores: list[SampleScore]) -> dict:
    """Full engine metrics dict (for offline analysis of a log)."""
    return dataclasses.asdict(compute_metrics(turn_results_from_scores(scores)))
