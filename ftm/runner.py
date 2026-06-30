"""
FTM benchmark runner with checkpoint-based resumption.

Checkpoint format  : checkpoints/<run_id>.jsonl  — one TurnResult per line, append-only
Results format     : results/<run_id>.jsonl       — one record per completed scenario
Final report       : results/<run_id>_report.json + results/<run_id>_report.csv

Resumption contract
-------------------
On startup the runner loads the existing checkpoint (if any) and reconstructs
{(scenario_id, turn): TurnResult} for every already-completed unit.  When
entering a partially-done scenario it replays raw_prompt / raw_response from
the checkpoint to rebuild the multi-turn message history before continuing
with the next turn. Already-done (scenario_id, turn) pairs are never repeated.

Isolation contract
------------------
A turn that fails after all retries is logged and skipped; the run continues
with the next unit. Failed turns remain absent from the checkpoint and will be
retried on the next invocation (same run_id).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ftm.engine import (
    T_MAX,
    Scenario,
    TurnResult,
    build_ftm_system_prompt,
    build_turn_user_message,
    classify_reason,
    compute_metrics,
    detect_archetype,
    generate_scenarios,
    parse_decision,
)
from ftm.adapters import ModelAdapter, get_adapter

logger = logging.getLogger(__name__)


# ── Run configuration ─────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    models: list[str]
    tier: str = "standard"
    domain: Optional[str] = None
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    adapter: str = "auto"
    max_turns: int = T_MAX  # override for testing (default = T_MAX = 10)


# ── Path helpers ──────────────────────────────────────────────────────────────

def _checkpoint_path(run_id: str) -> Path:
    p = Path("checkpoints")
    p.mkdir(exist_ok=True)
    return p / f"{run_id}.jsonl"


def _results_path(run_id: str) -> Path:
    p = Path("results")
    p.mkdir(exist_ok=True)
    return p / f"{run_id}.jsonl"


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def load_checkpoint(run_id: str) -> list[TurnResult]:
    path = _checkpoint_path(run_id)
    if not path.exists():
        return []
    results: list[TurnResult] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(TurnResult(**json.loads(line)))
            except Exception as exc:
                logger.warning("Checkpoint %s line %d skipped (%s)", run_id, lineno, exc)
    return results


def _append_checkpoint(run_id: str, tr: TurnResult) -> None:
    with _checkpoint_path(run_id).open("a") as f:
        f.write(json.dumps(dataclasses.asdict(tr)) + "\n")


def _load_completed_scenario_ids(run_id: str) -> set[str]:
    path = _results_path(run_id)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["scenario_id"])
                except Exception:
                    pass
    return ids


def _append_scenario_result(run_id: str, record: dict) -> None:
    with _results_path(run_id).open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── Core runner ───────────────────────────────────────────────────────────────

def run(config: RunConfig, adapter: ModelAdapter) -> dict:
    """
    Execute the FTM benchmark for a single (model, config) pair.
    Resumes transparently if checkpoints/<run_id>.jsonl already exists.
    Returns the aggregated report dict.
    """
    # Load checkpoint index: {(scenario_id, turn): TurnResult}
    checkpoint_records = load_checkpoint(config.run_id)
    done: dict[tuple[str, int], TurnResult] = {
        (tr.scenario_id, tr.turn): tr for tr in checkpoint_records
    }
    logger.info(
        "[%s] Loaded %d checkpoint records", config.run_id, len(done)
    )

    # Scenarios already written to results (avoid duplication on resume)
    completed_scenario_ids = _load_completed_scenario_ids(config.run_id)

    scenarios = generate_scenarios(config.tier, config.domain)
    logger.info(
        "[%s] %d scenarios to run (tier=%s domain=%s)",
        config.run_id, len(scenarios), config.tier, config.domain,
    )

    all_results: list[dict] = []

    for scenario in scenarios:
        # Map of already-done turns for this scenario
        scenario_done: dict[int, TurnResult] = {
            t: done[(scenario.scenario_id, t)]
            for t in range(1, config.max_turns + 1)
            if (scenario.scenario_id, t) in done
        }

        if len(scenario_done) == config.max_turns:
            # All turns complete — reconstruct for in-memory aggregation
            turn_results = [scenario_done[t] for t in sorted(scenario_done)]
            record = _build_scenario_record(scenario, turn_results)
            if scenario.scenario_id not in completed_scenario_ids:
                _append_scenario_result(config.run_id, record)
                completed_scenario_ids.add(scenario.scenario_id)
            all_results.append(record)
            logger.debug("[%s] %s — skipped (complete)", config.run_id, scenario.scenario_id)
            continue

        # ── Rebuild message history from checkpoint turns ─────────────────────
        system = build_ftm_system_prompt(scenario)
        messages: list[dict] = []
        turn_results: list[TurnResult] = []

        for t in sorted(scenario_done):
            tr = scenario_done[t]
            messages.append({"role": "user", "content": tr.raw_prompt})
            messages.append({"role": "assistant", "content": tr.raw_response})
            turn_results.append(tr)

        if scenario_done:
            logger.info(
                "[%s] %s — resuming at turn %d",
                config.run_id, scenario.scenario_id, max(scenario_done) + 1,
            )

        # ── Run remaining turns ───────────────────────────────────────────────
        for t in range(1, config.max_turns + 1):
            if t in scenario_done:
                continue

            user_msg = build_turn_user_message(scenario, t)
            messages.append({"role": "user", "content": user_msg})

            try:
                t0 = time.time()
                result = adapter.complete(system, messages)
                latency_ms = int((time.time() - t0) * 1000)
                raw_response = result["text"]
            except Exception as exc:
                logger.error(
                    "[%s] %s turn %d failed: %s — skipping",
                    config.run_id, scenario.scenario_id, t, exc,
                )
                messages.pop()  # remove the user message we just appended
                continue

            messages.append({"role": "assistant", "content": raw_response})

            d = parse_decision(raw_response)
            channels = (
                scenario.active_channels_by_turn[t - 1]
                if t - 1 < len(scenario.active_channels_by_turn)
                else []
            )
            tr = TurnResult(
                scenario_id=scenario.scenario_id,
                domain=scenario.domain,
                condition=scenario.condition,
                schedule_id=scenario.schedule_id,
                schedule_category=scenario.schedule_category,
                turn=t,
                channels=channels,
                n_active_channels=len(channels),
                optimal=scenario.optimal,
                decision=d["decision"],
                confidence=d["confidence"],
                reason=d["reason"],
                reason_class=classify_reason(d["reason"]),
                is_correct=(d["decision"] == scenario.optimal),
                raw_prompt=user_msg,
                raw_response=raw_response,
                latency_ms=latency_ms,
            )

            _append_checkpoint(config.run_id, tr)
            turn_results.append(tr)

        # ── Close scenario: metrics + archetype ───────────────────────────────
        if turn_results:
            record = _build_scenario_record(scenario, turn_results)
            if scenario.scenario_id not in completed_scenario_ids:
                _append_scenario_result(config.run_id, record)
                completed_scenario_ids.add(scenario.scenario_id)
            all_results.append(record)
            logger.info(
                "[%s] %s — done (%d turns, archetype=%s)",
                config.run_id,
                scenario.scenario_id,
                len(turn_results),
                record["archetype"]["name"],
            )

    return _write_final_report(config, all_results)


def _build_scenario_record(scenario: Scenario, turn_results: list[TurnResult]) -> dict:
    metrics = compute_metrics(turn_results)
    archetype = detect_archetype(metrics)
    return {
        "scenario_id": scenario.scenario_id,
        "domain": scenario.domain,
        "condition": scenario.condition,
        "schedule_id": scenario.schedule_id,
        "schedule_category": scenario.schedule_category,
        "optimal": scenario.optimal,
        "n_turns": len(turn_results),
        "metrics": dataclasses.asdict(metrics),
        "archetype": dataclasses.asdict(archetype),
    }


def _write_final_report(config: RunConfig, all_results: list[dict]) -> dict:
    report = {
        "run_id": config.run_id,
        "models": config.models,
        "tier": config.tier,
        "domain": config.domain,
        "n_scenarios": len(all_results),
        "scenarios": all_results,
    }

    Path("results").mkdir(exist_ok=True)
    report_path = Path("results") / f"{config.run_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", report_path)

    if all_results:
        csv_path = Path("results") / f"{config.run_id}_report.csv"
        flat: list[dict] = []
        for r in all_results:
            m = r["metrics"]
            a = r["archetype"]
            flat.append({
                "scenario_id": r["scenario_id"],
                "domain": r["domain"],
                "condition": r["condition"],
                "schedule_id": r["schedule_id"],
                "optimal": r["optimal"],
                "n_turns": r["n_turns"],
                "stay_acc": m.get("stay_acc"),
                "act_acc": m.get("act_acc"),
                "dis": m.get("dis"),
                "abi": m.get("abi"),
                "composite": m.get("composite"),
                "farp_rate": m.get("farp_rate"),
                "inaction_rate": m.get("inaction_rate"),
                "archetype": a.get("name"),
                "risk": a.get("risk"),
            })
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
        logger.info("CSV written to %s", csv_path)

    return report


# ── CLI entry point ───────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ftm.runner",
        description="FTM Benchmark Runner",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        metavar="MODEL",
        help="Model identifiers (e.g. gpt-4o claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--tier", default="standard",
        choices=["snapshot", "standard", "extended", "research"],
    )
    parser.add_argument(
        "--domain", default=None,
        choices=["devops_server", "medical", "financial", "legal", "industrial"],
    )
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument(
        "--adapter", default="auto",
        choices=["auto", "openai", "anthropic", "mock"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    base_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in args.models:
        model_run_id = f"{base_run_id}_{model.replace('/', '_')}"
        config = RunConfig(
            models=[model],
            tier=args.tier,
            domain=args.domain,
            run_id=model_run_id,
            adapter=args.adapter,
        )
        adapter_obj = get_adapter(model, args.adapter)
        report = run(config, adapter_obj)
        print(json.dumps({"model": model, "run_id": model_run_id, "n_scenarios": report["n_scenarios"]}, indent=2))


if __name__ == "__main__":
    _main()
