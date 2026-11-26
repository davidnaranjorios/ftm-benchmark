"""
ftm — command-line entry point.

Thin wrapper over ftm.runner: parses arguments, resolves the adapter, and
delegates to runner.run(). No benchmark logic lives here.

    ftm run --adapter mock                      # offline demo, no API key
    ftm run --models gpt-4o --tier standard     # real model sweep

For published results use the Inspect task instead (see README):
    inspect eval ftm_inspect/ftm --model openrouter/openai/gpt-5.2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from ftm.adapters import get_adapter
from ftm.runner import RunConfig, run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftm", description="FTM Benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the FTM benchmark")
    p_run.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="Model identifiers (e.g. gpt-4o claude-sonnet-4-6). "
             "Defaults to 'mock' when --adapter mock.",
    )
    p_run.add_argument(
        "--tier", default="standard",
        choices=["snapshot", "standard", "extended", "research"],
    )
    p_run.add_argument(
        "--domain", default=None,
        choices=["devops_server", "medical", "financial", "legal", "industrial"],
    )
    p_run.add_argument("--run-id", default=None, dest="run_id")
    p_run.add_argument(
        "--adapter", default="auto",
        choices=["auto", "openai", "anthropic", "mock"],
    )

    p_tr = sub.add_parser("transcript", help="Readable transcript of a run")
    p_tr.add_argument("run_id", help="Run id (reads checkpoints/<run_id>.jsonl)")
    p_tr.add_argument("--full", action="store_true",
                      help="Print full model text instead of a 300-char excerpt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "transcript":
        return _transcript(args)

    models = args.models
    if models is None:
        if args.adapter == "mock":
            models = ["mock"]
        else:
            print("error: --models is required unless --adapter mock", file=sys.stderr)
            return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    base_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in models:
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

        # Compact per-model summary from the run-level aggregate (the
        # archetype is only meaningful there — per-scenario it degenerates)
        agg = report.get("aggregate") or {}
        agg_m = agg.get("metrics", {})
        agg_a = agg.get("archetype", {})
        print(json.dumps({
            "model": model,
            "run_id": model_run_id,
            "n_scenarios": report["n_scenarios"],
            "farp_rate": agg_m.get("farp_rate"),
            "composite": agg_m.get("composite"),
            "archetype": agg_a.get("name"),
            "risk": agg_a.get("risk"),
            "report": f"results/{model_run_id}_report.json",
        }, indent=2))

    return 0


def _transcript(args) -> int:
    """Readable per-scenario transcript from the checkpoint."""
    from pathlib import Path

    cp = Path("checkpoints") / f"{args.run_id}.jsonl"
    if not cp.exists():
        print(f"error: {cp} not found", file=sys.stderr)
        return 2

    turns_by_scenario: dict[str, list[dict]] = {}
    failures: list[dict] = []
    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("_failure"):
            failures.append(rec)
            continue
        turns_by_scenario.setdefault(rec["scenario_id"], []).append(rec)

    limit = None if args.full else 300
    for sid, turns in turns_by_scenario.items():
        # keep-last dedup, same rule as the runner
        deduped = {t["turn"]: t for t in sorted(turns, key=lambda t: t["turn"])}
        print(f"\n{'=' * 78}\nSCENARIO {sid}")
        first = deduped[min(deduped)]
        print(f"optimal={first['optimal']}  schedule={first['schedule_id']}")
        for t in sorted(deduped):
            rec = deduped[t]
            text = rec["raw_response"]
            mark = "OK " if rec["is_correct"] else "XX "
            body = text if limit is None else text[:limit] + ("…" if len(text) > limit else "")
            print(f"\n[{mark}] turn {t}  decision={rec['decision']}  "
                  f"channels={rec['n_active_channels']}")
            print(f"      {body}")
    if failures:
        print(f"\n{'=' * 78}\nFAILURE RECORDS ({len(failures)}):")
        for f in failures:
            print(f"  {f['scenario_id']} turn {f['turn']}: {f['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
