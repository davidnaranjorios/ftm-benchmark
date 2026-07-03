"""
ftm — command-line entry point.

Thin wrapper over ftm.runner: parses arguments, resolves the adapter, and
delegates to runner.run(). No benchmark logic lives here.

    ftm run --adapter mock                      # offline demo, no API key
    ftm run --models gpt-4o --tier standard     # real model sweep
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

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

        # Compact per-model summary: headline metrics + archetype spread
        archetypes: dict[str, int] = {}
        farps, composites = [], []
        for sc in report["scenarios"]:
            archetypes[sc["archetype"]["name"]] = archetypes.get(sc["archetype"]["name"], 0) + 1
            farps.append(sc["metrics"]["farp_rate"])
            composites.append(sc["metrics"]["composite"])
        n = max(len(report["scenarios"]), 1)
        print(json.dumps({
            "model": model,
            "run_id": model_run_id,
            "n_scenarios": report["n_scenarios"],
            "mean_farp_rate": round(sum(farps) / n, 3) if farps else None,
            "mean_composite": round(sum(composites) / n, 3) if composites else None,
            "archetypes": archetypes,
            "report": f"results/{model_run_id}_report.json",
        }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
