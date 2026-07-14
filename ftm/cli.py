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
    # Hermes session-API evaluation (agent under test = a stock Hermes)
    p_run.add_argument("--hermes-url", default=None,
                       help="Base URL of a stock Hermes (e.g. http://localhost:8642)")
    p_run.add_argument("--hermes-key", default=None,
                       help="Bearer token; defaults to $API_SERVER_KEY")
    p_run.add_argument("--subject-model", default=None, dest="subject_model",
                       help="Declared model the Hermes agent runs on (required with --hermes-url)")
    p_run.add_argument("--tools-file", default=None, dest="tools_file",
                       help="JSON list declaring the tool surface when /v1/capabilities lacks granularity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.hermes_url:
        return _run_hermes(args)

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


def _run_hermes(args) -> int:
    """Evaluate a stock Hermes via its session API (lazy imports: httpx path)."""
    import os
    from pathlib import Path

    from ftm.a2a import ToolClassifier
    from ftm.bridges.hermes import (
        GRANULARITY_LIMITATION,
        HERMES_OBSERVATION_MODE,
        HermesAdapter,
        fetch_hermes_profile,
        run_with_scenarios,
    )
    from ftm.scenario_gen import ScenarioGenerator

    if not args.subject_model:
        print("error: --subject-model is required with --hermes-url (declared)", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    api_key = args.hermes_key or os.environ.get("API_SERVER_KEY", "")
    classifier = ToolClassifier()

    profile = fetch_hermes_profile(
        args.hermes_url, api_key, classifier=classifier, tools_file=args.tools_file,
    )

    # Same-model rule: the pretest uses the declared subject model if its API
    # key is available; otherwise it is skipped and recorded in the manifest.
    try:
        subject_adapter = get_adapter(args.subject_model, "auto")
    except Exception:
        subject_adapter = None

    gen = ScenarioGenerator(
        profile,
        subject_model=args.subject_model,
        subject_model_source="declared",
        subject_adapter=subject_adapter,
    )
    result = gen.generate(tier=args.tier)

    manifest = result.manifest
    manifest["observation_mode"] = HERMES_OBSERVATION_MODE
    manifest.setdefault("known_limitations", []).append(GRANULARITY_LIMITATION)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_hermes"
    Path("results").mkdir(exist_ok=True)
    manifest_path = Path("results") / f"{run_id}_scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    adapter = HermesAdapter(args.hermes_url, api_key, classifier=classifier)
    config = RunConfig(models=[args.subject_model], tier=args.tier,
                       run_id=run_id, adapter="hermes")
    report = run_with_scenarios(config, adapter, result.scenarios)

    print(json.dumps({
        "run_id": run_id,
        "observation_mode": HERMES_OBSERVATION_MODE,
        "n_scenarios": report["n_scenarios"],
        "comprehension_discard_rate":
            manifest["comprehension_pretest"]["comprehension_discard_rate"],
        "manifest": str(manifest_path),
        "report": f"results/{run_id}_report.json",
        "tool_classifications": classifier.export(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
