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
    # Generation filters (agent evaluation)
    p_run.add_argument("--schedule", default=None,
                       choices=["control_flat", "ramp_emot_lead", "shock_immediate"],
                       help="Restrict generated scenarios to one pressure schedule")
    p_run.add_argument("--condition", default=None, choices=["stay", "act"],
                       help="Restrict generated scenarios to one condition")
    p_run.add_argument("--max-scenarios", type=int, default=None, dest="max_scenarios",
                       help="Override the tier's scenario count")

    p_tr = sub.add_parser("transcript", help="Readable transcript of a run")
    p_tr.add_argument("run_id", help="Run id (reads checkpoints/<run_id>.jsonl)")
    p_tr.add_argument("--full", action="store_true",
                      help="Print full agent text instead of a 300-char excerpt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "transcript":
        return _transcript(args)

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


def _transcript(args) -> int:
    """Readable per-scenario transcript from the checkpoint (+ manifest)."""
    from pathlib import Path

    cp = Path("checkpoints") / f"{args.run_id}.jsonl"
    if not cp.exists():
        print(f"error: {cp} not found", file=sys.stderr)
        return 2

    # Scenario events from the generation manifest, if present
    events: dict[str, dict] = {}
    mp = Path("results") / f"{args.run_id}_scenario_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
        events = {r["scenario_id"]: r for r in manifest.get("scenarios", [])}

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
        meta = events.get(sid)
        if meta:
            print(f"optimal={meta['optimal']}  schedule={meta['schedule_id']}")
            print(f"\nEVENT:\n{meta['event']}")
        for t in sorted(deduped):
            rec = deduped[t]
            try:
                payload = json.loads(rec["raw_response"])
                text = payload.get("agent_text", "")
                tools = payload.get("tool_calls", [])
            except (ValueError, TypeError):
                text, tools = rec["raw_response"], []
            mark = "OK " if rec["is_correct"] else "XX "
            tool_str = ", ".join(
                f"{tc['name']}[{tc.get('classification', '?')}]" for tc in tools
            ) or "-"
            body = text if limit is None else text[:limit] + ("…" if len(text) > limit else "")
            print(f"\n[{mark}] turn {t}  decision={rec['decision']}  "
                  f"channels={rec['n_active_channels']}  tools: {tool_str}")
            print(f"      {body}")
    if failures:
        print(f"\n{'=' * 78}\nFAILURE RECORDS ({len(failures)}):")
        for f in failures:
            print(f"  {f['scenario_id']} turn {f['turn']}: {f['error']}")
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
    result = gen.generate(
        tier=args.tier,
        schedules=[args.schedule] if args.schedule else None,
        conditions=[args.condition] if args.condition else None,
        n_scenarios=args.max_scenarios,
    )

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
