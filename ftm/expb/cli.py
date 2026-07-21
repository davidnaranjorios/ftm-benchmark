"""
Experiment B CLI — offline-first. Runs the arms over the ftm_banking_v0 pack,
prints a budget estimate, and (for non-mock models) requires --confirm-budget
before any model call. Writes results/expB_report_<run_id>.{json,md} and
per-arm gate logs.

Examples:
  # offline pilot with the mock subject (no tokens, no confirmation needed)
  python -m ftm.expb.cli --model mock --adapter mock --scenarios snapshot=5

  # real pilot on a cheap model
  python -m ftm.expb.cli --model claude-haiku-4-5-20251001 \
      --scenarios snapshot=5 --confirm-budget
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from ftm.adapters import get_adapter
from ftm.expb import ARMS
from ftm.expb.analysis import arm_summary, build_report, write_report
from ftm.expb.driver import (
    build_run_manifest,
    estimate_budget,
    run_arm,
    _print_budget,
)
from ftm.expb.mapping import load_pack_specs
from ftm.grounding.santander import load_pack_scenarios


def _select_scenarios(scenarios, specs, mode: str):
    core = [s for s in scenarios if _is_core(s.scenario_id, specs)]
    supp = [s for s in scenarios if not _is_core(s.scenario_id, specs)]
    if mode == "core":
        return core, supp
    if mode == "all":
        return scenarios, []
    if mode.startswith("snapshot"):
        n = int(mode.split("=")[1]) if "=" in mode else 5
        return core[:n], []
    raise ValueError(f"bad --scenarios {mode!r}")


def _is_core(scenario_id: str, specs: dict) -> bool:
    # core scenarios have "_cellX_" in the id; supplementary have "_supp_"
    return "_supp_" not in scenario_id


def _load_mitigation(path: str | None) -> tuple[str | None, dict]:
    if not path:
        return None, {"provided": False, "note": "ARM-3 omitted (no --mitigation-file)"}
    p = Path(path)
    text = p.read_text()
    # Never record the content — only sha + length.
    return text, {
        "provided": True,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "length_chars": len(text),
        "note": "Operator-private mitigation; content never recorded or committed.",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ftm.expb")
    ap.add_argument("--pack", default="scenarios/packs/ftm_banking_v0")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--adapter", default="mock")
    ap.add_argument("--scenarios", default="core",
                    help="core | all | snapshot[=N]")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of arms")
    ap.add_argument("--mitigation-file", default=None,
                    help="operator-private ARM-3 prompt (never committed)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="per-(arm,scenario) checkpoints; a re-run with the "
                         "same --run-id resumes and skips completed scenarios")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--confirm-budget", action="store_true",
                    help="required to call a non-mock model")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-tokens-out", type=int, default=256,
                    help="max output tokens per turn (openrouter adapter)")
    args = ap.parse_args(argv)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    all_scenarios = load_pack_scenarios(args.pack)
    specs = load_pack_specs(args.pack)
    scenarios, supp_scenarios = _select_scenarios(all_scenarios, specs, args.scenarios)

    mitigation_text, mitigation_meta = _load_mitigation(args.mitigation_file)
    if "ARM-3" in arms and mitigation_text is None:
        print("[expB] ARM-3 requested but no --mitigation-file; dropping ARM-3.")
        arms = [a for a in arms if a != "ARM-3"]

    budget = estimate_budget(
        scenarios, arms, mitigation_len=mitigation_meta.get("length_chars", 0)
    )
    _print_budget(budget, args.model)

    is_mock = args.adapter == "mock" or args.model == "mock"
    if not is_mock and not args.confirm_budget:
        print("[expB] Non-mock model requires --confirm-budget to proceed. "
              "Nothing was called.")
        return 2

    def make_adapter():
        if args.adapter == "openrouter":
            from ftm.expb.adapters import OpenRouterAdapter
            return OpenRouterAdapter(args.model, max_tokens=args.max_tokens_out)
        return get_adapter(args.model, args.adapter)

    arm_turns: dict[str, list] = {}
    gate_logs: dict[str, list] = {}
    for arm in arms:
        turns, glog = run_arm(
            arm, scenarios, specs, make_adapter,
            mitigation_text=mitigation_text, max_turns=args.max_turns,
            checkpoint_dir=args.checkpoint_dir, run_id=run_id,
        )
        arm_turns[arm] = turns
        if glog:
            gate_logs[arm] = glog

    # Supplementary descriptive (core arms only, outside H1–H3)
    supplementary = None
    if supp_scenarios:
        supp_summaries = {}
        for arm in arms:
            s_turns, _ = run_arm(
                arm, supp_scenarios, specs, make_adapter,
                mitigation_text=mitigation_text, max_turns=args.max_turns,
                checkpoint_dir=args.checkpoint_dir, run_id=f"{run_id}_supp",
            )
            supp_summaries[arm] = arm_summary(s_turns)
        supplementary = {"label": "supplementary_stressed_only",
                         "outside_hypotheses": True,
                         "arm_summaries": supp_summaries}

    run_manifest = build_run_manifest(args.model, arms, budget, args.pack, mitigation_meta)
    report = build_report(arm_turns, gate_logs, run_manifest, supplementary)

    out_dir = Path(args.out)
    jpath, mpath = write_report(report, out_dir, run_id)
    # Persist gate logs alongside the report.
    if gate_logs:
        gl_dir = out_dir / f"expB_{run_id}_gatelogs"
        gl_dir.mkdir(parents=True, exist_ok=True)
        for arm, glog in gate_logs.items():
            (gl_dir / f"{arm}.jsonl").write_text(
                "\n".join(json.dumps(e, sort_keys=True) for e in glog) + "\n"
            )
    print(f"\n[expB] report: {jpath}\n[expB] table:  {mpath}")
    _print_verdicts(report)
    return 0


def _print_verdicts(report):
    h = report["hypotheses"]
    if h is None:
        print("\n(hypotheses not evaluated — partial/single-arm run)")
        for arm, s in report["arm_summaries"].items():
            print(f"  {arm}: FARP={s['farp_strict']:.3f} "
                  f"folded={s['n_folded']}/{s['n_stay_measurable']} "
                  f"parse_fail={s['parse_fail_rate']}")
        return
    print("\n=== Hypotheses ===")
    for k in ("H1", "H2", "H3"):
        print(f"  {k}: {h[k]['verdict']}")
    s = h["sanity_ARM_2a"]
    print(f"  ARM-2a sanity FARP={s['farp_strict']:.3f} "
          f"({'PASS' if s['passes'] else 'FAIL'})")


if __name__ == "__main__":
    raise SystemExit(main())
