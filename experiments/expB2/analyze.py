"""Analyze the v2 publishable run: per-model, per-arm unsafe/conservative drift
(model_decision AND final_decision), R2 model-wanted-vs-blocked, H3 trajectory,
inter-family comparison. Writes results/expb2_publishable/report.json + .md."""
import json
import sys
from collections import Counter

sys.path.insert(0, "/home/user/ftm-benchmark")
from ftm.expb2.build_pack import load_pack
from ftm.expb2.drift import (
    arm_drift_summary, evaluate_hypotheses, farp_equivalence_annex, h3_analysis,
)

ROOT = "/home/user/ftm-benchmark/results/expb2_publishable"
MODELS = {"A": ("deepseek/deepseek-v3.2", "v2pub_deepseek"),
          "B": ("meta-llama/llama-3.3-70b-instruct", "v2pub_llama")}
ARMS = ["R0", "R1", "R2a", "R2b"]


def load_arm(run_id, arm):
    p = f"{ROOT}/{run_id}/{arm}.records.jsonl"
    return [json.loads(l) for l in open(p)]


def analyze_model(run_id, cases_by_id):
    gt = {cid: c["gt_decision"] for cid, c in cases_by_id.items()}
    recs = {a: load_arm(run_id, a) for a in ARMS}
    final = {a: arm_drift_summary(recs[a], "final_decision", gt_by_case=gt) for a in ARMS}
    model = {a: arm_drift_summary(recs[a], "model_decision", gt_by_case=gt) for a in ARMS}
    h3 = h3_analysis(recs["R2b"], cases_by_id)
    verdicts = evaluate_hypotheses(final, h3)
    # R2 model-wanted vs blocked
    blocked = {}
    for a in ("R2a", "R2b"):
        interv = [r for r in recs[a] if r["gate_intervened"]]
        wanted_permissive = [r for r in recs[a] if r["model_decision"] != r["final_decision"]
                             and r["gate_id"]]
        blocked[a] = {"gate_intervened_turns": len(interv),
                      "gates_fired": dict(Counter(r["gate_id"] for r in recs[a] if r["gate_id"])),
                      "model_wanted_diff_final": len(wanted_permissive)}
    parse = {a: sum(1 for r in recs[a] if r["model_decision"] == "PARSE_FAIL") for a in ARMS}
    return {"final": final, "model": model, "h3": h3, "verdicts": verdicts,
            "r2_blocking": blocked, "parse_fail": parse,
            "farp_equiv": farp_equivalence_annex(recs["R0"], gt)}


def fmt_rate(s):
    r = s["unsafe_drift_rate"]
    ci = s.get("unsafe_drift_ci")
    cis = f"[{ci['lower']:.2f},{ci['upper']:.2f}]" if ci else "—"
    return f"{r if r is not None else '—'} {cis} ({s['n_unsafe_events']}/{s['n_unsafe_denominator']})"


def main():
    cases, _, pack_manifest = load_pack("/home/user/ftm-benchmark/scenarios/packs/expb2_banking_v0")
    by = {c["case_id"]: c for c in cases}
    report = {"pack_hypotheses": pack_manifest["hypotheses"], "models": {}}
    md = ["# Experiment B v2 — banking governance under stakeholder pressure",
          "", f"Pack: ftm_banking_v0 · 30 cases × 3 schedules × 4 arms · output capped @120",
          ""]
    for key, (slug, run_id) in MODELS.items():
        try:
            served = json.load(open(f"{ROOT}/{run_id}/run_manifest.json")).get("served_providers")
        except FileNotFoundError:
            served = None
        a = analyze_model(run_id, by)
        report["models"][key] = {"slug": slug, "served": served, **a}
        md += [f"## Model {key}: {slug}  (served: {served})", "",
               "**unsafe-drift — MODEL wanted vs FINAL (post-gate):**", "",
               "| Arm | model unsafe | final unsafe | final conservative | t1-acc vs GT | parse-fail |",
               "|---|---|---|---|---|---|"]
        for arm in ARMS:
            fm, mo = a["final"][arm], a["model"][arm]
            md.append(f"| {arm} | {fmt_rate(mo)} | {fmt_rate(fm)} | "
                      f"{fm['conservative_drift_rate']} | {fm['t1_accuracy_vs_gt']} | {a['parse_fail'][arm]} |")
        v = a["verdicts"]
        md += ["", f"**Verdicts:** H1={v['H1']['verdict']} · H2={v['H2']['verdict']} · H3={v['H3']['verdict']}",
               "", f"**R2 blocking (non-tautological):** "
               f"R2a model≠final in {a['r2_blocking']['R2a']['model_wanted_diff_final']} turns, "
               f"gates {a['r2_blocking']['R2a']['gates_fired']}",
               "", f"**H3:** sensitive={a['h3']['n_sensitive_cases']} "
               f"mean_completeness_delta_after_ambiguity={a['h3']['mean_delta_after_ambiguity']} "
               f"enabled_drifts={a['h3']['n_enabled_drifts']}", ""]
    json.dump(report, open(f"{ROOT}/report.json", "w"), indent=2, sort_keys=True, default=str)
    open(f"{ROOT}/report.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten: {ROOT}/report.json, report.md")


if __name__ == "__main__":
    main()
