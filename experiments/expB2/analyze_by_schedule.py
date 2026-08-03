"""Causal cut by schedule_category — control_flat vs ramp/shock — over the
already-run v2 publishable data. No model calls. The control_flat arm carries
no escalated pressure, so drifts there are baseline instability; the
control-vs-pressure subtraction is the causal claim.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/home/user/ftm-benchmark")
from ftm.expb2 import DRIFT_RANK
from ftm.expb2.drift import case_drift

ROOT = "/home/user/ftm-benchmark/results/expb2_publishable"
MODELS = {"DeepSeek-v3.2": "v2pub_deepseek", "Llama-3.3-70b": "v2pub_llama"}
ARMS = ["R0", "R1", "R2a", "R2b"]
SCHED_CAT = {"control_flat": "control", "ramp_emot_lead": "ramp",
             "shock_immediate": "shock"}
PRESSURE_ONSET = 3  # ramp & shock: first pressure turn is T3


def load(run_id, arm):
    return [json.loads(l) for l in open(f"{ROOT}/{run_id}/{arm}.records.jsonl")]


def by_case(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["case_id"]].append(r)
    return d


def stratum(recs, field):
    """Aggregate drift over a set of records (one schedule × arm)."""
    per = {}
    for cid, rs in by_case(recs).items():
        cd = case_drift(rs, field)
        if cd is not None:
            per[cid] = cd
    unsafe_denom = {c: v for c, v in per.items()
                    if v["t1_rank"] < DRIFT_RANK["APPROVE"]}
    cons_denom = {c: v for c, v in per.items()
                  if v["t1_rank"] > DRIFT_RANK["DECLINE"]}
    n_u = len(unsafe_denom)
    n_events = sum(1 for v in unsafe_denom.values() if v["unsafe_event"])
    first_turns = sorted(v["unsafe_first_turn"] for v in unsafe_denom.values()
                         if v["unsafe_event"])
    return {
        "n_cases_measurable": len(per),
        "unsafe_events": n_events, "unsafe_denom": n_u,
        "unsafe_first_turns": first_turns,
        "conservative_events": sum(1 for v in cons_denom.values() if v["conservative_event"]),
        "conservative_denom": len(cons_denom),
    }


def main():
    report = {"pressure_onset_turn": PRESSURE_ONSET,
              "note": ("control_flat carries no escalated pressure; drifts there "
                       "are baseline instability. Net pressure effect = "
                       "(ramp+shock) - control, per arm."),
              "models": {}}

    md = ["# Experiment B v2 — causal cut by schedule (control vs pressure)", "",
          "No model calls; re-analysis of persisted turns. **Pressure onset = turn 3** "
          "for ramp/shock; control_flat has no escalated pressure. Denominators are "
          "small (~8–10 cases per schedule×arm) — counts and directions only, no strong "
          "verdicts.", ""]

    for mname, run_id in MODELS.items():
        report["models"][mname] = {}
        md += [f"## {mname}", "",
               "unsafe = model wanted (model_decision) / final (post-gate). "
               "Format: events/denominator.", "",
               "| schedule | arm | unsafe model | unsafe final | conserv. | first-drift turns (final) |",
               "|---|---|---|---|---|---|"]
        # balance check
        balance = {}
        for arm in ARMS:
            recs = load(run_id, arm)
            for sid in SCHED_CAT:
                srecs = [r for r in recs if r["schedule_id"] == sid]
                ncases = len({r["case_id"] for r in srecs})
                balance[(arm, sid)] = ncases
                sm = stratum(srecs, "model_decision")
                sf = stratum(srecs, "final_decision")
                report["models"][mname].setdefault(arm, {})[SCHED_CAT[sid]] = {
                    "model": sm, "final": sf, "n_cases": ncases}
                md.append(
                    f"| {SCHED_CAT[sid]} | {arm} | {sm['unsafe_events']}/{sm['unsafe_denom']} "
                    f"| {sf['unsafe_events']}/{sf['unsafe_denom']} "
                    f"| {sf['conservative_events']}/{sf['conservative_denom']} "
                    f"| {sf['unsafe_first_turns'] or '—'} |")
        # balance verification
        bvals = set(balance.values())
        md += ["", f"**Balance:** cases per (arm×schedule) = "
               f"{'BALANCED at '+str(bvals.pop()) if len(bvals)==1 else 'UNBALANCED: '+str(balance)}"]

        # model-decision first-drift turns per schedule (baseline-instability probe)
        def model_first_turns(arm, sid):
            recs = [r for r in load(run_id, arm) if r["schedule_id"] == sid]
            return stratum(recs, "model_decision")["unsafe_first_turns"]

        # central subtraction: control vs pressure, aggregated across arms and per arm
        md += ["", "### Control vs presión (la resta causal)", "",
               "Temporal probe on MODEL decision (where baseline instability shows): "
               f"drift turns < {PRESSURE_ONSET} are not attributable to pressure.", "",
               "| arm | control final | ramp+shock final | control model | ramp+shock model | model-drift turns (shock) | before-onset (model, r+s) |",
               "|---|---|---|---|---|---|---|"]
        agg = {"ctrl_f": [0, 0], "pr_f": [0, 0], "ctrl_m": [0, 0], "pr_m": [0, 0],
               "before_onset": 0, "at_after_onset": 0}
        for arm in ARMS:
            d = report["models"][mname][arm]
            cf = d["control"]["final"]; cm = d["control"]["model"]
            pf_e = d["ramp"]["final"]["unsafe_events"] + d["shock"]["final"]["unsafe_events"]
            pf_d = d["ramp"]["final"]["unsafe_denom"] + d["shock"]["final"]["unsafe_denom"]
            pm_e = d["ramp"]["model"]["unsafe_events"] + d["shock"]["model"]["unsafe_events"]
            pm_d = d["ramp"]["model"]["unsafe_denom"] + d["shock"]["model"]["unsafe_denom"]
            # temporal coherence: first-drift turns before onset in ramp+shock (final)
            pturns = (d["ramp"]["final"]["unsafe_first_turns"]
                      + d["shock"]["final"]["unsafe_first_turns"])
            before = sum(1 for t in pturns if t < PRESSURE_ONSET)
            after = sum(1 for t in pturns if t >= PRESSURE_ONSET)
            # model-decision temporal probe (ramp+shock)
            mturns = model_first_turns(arm, "ramp_emot_lead") + model_first_turns(arm, "shock_immediate")
            m_before = sum(1 for t in mturns if t < PRESSURE_ONSET)
            m_after = sum(1 for t in mturns if t >= PRESSURE_ONSET)
            shock_mturns = model_first_turns(arm, "shock_immediate")
            agg["ctrl_f"][0] += cf["unsafe_events"]; agg["ctrl_f"][1] += cf["unsafe_denom"]
            agg["pr_f"][0] += pf_e; agg["pr_f"][1] += pf_d
            agg["ctrl_m"][0] += cm["unsafe_events"]; agg["ctrl_m"][1] += cm["unsafe_denom"]
            agg["pr_m"][0] += pm_e; agg["pr_m"][1] += pm_d
            agg["before_onset"] += before; agg["at_after_onset"] += after
            md.append(f"| {arm} | {cf['unsafe_events']}/{cf['unsafe_denom']} | {pf_e}/{pf_d} "
                      f"| {cm['unsafe_events']}/{cm['unsafe_denom']} | {pm_e}/{pm_d} "
                      f"| {shock_mturns or '—'} | {m_before} before / {m_after} at-after |")
        md.append(f"| **ALL** | **{agg['ctrl_f'][0]}/{agg['ctrl_f'][1]}** | "
                  f"**{agg['pr_f'][0]}/{agg['pr_f'][1]}** | {agg['ctrl_m'][0]}/{agg['ctrl_m'][1]} "
                  f"| {agg['pr_m'][0]}/{agg['pr_m'][1]} "
                  f"| **{agg['before_onset']} before / {agg['at_after_onset']} at-after** |")
        report["models"][mname]["_aggregate"] = agg
        md.append("")

    with open(f"{ROOT}/report_by_schedule.json", "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    import shutil
    shutil.copy(f"{ROOT}/report_by_schedule.json",
                "/home/user/ftm-benchmark/experiments/expB2/report_by_schedule.json")
    text = "\n".join(md) + "\n"
    open("/home/user/ftm-benchmark/experiments/expB2/report_by_schedule.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
