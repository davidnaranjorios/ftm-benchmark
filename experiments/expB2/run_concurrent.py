"""Concurrent Exp B v2 publishable run: units processed in parallel (threads),
turn-loop sequential within a unit, per-turn checkpoint (thread-safe), resumes
from existing checkpoint. Provider+quant pinned, output capped @120."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, "/home/user/ftm-benchmark")

from ftm.engine import PRESSURE_SCHEDULES
from ftm.expb.adapters import OpenRouterAdapter
from ftm.expb2 import DEVIATIONS, HONESTY_DECLARATION
from ftm.expb2.build_pack import load_pack
from ftm.expb2.driver import N_TURNS, TurnCheckpoint, turn_messages
from ftm.expb2.gates5 import apply_pipeline
from ftm.expb2.prompts import arm_system_prompt, pressure_user_message, turn1_user_message
from ftm.expb2.resolver import resolve_completeness, resolve_decision

PACK = "/home/user/ftm-benchmark/scenarios/packs/expb2_banking_v0"
CKPT = "/home/user/ftm-benchmark/checkpoints"
OUTROOT = "/home/user/ftm-benchmark/results/expb2_publishable"
ARMS = ["R0", "R1", "R2a", "R2b"]
MAX_TOKENS = 120
WORKERS = 8

MODELS = {
    "A": {"slug": "deepseek/deepseek-v3.2", "family": "DeepSeek",
          "provider": {"order": ["Novita"], "allow_fallbacks": False, "quantizations": ["fp8"]},
          "run_id": "v2pub_deepseek"},
    "B": {"slug": "meta-llama/llama-3.3-70b-instruct", "family": "Meta-Llama",
          "provider": {"order": ["DeepInfra"], "allow_fallbacks": False, "quantizations": ["fp8"]},
          "run_id": "v2pub_llama"},
}


def process_unit(case, schedule_id, arm, pressure, make_adapter, ckpt, lock, done):
    cid = case["case_id"]
    pturns = turn_messages(case, schedule_id, pressure)
    system = arm_system_prompt(arm)
    adapter = make_adapter()
    messages, out_records = [], []
    for turn in range(1, N_TURNS + 1):
        key = (cid, schedule_id, turn)
        um = turn1_user_message(case) if turn == 1 else pressure_user_message(turn, pturns[turn - 2])
        if key in done:
            r = done[key]
            messages.append({"role": "user", "content": r["raw_prompt"]})
            messages.append({"role": "assistant", "content": r["history_response"]})
            continue
        messages.append({"role": "user", "content": um})
        out = adapter.complete(system, messages)
        text = out.get("text", "")
        model_decision, source = resolve_decision(text)
        comp = resolve_completeness(text) if arm == "R2b" else None
        if arm in ("R2a", "R2b"):
            pipe = apply_pipeline(case, model_decision, "2a" if arm == "R2a" else "2b",
                                  completeness_reported=comp)
            if pipe["gate_intervened"] and pipe["forced_decision"]:
                history_response = json.dumps({"decision": pipe["final_decision"],
                    "rationale": f"[governance override gate={pipe['gate_id']}]"})
            else:
                history_response = text
        else:
            pipe = {"final_decision": model_decision, "gate_id": None, "gate_intervened": False,
                    "forced_decision": None, "completeness_used": None,
                    "ambiguity_static_would_fire": None}
            history_response = text
        tch = [] if turn == 1 else [m["channel"] for m in pturns[turn - 2]]
        rec = {"arm": arm, "case_id": cid, "gt_decision": case["gt_decision"],
               "gt_rule_id": case["gt_rule_id"], "schedule_id": schedule_id, "turn": turn,
               "channels": tch, "actors": [] if turn == 1 else [m["actor"] for m in pturns[turn - 2]],
               "pressure_source": [] if turn == 1 else [m["pressure_source"] for m in pturns[turn - 2]],
               "model_decision": model_decision, "decision_source": source,
               "final_decision": pipe["final_decision"], "gate_id": pipe["gate_id"],
               "gate_intervened": pipe["gate_intervened"], "forced_decision": pipe["forced_decision"],
               "completeness_reported": comp, "completeness_used": pipe["completeness_used"],
               "ambiguity_static_would_fire": pipe["ambiguity_static_would_fire"],
               "raw_prompt": um, "raw_response": text, "history_response": history_response,
               "provider": out.get("provider"), "ts": datetime.now().isoformat(timespec="seconds")}
        with lock:
            ckpt.append(rec)
        out_records.append(rec)
        messages.append({"role": "assistant", "content": history_response})
    return cid, schedule_id, len(out_records)


def run_model(key, cases, pressure, pack_manifest):
    m = MODELS[key]
    scheds = [s["id"] for s in PRESSURE_SCHEDULES]
    units = [(c, s) for c in cases for s in scheds]
    served = {}
    served_lock = threading.Lock()

    def make_adapter():
        return OpenRouterAdapter(m["slug"], max_tokens=MAX_TOKENS, provider=m["provider"])

    t0 = time.time()
    for arm in ARMS:
        ckpt = TurnCheckpoint(CKPT, m["run_id"], arm)
        done = ckpt.load()
        lock = threading.Lock()
        todo = [(c, s) for c, s in units
                if not all((c["case_id"], s, t) in done for t in range(1, N_TURNS + 1))]
        print(f"[{key}/{arm}] {len(units)-len(todo)} units done, {len(todo)} to run", flush=True)

        def work(cs):
            c, s = cs
            a = make_adapter()
            try:
                r = process_unit(c, s, arm, pressure, lambda: a, ckpt, lock, done)
            except Exception as e:  # backstop: one bad unit must not kill the run
                print(f"  [{key}/{arm}] unit {c['case_id']}/{s} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = (c["case_id"], s, -1)
            if a.served:
                with served_lock:
                    served[a.served["provider"]] = served.get(a.served["provider"], 0) + 1
            return r

        n = failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for fut in as_completed([ex.submit(work, cs) for cs in todo]):
                cid, s, k = fut.result()
                if k == -1:
                    failed += 1
                n += 1
                if n % 20 == 0:
                    print(f"  [{key}/{arm}] {n}/{len(todo)} units ({time.time()-t0:.0f}s)", flush=True)
        # write consolidated records for this arm from checkpoint
        outdir = f"{OUTROOT}/{m['run_id']}"
        os.makedirs(outdir, exist_ok=True)
        allrec = list(TurnCheckpoint(CKPT, m["run_id"], arm).load().values())
        allrec.sort(key=lambda r: (r["case_id"], r["schedule_id"], r["turn"]))
        with open(f"{outdir}/{arm}.records.jsonl", "w") as f:
            for r in allrec:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"[{key}/{arm}] DONE {len(allrec)} turns ({time.time()-t0:.0f}s)", flush=True)

    outdir = f"{OUTROOT}/{m['run_id']}"
    json.dump({"model_key": key, "slug": m["slug"], "family": m["family"],
               "pinned_provider": m["provider"], "served_providers": served,
               "max_tokens": MAX_TOKENS, "hypotheses": pack_manifest["hypotheses"],
               "honesty_declaration": HONESTY_DECLARATION, "deviations": DEVIATIONS,
               "pack_files_sha256": pack_manifest.get("files_sha256")},
              open(f"{outdir}/run_manifest.json", "w"), indent=2, sort_keys=True)
    print(f"[{key}] served: {served}", flush=True)


def main():
    only = next((a for a in sys.argv[1:] if a in ("A", "B")), None)
    cases, pressure, pack_manifest = load_pack(PACK)
    for k in ([only] if only else ["A", "B"]):
        print(f"\n### Model {k}: {MODELS[k]['slug']} ###", flush=True)
        run_model(k, cases, pressure, pack_manifest)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
