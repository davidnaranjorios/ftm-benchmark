"""
Exp B v2 driver: run units = (case × schedule), 11 turns each, per-TURN
checkpointing (v1 lost a paid run to container restarts; v2 checkpoints every
turn and resumes mid-case by rebuilding the message history).

engine.py / runner.py untouched; only PRESSURE_SCHEDULES is imported.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from ftm.engine import PRESSURE_SCHEDULES
from ftm.expb2.gates5 import apply_pipeline
from ftm.expb2.pressure import occurrences_for_schedule
from ftm.expb2.prompts import arm_system_prompt, pressure_user_message, turn1_user_message
from ftm.expb2.resolver import resolve_completeness, resolve_decision

SCHEDULES = {s["id"]: s for s in PRESSURE_SCHEDULES}
N_TURNS = 11  # T1 + 10 pressure turns


# ── Checkpoint: one JSONL per (arm), key = (case_id, schedule_id, turn) ──────

class TurnCheckpoint:
    def __init__(self, checkpoint_dir, run_id, arm):
        self.dir = Path(checkpoint_dir) / f"expb2_{run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{arm}.turns.jsonl"

    def load(self) -> dict[tuple, dict]:
        done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["case_id"], r["schedule_id"], r["turn"])] = r
        return done

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


# ── Turn expansion from the frozen pressure pack ─────────────────────────────

def turn_messages(case: dict, schedule_id: str, pressure_pack: dict) -> list[list[dict]]:
    """Per pressure-turn (T2..T11) list of message dicts for this case."""
    sched = SCHEDULES[schedule_id]
    occ_rows = occurrences_for_schedule(sched["turns"])
    msgs_by_key = pressure_pack.get(case["case_id"], {})
    out = []
    for row in occ_rows:
        turn_msgs = []
        for channel, occ in row:
            m = msgs_by_key.get(f"{channel}:{occ}")
            if m is None:
                raise KeyError(
                    f"pressure pack missing {case['case_id']} {channel}:{occ}")
            turn_msgs.append(m)
        out.append(turn_msgs)
    return out


# ── Arm runner ───────────────────────────────────────────────────────────────

def run_arm(
    arm: str,
    run_units: list[tuple[dict, str]],       # [(case, schedule_id)]
    pressure_pack: dict,
    make_adapter,
    mitigation_text: str | None = None,
    checkpoint_dir=None,
    run_id: str = "run",
    max_turns: int = N_TURNS,
) -> list[dict]:
    system = None
    ckpt = TurnCheckpoint(checkpoint_dir, run_id, arm) if checkpoint_dir else None
    done = ckpt.load() if ckpt else {}
    if done:
        print(f"  [{arm}] resuming: {len(done)} turns already checkpointed")
    records: list[dict] = []

    for case, schedule_id in run_units:
        cid = case["case_id"]
        if system is None:
            system = arm_system_prompt(arm, mitigation_text)
        pturns = turn_messages(case, schedule_id, pressure_pack)
        adapter = make_adapter()

        messages: list[dict] = []
        for turn in range(1, max_turns + 1):
            user_msg = (turn1_user_message(case) if turn == 1
                        else pressure_user_message(turn, pturns[turn - 2]))
            key = (cid, schedule_id, turn)
            if key in done:
                r = done[key]
                records.append(r)
                messages.append({"role": "user", "content": r["raw_prompt"]})
                messages.append({"role": "assistant", "content": r["history_response"]})
                continue

            messages.append({"role": "user", "content": user_msg})
            t0 = time.perf_counter()
            out = adapter.complete(system, messages)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            text = out.get("text", "")
            model_decision, source = resolve_decision(text)
            comp = resolve_completeness(text) if arm == "R2b" else None

            if arm in ("R2a", "R2b"):
                pipe = apply_pipeline(
                    case, model_decision,
                    "2a" if arm == "R2a" else "2b",
                    completeness_reported=comp,
                )
                # History rewrite (declared limitation): the assistant message
                # entering the context carries the FINAL decision.
                if pipe["gate_intervened"] and pipe["forced_decision"]:
                    history_response = json.dumps({
                        "decision": pipe["final_decision"],
                        "rationale": (
                            f"[governance override gate={pipe['gate_id']}] "
                            "Decision set mechanically by the governance "
                            "pipeline."),
                    })
                else:
                    history_response = text
            else:
                pipe = {"final_decision": model_decision, "gate_id": None,
                        "gate_intervened": False, "forced_decision": None,
                        "completeness_used": None,
                        "ambiguity_static_would_fire": None}
                history_response = text

            turn_channels = ([] if turn == 1
                             else [m["channel"] for m in pturns[turn - 2]])
            record = {
                "run_id": run_id, "arm": arm,
                "case_id": cid, "gt_decision": case["gt_decision"],
                "gt_rule_id": case["gt_rule_id"],
                "schedule_id": schedule_id, "turn": turn,
                "channels": turn_channels,
                "actors": ([] if turn == 1
                           else [m["actor"] for m in pturns[turn - 2]]),
                "pressure_source": ([] if turn == 1 else
                                    [m["pressure_source"] for m in pturns[turn - 2]]),
                "model_decision": model_decision,
                "decision_source": source,
                "final_decision": pipe["final_decision"],
                "gate_id": pipe["gate_id"],
                "gate_intervened": pipe["gate_intervened"],
                "forced_decision": pipe["forced_decision"],
                "completeness_reported": comp,
                "completeness_used": pipe["completeness_used"],
                "ambiguity_static_would_fire": pipe["ambiguity_static_would_fire"],
                "raw_prompt": user_msg,
                "raw_response": text,
                "history_response": history_response,
                "latency_ms": latency_ms,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            if ckpt:
                ckpt.append(record)
            records.append(record)
            messages.append({"role": "assistant", "content": history_response})
    return records


# ── Budget ───────────────────────────────────────────────────────────────────

_CPT = 4  # chars per token, rough


def estimate_budget(run_units, pressure_pack, arms, mitigation_len=0) -> dict:
    per_arm = {}
    for arm in arms:
        sys_len = len(arm_system_prompt(arm, "x" * mitigation_len if arm == "R3" else None)) \
            if arm != "R3" or mitigation_len else len(arm_system_prompt("R0")) + mitigation_len
        in_tok = out_tok = 0
        for case, schedule_id in run_units:
            pturns = turn_messages(case, schedule_id, pressure_pack)
            user_lens = [len(turn1_user_message(case))] + [
                len(pressure_user_message(t + 2, pturns[t])) for t in range(len(pturns))
            ]
            out_per_turn = 220 if arm != "R2b" else 240
            cum = 0
            for t in range(N_TURNS):
                cum += user_lens[t] + out_per_turn * _CPT
                in_tok += (sys_len + cum) // _CPT
            out_tok += N_TURNS * out_per_turn
        per_arm[arm] = {"input_tokens": in_tok, "output_tokens": out_tok}
    totals = {k: sum(a[k] for a in per_arm.values())
              for k in ("input_tokens", "output_tokens")}
    return {"per_arm": per_arm, "totals": totals, "n_run_units": len(run_units)}


def print_budget(budget: dict, model: str) -> None:
    print(f"\n=== Exp B v2 budget estimate (model={model}) ===")
    print(f"run units (case x schedule): {budget['n_run_units']}")
    for arm, v in budget["per_arm"].items():
        print(f"  {arm:4s}  in={v['input_tokens']:>10,}  out={v['output_tokens']:>8,}")
    t = budget["totals"]
    print(f"  {'TOT':4s}  in={t['input_tokens']:>10,}  out={t['output_tokens']:>8,}")
    print("=" * 50)
