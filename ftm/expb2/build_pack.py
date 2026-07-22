"""
Frozen-pack builder for Exp B v2: cases (native mech_gov generator) +
pressure messages (validated, frozen) + manifest (hypotheses verbatim, SHAs,
deviations) — all committed BEFORE the first run call.

Case generation requires a checkout of mech-gov-framework (--mech-gov-path);
the frozen cases are then re-verifiable offline via ftm.expb2.rules.
Pressure generation uses the designated generator adapter (Sonnet per design
decision Q5) or the deterministic templates when adapter is None (offline).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ftm.engine import PRESSURE_SCHEDULES
from ftm.expb2 import ARMS, DEVIATIONS, DRIFT_ORDER, HONESTY_DECLARATION, HYPOTHESES
from ftm.expb2.pressure import generate_message, validate_message
from ftm.expb2.prompts import r0_sha256
from ftm.expb.arm1_policy import R1_SHA256
from ftm.expb2.r2_policy import R2_SHA256
from ftm.expb2.rules import verify_frozen_case

SCHEDULE_IDS = [s["id"] for s in PRESSURE_SCHEDULES]
N_OCCURRENCES = 9      # shock fires each channel 9 times (T2..T11)
CHANNELS = ["hierarchical", "peer", "reputational", "temporal", "emotional", "ambiguity"]


# ── Cases ────────────────────────────────────────────────────────────────────

def generate_cases(mech_gov_path: str, seed: int = 42, pool_n: int = 500,
                   per_type: int = 6) -> list[dict]:
    """Run mech_gov's own generator + ground truth; stratified deterministic
    sample: per_type cases per TransactionType, greedy per-decision balance."""
    sys.path.insert(0, str(Path(mech_gov_path) / "src"))
    from mech_gov.data.generator import generate_dataset_from_config
    from mech_gov.data.ground_truth import assign_ground_truth

    dist = Path(mech_gov_path) / "src/mech_gov/data/banking_distributions.yaml"
    pool = generate_dataset_from_config(
        seed=seed, n_cases_per_condition=pool_n, stress_conditions=["S0"],
        distributions_path=str(dist))
    assign_ground_truth(pool)
    det = [c for c in pool
           if c.gt_confidence and c.gt_confidence.value == "deterministic"]

    by_type: dict[str, list] = defaultdict(list)
    for c in sorted(det, key=lambda c: c.case_id):
        by_type[c.transaction_type.value].append(c)

    chosen = []
    for ttype in sorted(by_type):
        picked, decisions_seen = [], Counter()
        # greedy: prefer decisions not yet covered within the type
        candidates = sorted(by_type[ttype], key=lambda c: c.case_id)
        while len(picked) < per_type and candidates:
            candidates.sort(key=lambda c: (decisions_seen[c.gt_decision], c.case_id))
            c = candidates.pop(0)
            picked.append(c)
            decisions_seen[c.gt_decision] += 1
        chosen.extend(picked)

    out = []
    for c in sorted(chosen, key=lambda c: c.case_id):
        d = c.to_dict()
        keep = {k: d[k] for k in (
            "case_id", "transaction_type", "risk_score", "completeness",
            "regulatory_flags", "amount_usd", "jurisdiction",
            "customer_tenure_years", "counterparty_risk",
            "gt_decision", "gt_rule_id", "gt_rationale")}
        out.append(keep)
    dist_sha = hashlib.sha256(dist.read_bytes()).hexdigest()
    return out, {"seed": seed, "pool_n": pool_n, "pool_deterministic": len(det),
                 "per_type": per_type, "distributions_sha256": dist_sha}


# ── Pressure pack ────────────────────────────────────────────────────────────

def build_pressure(cases: list[dict], adapter=None, max_attempts: int = 3,
                   progress=None) -> dict:
    """{case_id: {"channel:occurrence": message}} for occurrences 1..9.
    Turn passed to the generator = occurrence + 2 (clock coherence: the k-th
    firing cannot happen before turn k+2 in any schedule... conservative)."""
    pack: dict[str, dict] = {}
    for i, case in enumerate(cases):
        row = {}
        for ch in CHANNELS:
            for occ in range(1, N_OCCURRENCES + 1):
                msg = generate_message(case, ch, occ, turn=occ + 2,
                                       adapter=adapter, max_attempts=max_attempts)
                ok, reason = validate_message(msg["text"], case)
                if not ok:  # belt & suspenders: templates must always pass
                    raise AssertionError(
                        f"invariant violation survived generation: {reason}")
                row[f"{ch}:{occ}"] = msg
        pack[case["case_id"]] = row
        if progress:
            progress(i + 1, len(cases))
    return pack


# ── Manifest + write ─────────────────────────────────────────────────────────

def build_manifest(cases, case_meta, pressure_pack, generator_model: str) -> dict:
    n_generated = sum(
        1 for row in pressure_pack.values() for m in row.values()
        if m["pressure_source"] == "generated")
    n_total = sum(len(row) for row in pressure_pack.values())
    return {
        "experiment": "expB2_stakeholder_pressure_S4",
        "drift_order": DRIFT_ORDER,
        "hypotheses": HYPOTHESES,
        "honesty_declaration": HONESTY_DECLARATION,
        "deviations": DEVIATIONS,
        "arms": list(ARMS),
        "prompt_sha256": {"R1": R1_SHA256, "R2": R2_SHA256, "R0_cut": r0_sha256()},
        "cases": {
            "n": len(cases),
            "generation": case_meta,
            "by_type": dict(Counter(c["transaction_type"] for c in cases)),
            "by_decision": dict(Counter(c["gt_decision"] for c in cases)),
            "by_rule": dict(Counter(c["gt_rule_id"] for c in cases)),
            "case_ids": [c["case_id"] for c in cases],
        },
        "schedules": SCHEDULE_IDS,
        "run_units": len(cases) * len(SCHEDULE_IDS),
        "pressure": {
            "generator_model": generator_model,
            "n_messages": n_total,
            "n_generated": n_generated,
            "n_template_fallback": n_total - n_generated,
            "invariant": ("machine-verified: numeric allowlist + new-assertion "
                          "detector + deterministic template fallback"),
        },
    }


def write_pack(out_dir, cases, case_meta, pressure_pack, generator_model="template"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        assert verify_frozen_case(case), f"GT mismatch (offline rules): {case['case_id']}"
    (out / "cases.json").write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    (out / "pressure.json").write_text(
        json.dumps(pressure_pack, indent=1, sort_keys=True) + "\n")
    manifest = build_manifest(cases, case_meta, pressure_pack, generator_model)
    manifest["files_sha256"] = {
        "cases.json": hashlib.sha256((out / "cases.json").read_bytes()).hexdigest(),
        "pressure.json": hashlib.sha256((out / "pressure.json").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def load_pack(pack_dir):
    p = Path(pack_dir)
    cases = json.loads((p / "cases.json").read_text())
    pressure = json.loads((p / "pressure.json").read_text())
    manifest = json.loads((p / "manifest.json").read_text())
    return cases, pressure, manifest
