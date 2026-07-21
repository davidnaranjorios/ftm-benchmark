# Attribution

ftm-benchmark uses open artifacts published by Santander AI Lab. It makes no
claim of endorsement by Santander AI Lab or Santander Group; all artifacts
are used in accordance with their published open licenses.

## 1. Vendored governance code — mech-gov-framework (Apache-2.0)

Experiment B (`ftm/expb/`) reuses mechanical-governance primitives from
mech-gov-framework. Rather than depend on the package at runtime, the needed
logic is **transcribed verbatim** (logic and thresholds) as literals, with
the SHA-256 of each upstream source file pinned in code and in every run
manifest. No upstream source files are copied into this repository.

- **License:** Apache License 2.0
- **Source:** Santander AI Lab. *mech_gov: Mechanical Governance for LLM
  Decisions.* Version 0.1.0, 2026-06-12.
  <https://github.com/SantanderAI/mech-gov-framework>
- **Transcribed:**
  - `primitives/hard_gates.py` → `ftm/expb/gates.py`
    (`evaluate_hard_gates`, `build_default_gates`, the K0_* gate table)
  - `primitives/ambiguity_gate.py` → `ftm/expb/gates.py` (`ambiguity_gate`)
  - `policy_templates/r1_system_prompt.txt` → `ftm/expb/arm1_policy.py`
    (ARM-1 R1 text, byte-identical; the FTM protocol bridge is ours)

Per-turn application of these primitives over a multi-turn pressure protocol
is an extension by the FTM authors, not a documented use of
mech-gov-framework (see the honesty declaration in each run manifest).

## 2. Benchmark data — Stressed German Credit Dataset (CC BY 4.0)

The scenario pack `ftm_banking_v0` derives READING values from a curated
subset of SGCD. Full data attribution (Santander AI Lab + the original UCI
Statlog source) is in
[`scenarios/packs/ftm_banking_v0/ATTRIBUTION.md`](scenarios/packs/ftm_banking_v0/ATTRIBUTION.md).
