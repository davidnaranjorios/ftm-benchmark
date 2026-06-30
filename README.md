# ftm-benchmark

FTM Evaluation Engine — **neutral baseline build**.

This is the clean measurement core of the FTM benchmark: it generates pressure
scenarios across 5 domains, parses and classifies model decisions, and computes
the full v10 metric suite (DIS, ABI, PRI, FARP variants, conviction decay,
rationalization drift, archetype detection, etc.).

The engine runs **only on the neutral `DEFAULT_SYSTEM_PROMPT`**. It deliberately
contains **no intervention ("cura") prompt blocks** and no prompt-optimization
logic — it measures behavior, it does not try to fix it. `detect_archetype()`
diagnoses the failure mode and notes that a mitigation is required, without
naming or describing any concrete intervention.

## Requirements

Python 3.9+ and the standard library only. No third-party dependencies.

## Usage

```python
import ftm

# 1. Generate scenarios for a tier ("snapshot" | "standard" | "extended" | "research")
scenarios = ftm.generate_scenarios(tier="snapshot")

# 2. For each scenario / turn, build the prompts to send to a model under test
sc = scenarios[0]
system_prompt = ftm.build_ftm_system_prompt(sc)        # uses the scenario role + format suffix
user_msg = ftm.build_turn_user_message(sc, turn=1)

# (call your model here, collect raw_response per turn)

# 3. Parse and classify responses, build TurnResult logs, then compute metrics
parsed = ftm.parse_decision(raw_response)              # {"decision", "confidence", "reason"}
reason_class = ftm.classify_reason(parsed["reason"])   # DATA | PRESSURE | MIXED | EMPTY

# logs: list[ftm.TurnResult]
metrics = ftm.compute_metrics(logs)
archetype = ftm.detect_archetype(metrics)
print(archetype.name, archetype.risk)
```

`ftm.DEFAULT_SYSTEM_PROMPT` is the neutral baseline prompt used for evaluation.

## Layout

- `ftm/engine.py` — the full engine (constants, scenario generator, parsers,
  metrics, archetype detector, prompt builders).
- `ftm/__init__.py` — public re-exports.
