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
- `ftm/adapters.py` — `ModelAdapter` interface + `OpenAIAdapter`, `AnthropicAdapter`,
  `MockAdapter` (deterministic, no API cost), and `get_adapter()` factory.
- `ftm/runner.py` — resilient benchmark runner with checkpoint/resumption and CLI.
- `tests/test_resumption.py` — acceptance test for the resumption contract.

---

## Runner — quick start

### Mock run (no API key needed)

```bash
python -m ftm.runner --models mock --adapter mock --tier snapshot --domain devops_server
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
python -m ftm.runner --models gpt-4o --tier standard
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m ftm.runner --models claude-sonnet-4-6 --tier standard
```

### Multiple models in one sweep

```bash
python -m ftm.runner \
  --models gpt-4o claude-sonnet-4-6 \
  --tier standard \
  --domain financial
```

### Resume an interrupted run

Pass the same `--run-id` that was used before. The runner reads
`checkpoints/<run_id>.jsonl`, skips already-completed (scenario, turn) pairs,
rebuilds the per-scenario message history from `raw_prompt`/`raw_response`
records, and continues from where it left off.

```bash
python -m ftm.runner \
  --models gpt-4o \
  --tier standard \
  --run-id 20250101_120000_gpt-4o   # same ID as the crashed run
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--models` | (required) | One or more model identifiers |
| `--tier` | `standard` | `snapshot` / `standard` / `extended` / `research` |
| `--domain` | all | Filter to one of the 5 domains |
| `--run-id` | timestamp | Stable ID for checkpointing; reuse to resume |
| `--adapter` | `auto` | `auto` / `openai` / `anthropic` / `mock` |

### Output files

```
checkpoints/<run_id>.jsonl       # append-only, one TurnResult per line
results/<run_id>.jsonl           # one record per completed scenario
results/<run_id>_report.json     # full aggregated report
results/<run_id>_report.csv      # flat CSV, one row per scenario
```

---

## Resumption contract

The runner guarantees:
- Each `(scenario_id, turn)` pair is written to the checkpoint at most once.
- On resume, the multi-turn message history is reconstructed from
  `raw_prompt` + `raw_response` fields stored in the checkpoint, so the
  model sees its full prior context before the next turn is sent.
- Turns that fail after all retries are skipped (not checkpointed) and
  retried on the next invocation.
- `compute_metrics` and `detect_archetype` are called once per scenario
  after all its turns complete; metrics are identical to a single
  uninterrupted run (verified by `tests/test_resumption.py`).

---

## Adapters

| Adapter | Key from env | Notes |
|---|---|---|
| `OpenAIAdapter` | `OPENAI_API_KEY` | Retries on `RateLimitError`, `APIConnectionError`, `APITimeoutError` |
| `AnthropicAdapter` | `ANTHROPIC_API_KEY` | Same retry policy |
| `MockAdapter` | — | Deterministic; infers STAY/ACT from event keywords in turn-1 message |

Both real adapters retry up to 5 times with exponential backoff (1 s, 2 s, 4 s, 8 s, 16 s).
