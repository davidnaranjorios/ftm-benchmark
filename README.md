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
- `ftm/observation.py` — `TurnObservation`: normalized per-turn output; the
  decision is derived inside the adapter, keeping the runner model-vs-agent agnostic.
- `ftm/a2a.py` — `A2AAgentAdapter` (evaluates an AGENT via its A2A endpoint,
  deriving STAY/ACT from real tool calls observed as OTel spans), plus
  `InProcessSpanCollector` and `ToolClassifier`.
- `ftm/runner.py` — resilient benchmark runner with checkpoint/resumption and CLI.
- `tests/test_resumption.py` — acceptance test for the resumption contract.
- `tests/fakes.py` + `tests/test_a2a_adapter.py` — in-process fake A2A agent +
  fake OTel emitter and the A2A acceptance tests (no network, no tokens).

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

---

## A2A agent evaluation

`A2AAgentAdapter` (in `ftm/a2a.py`) evaluates an **agent** rather than a text
model: each FTM turn is sent as an A2A message within a per-scenario
`contextId`, and the decision is derived from the agent's **real tool calls**,
observed as OpenTelemetry spans following the GenAI semantic conventions
(`gen_ai.operation.name == "execute_tool"`, `gen_ai.tool.name` — never
span-name substrings).

Key behaviors:

- **Async telemetry**: spans are read by polling the in-process collector with
  a per-trace timeout. Zero spans in the window → **UNKNOWN** (recorded as
  `PARSE_FAIL` with an `UNKNOWN:` reason so metrics exclude the turn) — never
  STAY by default. STAY requires at least one span proving the pipeline is
  alive.
- **Stateful resumption**: A2A agents hold history server-side, so an
  interrupted scenario is **redone from turn 1 with a fresh contextId** rather
  than continued mid-way (`stateful = True`; the checkpoint keeps the last
  record per turn, so redone turns supersede stale partial ones). Stateless
  model adapters still resume turn-by-turn.
- **Tool classification**: tools are classified ACTION/READ lazily as they
  first appear in spans (no Agent-Card pre-enumeration). Priority: per-tool
  operator override > LLM judge (language-agnostic) > English verb heuristic.
  The verdict table is exportable via `ToolClassifier.export()` for audit.

Everything is testable offline: `tests/fakes.py` provides an in-process fake
A2A agent whose spans export synchronously (SimpleSpanProcessor semantics),
with configurable late-span delay and span-drop modes.
