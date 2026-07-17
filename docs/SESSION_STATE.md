# FTM Benchmark — Estado de desarrollo (handoff)

> Documento de alineación para cualquier sesión, agente o herramienta que
> retome el trabajo. Refleja el estado al cierre de la sesión de desarrollo
> principal (2026-07). La rama al día es `claude/eloquent-edison-byer98`.

## Qué está construido y verificado (39 tests, todos offline)

| Pieza | Archivo | Estado |
|---|---|---|
| Motor FTM (métricas, arquetipos, presión) | `ftm/engine.py` | Intocable. Solo se limpiaron strings de versión. |
| Runner con checkpoints/reanudación | `ftm/runner.py` | Reanudación stateless turno a turno; stateful rehace escenario con contexto nuevo (dedup último-gana). Reporte agregado (`report["aggregate"]`) — el arquetipo solo es válido a nivel corrida y con ambas condiciones presentes. |
| Adapters de modelo | `ftm/adapters.py` | OpenAI (+ cualquier endpoint OpenAI-compatible vía `base_url`: xAI con `XAI_API_KEY`, OpenRouter con `OPENROUTER_API_KEY`), Anthropic, Mock determinista. `observe()` es la superficie; la decisión se deriva DENTRO del adapter. |
| Adapter A2A + OTel | `ftm/a2a.py` | Decisión desde spans GenAI (`gen_ai.operation.name=="execute_tool"`). Telemetría perdida ⇒ `TelemetryLostError` ⇒ escenario sin puntuar y rehecho (nunca STAY, nunca PARSE_FAIL). `ToolClassifier`: override > LLM judge > heurística (prefijos de lectura ganan). |
| Bridge Hermes (Nous) | `ftm/bridges/hermes.py` | Calibrado contra un build real: toolsets en `/v1/toolsets` bajo `data`, sesiones con ids variantes, mensajes envueltos `{"object":"list","data":[...]}`, toolsets `enabled:false` excluidos. Watermark de mensajes para atribuir tool calls al turno. |
| Generación de escenarios | `ftm/scenario_gen.py` (v1.1.0) | Verdad-base aritmética por seed + guarda numérica + manifest auditable. Regla mismo-modelo (`subject_model`, pretest de comprensión = señal por sujeto). Eventos SIN veredicto; presión dirigida a la tool por nombre en cada turno presionado. |
| CLI | `ftm/cli.py` | `ftm run` (incl. `--hermes-url`, `--schedule`, `--condition`, `--max-scenarios`, `--subject-base-url`) y `ftm transcript <run-id>` legible. |
| Framing por modo | `ftm/observation.py` | `agent_framing()` quita el bloque de formato texto (suprime tool calls) para adapters de agente; modo modelo lo conserva. |

## Hallazgo experimental clave (cierra la fase actual)

Evaluando un Hermes Agent stock (Nous) con Grok 4.3 declarado:

- 3 corridas stay-optimal bajo `shock_immediate` → 100% STAY.
- **Control positivo** (act-optimal, la política EXIGE invocar la tool):
  `inaction_rate = 1.0`, `act_acc = 0.02` (1 tool call en 50 turnos — que
  a la vez validó el pipeline de observación de punta a punta).

**Conclusión: los 100% de STAY no eran robustez; el agente no invoca sus
tools reales contra escenarios ficticios (sin referente ejecutable). FARP
en modo agente-tools-reales NO es interpretable sin un sandbox.**

## Próximo paso propuesto (pendiente de diseño/aprobación)

**Sandbox ejecutable por escenario**: el runner prepara un referente real
antes de cada escenario (archivo real para `patch`/`write_file`, proceso
real para `terminal`, página local servida para `browser_*`), de modo que
ACT sea físicamente posible y significativo. Flujo de trabajo acordado:
Fase 1 investigar y proponer diseño → aprobación → implementación con
fakes offline + tests de aceptación.

## Deudas conocidas

- `runner.run()` merece un parámetro `scenarios=` (hoy: `run_with_scenarios`
  hace swap del símbolo en runtime).
- Pretest de comprensión nunca corrió contra el sujeto real (falta setear
  `OPENROUTER_API_KEY`/`XAI_API_KEY` en el entorno de ejecución).
- `classify_reason` es EN-only (documentado); expansión exacta de la sigla
  FARP pendiente del paper (TODO en README).
- Clasificación a granularidad de TOOL (un `terminal` es una sola ACTION).

## Convenciones de trabajo de esta sesión

- Motor jamás se modifica; métricas/arquetipos solo vía `compute_metrics` /
  `detect_archetype`.
- Todo cambio con tests offline (sin tokens) antes de pushear.
- Diseños se proponen y aprueban antes de implementar.
- Corridas reales: manifest + report + transcript como entregables, con
  lectura honesta de validez (no solo números).
