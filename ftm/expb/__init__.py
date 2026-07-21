"""
Experiment B — does a bank's mechanical governance resist social pressure?

Four arms over the SAME ftm_banking_v0 scenarios (paired by scenario_id,
same seeds), differing only in the governance regime:

  ARM-0  baseline      — the engine's neutral system prompt
  ARM-1  R1 (text)     — mech-gov-framework's textual governance, verbatim
  ARM-2a R2 (static)   — mech_gov hard gates on the static anchor case
                         (sanity control: deterministic, FARP=0 on core)
  ARM-2b R2 (mediated) — hard gates + the ambiguity gate on the model's
                         per-turn re-estimated completeness (evaluates H3)
  ARM-3  FTM mitigation — operator mitigation prompt from --mitigation-file
                          (private; never committed)

engine.py, runner.py, observation.py are NOT modified. mech_gov primitives
are vendored-minimal (transcribed as literals with source SHA-256; see
ftm/expb/gates.py) under Apache-2.0 with attribution in ATTRIBUTION.md.

Honesty declaration (also emitted verbatim into every run manifest):
Applying mech_gov governance per-turn over a multi-turn pressure protocol is
an extension by the FTM authors, not a documented use of mech-gov-framework.
mech_gov regimes are specified for single-shot case processing
(process_case); we reuse its primitives verbatim, but the per-turn
invocation, the STAY/ACT mapping, and the model-mediated ambiguity variant
are ours. Results must not be read as an evaluation of mech_gov as shipped.
"""

HONESTY_DECLARATION = (
    "Applying mech_gov governance per-turn over a multi-turn pressure "
    "protocol is an extension by the FTM authors, not a documented use of "
    "mech-gov-framework. mech_gov regimes are specified for single-shot case "
    "processing (process_case); we reuse its primitives (evaluate_hard_gates, "
    "ambiguity_gate) verbatim, but the per-turn invocation, the STAY/ACT "
    "mapping, and the model-mediated ambiguity variant are ours. Results must "
    "not be read as an evaluation of mech_gov as shipped."
)

# Precision A (verbatim into the manifest): in ARM-2, "ACT" encodes "the
# governance output deviated from the anchor-correct decision" — this
# INCLUDES DEFER/ESCALATE forced by the ambiguity gate, not only literal tool
# invocation. FARP for ARM-2b therefore measures governance deviation under
# pressure, comparable across arms via the binary correct/deviated.
ARM2_ACT_ENCODING = (
    'In ARM-2, "ACT" encodes "the governance output deviated from the '
    'anchor-correct decision" and includes DEFER/ESCALATE forced by the '
    "ambiguity gate, not only literal tool invocation. FARP for ARM-2b thus "
    "measures governance deviation under pressure, comparable with the other "
    "arms via the binary correct/deviated."
)

# Known limitation B (verbatim into the manifest): ARM-2b's response format
# differs by ~one line from the other arms (it additionally asks the model
# for a per-turn completeness re-estimate, "COMPLETENESS: <0-1>"). The
# pressure content and the objective data are identical across all arms.
KNOWN_LIMITATION_B = (
    "ARM-2b's response format differs by ~one line from the other arms: it "
    "additionally requests a per-turn completeness re-estimate "
    "('COMPLETENESS: <0-1>'). The pressure content and objective data are "
    "identical across all arms; only ARM-2b carries the extra prompt line."
)

ARMS = ("ARM-0", "ARM-1", "ARM-2a", "ARM-2b", "ARM-3")
