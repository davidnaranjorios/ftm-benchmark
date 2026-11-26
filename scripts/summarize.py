"""Summarize FTM Inspect logs into a table and the README chart.

    python scripts/summarize.py LOG_DIR [--out docs/results]

Reads every successful `ftm` eval log in LOG_DIR (latest run wins per
model × data_anchoring), writes summary.csv / summary.md and two PNGs
(light and dark) of FARP_strict per model, baseline vs Data Anchoring,
with 95% scenario-level bootstrap CIs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

METRICS = ["farp_strict", "farp_ci_lower", "farp_ci_upper", "farp_under_pressure",
           "farp_control", "inaction_rate", "breaking_turn", "parse_fail_rate",
           "accuracy", "crs"]

# Validated (dataviz validator) categorical slots 1-2, light and dark.
THEMES = {
    "light": dict(surface="#ffffff", text="#191919", text2="#666666", grid="#e7e7e7",
                  axis="#c2c2c2", base="#257de0", anchor="#f47230"),
    "dark": dict(surface="#0d1117", text="#efefef", text2="#9c9c9c", grid="#30363d",
                 axis="#474747", base="#338ced", anchor="#e76624"),
}


NAMES = {
    "openai/gpt-5.2": "GPT-5.2",
    "meta-llama/llama-4-scout": "Llama 4 Scout",
    "qwen/qwen3-14b": "Qwen3 14B",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 Super",
}


def load(log_dir: str) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info, header_only=True)
        if log.eval.task.split("/")[-1] != "ftm" or log.status != "success" or not log.results:
            continue
        args = log.eval.task_args or {}
        anchoring = bool(args.get("data_anchoring", False))
        model = log.eval.model.removeprefix("openrouter/")
        m = {name: mt.value for s in log.results.scores for name, mt in s.metrics.items()}
        row = {"model": model, "data_anchoring": anchoring,
               "tier": args.get("tier", "extended"),
               "n_scenarios": log.results.completed_samples,
               "created": log.eval.created, "log": Path(info.name).name,
               **{k: m.get(k) for k in METRICS}}
        key = (model, anchoring)
        if key not in rows or row["created"] > rows[key]["created"]:
            rows[key] = row
    return sorted(rows.values(), key=lambda r: (r["model"], r["data_anchoring"]))


def write_tables(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = ["model", "data_anchoring", "tier", "n_scenarios", *METRICS, "log"]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def pct(v):
        return "—" if v is None or v != v else f"{v:.0%}"

    lines = [
        "| Model | Data Anchoring | FARP_strict [95% CI] | FARP pressure | FARP control | Inaction rate | Breaking turn | Parse fail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bt = r["breaking_turn"]
        lines.append(
            f"| {r['model']} | {'yes' if r['data_anchoring'] else 'no'} | "
            f"{pct(r['farp_strict'])} [{pct(r['farp_ci_lower'])}, {pct(r['farp_ci_upper'])}] | "
            f"{pct(r['farp_under_pressure'])} | {pct(r['farp_control'])} | "
            f"{pct(r['inaction_rate'])} | {'—' if bt is None or bt != bt else f'{bt:.1f}'} | "
            f"{pct(r['parse_fail_rate'])} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def plot(rows: list[dict], path: Path, theme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    models = sorted({r["model"] for r in rows},
                    key=lambda m: -max(r["farp_strict"] for r in rows if r["model"] == m))
    by = {(r["model"], r["data_anchoring"]): r for r in rows}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.6, 0.95 * len(models) + 1.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    h = 0.36
    for i, model in enumerate(models):
        for j, (anch, color, label) in enumerate([(False, t["base"], "Baseline prompt"),
                                                  (True, t["anchor"], "+ Data Anchoring")]):
            r = by.get((model, anch))
            if r is None:
                continue
            y = i + (j - 0.5) * (h + 0.04)
            v = r["farp_strict"]
            ax.barh(y, v, height=h, color=color, label=label if i == 0 else None, zorder=2)
            lo, hi = r["farp_ci_lower"], r["farp_ci_upper"]
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color=t["text2"], lw=1.2, zorder=3,
                        solid_capstyle="butt")
            ax.text(max(v, hi or v) + 0.015, y, f"{v:.0%}", va="center", ha="left",
                    color=t["text"], fontsize=10)

    ax.set_yticks(range(len(models)), [NAMES.get(m, m) for m in models], color=t["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"], color=t["text2"])
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])
    ax.set_xlabel("FARP_strict: share of STAY-optimal scenarios where the model switched to ACT",
                  color=t["text2"], fontsize=9.5)
    n = rows[0]["n_scenarios"]
    n_stay = n // 2
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
                    fontsize=10, handlelength=1.2, borderaxespad=0.3)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.text(0.015, 0.965, "Does the model abandon a correct decision under social pressure?",
             ha="left", va="top", color=t["text"], fontsize=13, fontweight="bold")
    fig.text(0.015, 0.915,
             f"FTM, {n} scenarios ({n_stay} STAY-optimal) × 10 turns per run, temperature 0, via OpenRouter.\n"
             "Lower is better. Whisker: 95% scenario-level bootstrap CI.",
             ha="left", va="top", color=t["text2"], fontsize=9.5)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--out", default="docs/results")
    a = p.parse_args()
    rows = load(a.log_dir)
    if not rows:
        raise SystemExit(f"no successful ftm logs in {a.log_dir}")
    out = Path(a.out)
    write_tables(rows, out)
    plot(rows, out / "farp_by_model.png", "light")
    plot(rows, out / "farp_by_model_dark.png", "dark")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
