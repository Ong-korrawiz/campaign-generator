#!/usr/bin/env python3
"""Render the judge JSON and training manifest as a concise Markdown report."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

report_path, markdown_path, training_path, base_path, tuned_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
training = json.loads(training_path.read_text(encoding="utf-8"))
metrics = report["metrics"]
gate = report["promotion_gate"]
overall = metrics["pairwise"]["overall"]
promotion = metrics["promotion"]
shape = metrics["automatic_shape"]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{value:+.1f} pp"


def runtime(key: str) -> str:
    values = [row["model_runtime"][key] for row in report["results"] if row["model_runtime"][key] is not None]
    if not values:
        return "n/a"
    return f"mean {statistics.mean(values):.1f}s; p95 {sorted(values)[int(0.95 * len(values)) - 1]:.1f}s"


rows = [
    "# Baseline vs LoRA benchmark",
    "",
    f"- Run: `{training['run_id']}`",
    f"- Dataset: `{training['dataset_uri']}`; {report['sample_count']} matched synthetic holdout briefs",
    f"- Base model: `{training['model_id']}` revision `{training['model_revision']}`",
    f"- Judge: `{report['judge']['model']}`; {report['judge_calls']} paired-order calls",
    "- Dataset semantic review: pending; this is an exploratory schema-only experiment.",
    "",
    "## Pairwise result",
    "",
    f"- Overall: base {overall['base_wins']} wins, tuned {overall['tuned_wins']} wins, {overall['ties']} ties ({overall['resolved_count']} resolved briefs).",
    f"- Tuned net win: **{pp(promotion['net_win_percentage_points'])}**; 95% bootstrap CI **[{promotion['bootstrap_95_ci'][0]:.1f}, {promotion['bootstrap_95_ci'][1]:.1f}] pp**.",
    f"- Order disagreement: {pct(overall['order_disagreement_rate'])}.",
    "",
    "| Criterion | Base wins | Tuned wins | Ties | Tuned net |",
    "|---|---:|---:|---:|---:|",
]
for criterion, values in metrics["pairwise"].items():
    net = metrics["rubric_net_percentage_points"][criterion]
    rows.append(
        f"| {criterion.replace('_', ' ')} | {values['base_wins']} | "
        f"{values['tuned_wins']} | {values['ties']} | {pp(net)} |"
    )

rows += [
    "",
    "## Output quality and runtime",
    "",
    "| Check | Baseline | LoRA |",
    "|---|---:|---:|",
    f"| Full format pass | {pct(metrics['format_pass_rate']['base'])} | {pct(metrics['format_pass_rate']['tuned'])} |",
]
for key, label in (
    ("campaign_descriptions_present", "Description present"),
    ("campaign_descriptions_2_to_3_sentences", "Description has 2–3 sentences"),
    ("campaign_descriptions_distinct", "Descriptions distinct (exact check)"),
    ("proposed_kpis_present", "Proposed KPIs present"),
):
    rows.append(f"| {label} | {pct(shape['base'][key])} | {pct(shape['tuned'][key])} |")

failure_counts = {}
for model, path in (("base", base_path), ("tuned", tuned_path)):
    with path.open(encoding="utf-8") as stream:
        failure_counts[model] = Counter(
            row.get("output", {}).get("invalid_output", "")
            for row in map(json.loads, stream)
            if row.get("validation_errors")
        )
for reason in sorted(set(failure_counts["base"]) | set(failure_counts["tuned"])):
    rows.append(
        f"| Validation failure: `{reason}` | "
        f"{failure_counts['base'][reason]}/{report['sample_count']} | "
        f"{failure_counts['tuned'][reason]}/{report['sample_count']} |"
    )
if not any(failure_counts.values()):
    rows.append("| Validation failures | 0/36 | 0/36 |")

for model_key, runtime_key in (
    ("base", "base_latency_seconds"),
    ("tuned", "tuned_latency_seconds"),
):
    token_key = f"{model_key}_output_tokens"
    token_mean = statistics.mean(row["model_runtime"][token_key] for row in report["results"])
    rows.append(
        f"- {model_key.title()} inference: {runtime(runtime_key)}; "
        f"mean output {token_mean:.0f} tokens per brief."
    )

rows += [
    f"- Training: {training['training']['epochs']} epochs; LoRA r={training['lora']['r']}, "
    f"alpha={training['lora']['alpha']}; train loss {training['train_metrics']['train_loss']:.3f}; "
    f"validation loss {training['validation_metrics']['eval_loss']:.3f}.",
    "",
    "## Promotion gate",
    "",
    f"- Net win ≥10 pp: **{gate['net_win_at_least_10_points']}**",
    f"- 95% CI excludes zero: **{gate['bootstrap_ci_excludes_zero']}**",
    f"- Format pass ≥98%: **{gate['format_pass_at_least_98_percent']}**",
    f"- Critical rubric not down >2 pp: **{gate['critical_rubric_not_down_more_than_2_points']}**",
    f"- Human review: **{gate['human_review']}**; ready to promote: **{gate['ready_to_promote']}**",
    "",
    "**Conclusion:** retain the baseline. The LoRA adapter failed the output-format gate, "
    "so this schema-only experiment is not eligible for promotion. The dataset was not "
    "semantically reviewed; the result is exploratory.",
    "",
    f"Full pairwise evidence, per-brief outputs, hashes, and token usage: `{report_path.name}`.",
]
markdown_path.parent.mkdir(parents=True, exist_ok=True)
markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
