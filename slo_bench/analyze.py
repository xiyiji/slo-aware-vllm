"""Generate plots and a tabular evidence report from raw run artifacts."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_counter(path, name):
    values = []
    for line in path.read_text().splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/seed-*/summary.json")):
        summary = json.loads(path.read_text())
        env = json.loads((path.parent.parent / "environment.json").read_text())
        command = env["command"]
        row = {"run": str(path.parent.relative_to(root)),
               "seqs": int(command[command.index("--max-num-seqs")+1]),
               "token_budget": int(command[command.index("--max-num-batched-tokens")+1]), **summary}
        records = [json.loads(line) for line in (path.parent / "requests.jsonl").read_text().splitlines()]
        if len(records) != summary["attempted"]:
            raise ValueError(f"Record count mismatch: {path}")
        telemetry = path.parent.parent / "telemetry.jsonl"
        gpu, kv = [], []
        if telemetry.exists() and "measured_start_unix" in summary:
            for line in telemetry.read_text().splitlines():
                sample = json.loads(line)
                if not summary["measured_start_unix"] <= sample["unix"] <= summary["measured_end_unix"]:
                    continue
                if sample.get("gpu"):
                    try:
                        gpu.append(float(sample["gpu"].split(",")[0]))
                    except ValueError:
                        pass
                for metric in sample.get("metrics", []):
                    if re.match(r"vllm:(kv_cache_usage_perc|gpu_cache_usage_perc)[{ ]", metric):
                        kv.append(float(metric.rsplit(" ", 1)[1]))
        row.update(gpu_samples=len(gpu), gpu_mean_pct=float(np.mean(gpu)) if gpu else None,
                   gpu_peak_pct=max(gpu) if gpu else None, kv_peak_fraction=max(kv) if kv else None)
        before = read_counter(path.parent / "metrics-before.prom", "vllm:num_preemptions_total")
        after = read_counter(path.parent / "metrics-after.prom", "vllm:num_preemptions_total")
        row["preemptions"] = after-before if before is not None and after is not None else None
        rows.append(row)
    if not rows:
        raise ValueError("No completed runs found")
    (root / "all-runs.json").write_text(json.dumps(rows, indent=2))
    with (root / "all-runs.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    formal = [r for r in rows if "formal" in r["run"]]
    plotted = formal or rows
    groups = {}
    for row in plotted:
        groups.setdefault(row["run"].split("/")[0], []).append(row)
    labels = [name.replace("formal-", "") for name in groups]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric, title in zip(axes, ["output_tokens_s", "goodput_rps", "p95_ttft_s"],
                                 ["Exact output tokens / s", "SLO goodput (requests / s)", "p95 TTFT (seconds)"]):
        values = [[r[metric] for r in group] for group in groups.values()]
        ax.bar(range(len(groups)), [np.mean(v) for v in values], color=["#64748b", "#2563eb", "#8b5cf6"][:len(groups)] if len(groups) <= 3 else "#2563eb")
        for i, samples in enumerate(values):
            ax.scatter([i]*len(samples), samples, color="#111827", s=15, zorder=3)
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(title)
        if metric == "p95_ttft_s":
            ax.set_yscale("log")
            ax.axhline(1, color="#dc2626", linestyle="--", linewidth=1, label="TTFT SLO")
            ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTX 4090 · Qwen2.5-7B · fixed 256 / 128 token workload")
    fig.savefig(root / "comparison.png", dpi=160)
    lines = ["# Experiment results", "", "Each row is a completed run; failures remain in the denominator. See raw JSONL and manifests.", "",
             "| Run | Seqs | Budget | Success / attempted | Tokens/s | Goodput | p95 TTFT s | p95 E2E s | GPU mean % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        gpu = f'{r["gpu_mean_pct"]:.1f}' if r["gpu_mean_pct"] is not None else "unavailable"
        lines.append(f'| {r["run"]} | {r["seqs"]} | {r["token_budget"]} | {r["succeeded"]}/{r["attempted"]} | {r["output_tokens_s"]:.2f} | {r["goodput_rps"]:.3f} | {r["p95_ttft_s"]:.3f} | {r["p95_e2e_s"]:.3f} | {gpu} |')
    lines += ["", "![Measured comparison](comparison.png)", "", "Percentiles on small samples are descriptive. TPOT is a client-observed per-request average, not an inter-token histogram. Missing GPU/KV metrics are not zero."]
    if formal:
        aggregates = {name: {metric: {"mean":float(np.mean([r[metric] for r in group])),
                                     "min":min(r[metric] for r in group), "max":max(r[metric] for r in group)}
                             for metric in ("output_tokens_s", "goodput_rps", "p95_ttft_s", "p95_e2e_s")}
                      for name, group in groups.items()}
        (root / "formal-aggregates.json").write_text(json.dumps(aggregates, indent=2))
        lines += ["", "## Formal repeat means", "", "Bars show means; dots show individual repeats. Grouped execution order can confound thermal/time effects. Three repeats do not establish production tail guarantees.", ""]
        for name, metrics in aggregates.items():
            lines.append(f'- {name}: {metrics["output_tokens_s"]["mean"]:.2f} tokens/s; {metrics["goodput_rps"]["mean"]:.3f} good requests/s; mean per-run p95 TTFT {metrics["p95_ttft_s"]["mean"]:.3f}s.')
        selected = aggregates.get("formal-selected")
        if selected:
            for reference in ("formal-baseline", "formal-reference128"):
                if reference in aggregates:
                    base = aggregates[reference]["output_tokens_s"]["mean"]
                    delta = (selected["output_tokens_s"]["mean"]/base-1)*100 if base else None
                    lines.append(f'- Selected output-throughput change versus {reference}: {delta:.2f}%.' if delta is not None else f'- {reference} has zero throughput; percentage undefined.')
        lines += ["", "The sequence-limit-1 baseline is deliberately serial. Its improvement is not an improvement over the original 128-sequence integration. Inspect the 128 reference separately before making any incremental tuning claim."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"Validated and plotted {len(rows)} runs")


if __name__ == "__main__":
    main()
