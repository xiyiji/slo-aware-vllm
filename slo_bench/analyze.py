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
    labels = [r["run"].replace("formal-", "") for r in plotted]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric, title in zip(axes, ["output_tokens_s", "goodput_rps", "p95_ttft_s"],
                                 ["Exact output tokens / s", "SLO goodput (requests / s)", "p95 TTFT (seconds)"]):
        ax.bar(range(len(plotted)), [r[metric] for r in plotted], color=["#64748b" if r["seqs"] == 1 else "#2563eb" for r in plotted])
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTX 4090 · Qwen2.5-7B · fixed 256 / 128 token workload")
    fig.savefig(root / "comparison.png", dpi=160)
    lines = ["# Experiment results", "", "Each row is a completed run; failures remain in the denominator. See raw JSONL and manifests.", "",
             "| Run | Seqs | Budget | Success / attempted | Tokens/s | Goodput | p95 TTFT s | p95 E2E s | GPU mean % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        gpu = f'{r["gpu_mean_pct"]:.1f}' if r["gpu_mean_pct"] is not None else "unavailable"
        lines.append(f'| {r["run"]} | {r["seqs"]} | {r["token_budget"]} | {r["succeeded"]}/{r["attempted"]} | {r["output_tokens_s"]:.2f} | {r["goodput_rps"]:.3f} | {r["p95_ttft_s"]:.3f} | {r["p95_e2e_s"]:.3f} | {gpu} |')
    lines += ["", "![Measured comparison](comparison.png)", "", "Percentiles on small samples are descriptive. TPOT is a client-observed per-request average, not an inter-token histogram. Missing GPU/KV metrics are not zero."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"Validated and plotted {len(rows)} runs")


if __name__ == "__main__":
    main()
