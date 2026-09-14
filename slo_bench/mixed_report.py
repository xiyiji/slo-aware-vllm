"""Aggregate results/mixed/<arm>/seed-*/summary.json into results/mixed/REPORT.md."""
import argparse
import json
import statistics
from pathlib import Path


def load(root):
    arms = {}
    for arm_dir in sorted(Path(root).iterdir()):
        seeds = sorted(arm_dir.glob("seed-*/summary.json"))
        if seeds:
            arms[arm_dir.name] = [json.loads(s.read_text()) for s in seeds]
    return arms


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, digits=2, unit=""):
    return "-" if v is None else f"{v:.{digits}f}{unit}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/mixed")
    args = p.parse_args()
    arms = load(args.root)
    order = ["fcfs", "priority", "priority-nochunk", "priority-halfkv"]
    lines = ["# Mixed load: online stream + offline burst, FCFS vs priority scheduling", "",
             "Median over seeds. Online: 64 requests at 2 req/s, 256 in / 128 out, SLO TTFT <= 1 s and E2E <= 10 s. "
             "Offline: 96 requests of 1024 in / 256 out submitted at t = 5 s. Qwen2.5-7B-Instruct bf16 on one RTX 4090, "
             "vLLM 0.29.0, max_num_seqs 128, 4096-token budget, prefix caching off.", "",
             "| arm | seeds | online p50 TTFT | online p99 TTFT | online p99 E2E | online SLO-good | online errors | offline makespan | offline tok/s | offline errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in order + [a for a in arms if a not in order]:
        if name not in arms:
            continue
        runs = arms[name]
        off = [r.get("offline") or {} for r in runs]
        good = med([r["good_requests"] / r["attempted"] * 100 for r in runs])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, len(runs),
            fmt(med([r["p50_ttft_s"] for r in runs]), 3, " s"),
            fmt(med([r["p99_ttft_s"] for r in runs]), 3, " s"),
            fmt(med([r["p99_e2e_s"] for r in runs]), 2, " s"),
            fmt(good, 0, "%"),
            fmt(med([r["errors"] for r in runs]), 0),
            fmt(med([o.get("makespan_s") for o in off]), 1, " s"),
            fmt(med([o.get("output_tokens_s") for o in off]), 0),
            fmt(med([o.get("errors") for o in off]), 0)))
    lines += ["", "Per-seed records: `<arm>/seed-*/requests.jsonl`; server command and package pins: `<arm>/environment.json`."]
    Path(args.root, "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
