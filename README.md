# SLO-aware vLLM Scheduling Optimization

[![Measurement contracts](https://github.com/xiyiji/slo-aware-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/xiyiji/slo-aware-vllm/actions/workflows/ci.yml)

Controlled single-RTX-4090 experiments: select vLLM scheduler settings using latency-constrained goodput, not just peak tokens/s.

**Status: 16 completed runs collected and locally validated, including nine formal runs. Public-service restoration and demo delivery are still in progress.**

## What the measurements showed

The screening-selected setting (8 sequences, 2048-token budget) did **not** improve on the existing-style 128-sequence / 4096-token reference. Across three paired-seed runs, mean output throughput was effectively equal (265.41 versus 265.38 tokens/s), while selected mean goodput was lower (1.999 versus 2.073 requests/s). In seed103, selected p95 TTFT reached1.464s versus0.076s for the reference. Retain the128 reference for this tested workload; the screening choice is not a validated upgrade.

The deliberately serial sequence-limit1 reference averaged40.47 tokens/s with270 timeouts across384 attempts. This is evidence of overload under the offered workload, not a fair basis for claiming a555% improvement over the original batched application. Results describe successful outputs during a deadline-limited run, not unrestricted engine capacity.

Read the [per-run tables and comparison plot](results/REPORT.md), [formal aggregates](results/formal-aggregates.json), and [frozen screening selection](results/selection.json). Raw request records, telemetry and manifests accompany each run. These scheduler experiments do not fulfill the separate InferenceGateway SPEC's HF-versus-vLLM concurrency1/4/16 comparison.

- [完整中文设计文档](docs/DESIGN.zh-CN.md)
- [English design and measurement contract](docs/DESIGN.md)
- [Operations, recovery and evidence boundaries](docs/OPERATIONS.md)
- Exact server token counts; missing usage fails the run contract.
- Seeded open-loop arrivals, per-request JSONL, raw engine and GPU telemetry.
- Serial-sequence baseline vs engine continuous batching; no gateway cache confound.

## CPU validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## GPU experiment

Use a dedicated RTX 4090 with vLLM **0.29.0** installed. The command starts an isolated server on loopback port 18001 and stops only its own process group. First free the GPU by gracefully stopping any existing model deployment. Do not run alongside a second 7B engine.

```bash
pip install -e .
python -m slo_bench.experiment --seqs 1 --tokens 4096 --requests 64 --output results/screen-s1
python -m slo_bench.experiment --seqs 16 --tokens 4096 --requests 64 --output results/screen-s16
# After screening, freeze the candidate and repeat both arms with paired seeds:
python -m slo_bench.experiment --seqs 1 --seeds 101,102,103 --requests 128 --output results/formal-baseline
```

For a fresh end-to-end study, use `python -m slo_bench.sweep` followed by
`python -m slo_bench.finalize`. The latter checks screening completeness, evaluates
two additional token budgets, freezes the selection, and runs three paired-seed
trials each for the serial baseline, selected configuration and sequence-limit128
reference. It does not stop or provision a cloud Pod. Allow for repeated model
loading, and use a persistent terminal/process supervisor.

Same 256 input / 128 exact output tokens, 2 requests/s, SLO TTFT ≤1s and E2E ≤10s. `max_num_seqs=1` is not a claim that all GPU batching/optimizations are disabled. Results apply only to the recorded synthetic workload and runtime.

Formal warmup is three rounds of eight concurrent requests. Initial exploratory
screens used three sequential warmups; see the recorded method changes. Formal
repetitions are grouped by configuration to limit rented-GPU startup cost, so
time/thermal drift is a limitation, not a controlled factor.

The 128-sequence reference matters: an improvement over a deliberately serial
baseline does **not** establish an improvement over an existing batched service.

The separate [serving console](https://github.com/xiyiji/llm-serving-platform) and [Ray/vLLM integration](https://github.com/xiyiji/InferenceGateway) demonstrate application integration; they are not included in this local engine benchmark's timing.
