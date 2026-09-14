# Priority Scheduling for Mixed Online/Offline LLM Serving on One GPU: A Measurement Study

**Mengyun Wang** · September 2026 · github.com/xiyiji/tempo

## Abstract

Production LLM clusters serve two kinds of traffic on the same GPUs: interactive requests
with a time-to-first-token (TTFT) objective, and batch jobs that submit hundreds of long
requests at once and care only about throughput. vLLM's default first-come-first-served
scheduler admits a batch burst ahead of every interactive request that arrives behind it. We
measure, on a single RTX 4090 serving Qwen2.5-7B-Instruct, how much of the interactive SLO
vLLM's built-in priority scheduling recovers and what the batch job pays for it. Under a
paced online stream (2 req/s, TTFT ≤ 1 s, E2E ≤ 10 s) with a 96-request offline burst,
FCFS meets the SLO for 44% of online requests with a p99 TTFT of 8.7 s; priority scheduling
raises SLO attainment to 82% and brings p99 TTFT to 1.06 s, while the batch job finishes 8%
later. The remaining misses are end-to-end, not TTFT, misses: priority ordering protects
admission but not decode-phase interference, which raises online time-per-output-token from
20 ms to 35 ms while the burst is resident. Disabling chunked prefill does not hurt and
slightly helps online TTFT in this configuration (p99 0.94 s, 0 TTFT misses). All numbers
come from paired seeds on real hardware with exact server-side token accounting; raw
per-request records are in the repository.

## 1. Problem

A GPU pool rarely serves one workload. Chat and copilot traffic arrives continuously and is
judged on TTFT; evaluation suites, offline scoring and synthetic-data generation arrive as
bursts of hundreds of long requests and are judged on completion time. The scheduler decides
who waits. vLLM's default policy is FCFS over the waiting queue: once a burst is queued, every
online request behind it waits for the burst's prefill to drain through the token budget.

The operational question is not whether this happens — it obviously does — but by how much,
and what the alternatives cost. Prior measurements of this trade-off in portfolio projects
are typically produced by CPU simulators of the scheduler; a simulator cannot reproduce the
GPU's prefill/decode interference, CUDA-graph behaviour, or KV-block pressure that determine
the real numbers. This study measures the trade-off directly.

## 2. Method

**System.** One NVIDIA RTX 4090 (24 GB), Qwen2.5-7B-Instruct in bfloat16, vLLM 0.29.0,
`max_model_len 4096`, `max_num_seqs 128`, `max_num_batched_tokens 4096`, prefix caching off
(prompts are random token ids, so caching could only distort the comparison),
`gpu_memory_utilization 0.9` unless stated. Each arm runs in a fresh server process bound to
loopback; the harness records the exact server command, package pins and model revision
(`environment.json`) and samples GPU and scheduler telemetry once per second.

**Workload.** Two request classes share the server:

| class | requests | arrival | input / output tokens | objective |
|---|---:|---|---:|---|
| online | 64 | open-loop Poisson at 2 req/s over ~32 s | 256 / 128 | TTFT ≤ 1 s and E2E ≤ 10 s |
| offline | 96 | all submitted at t = 5 s | 1024 / 256 | finish as early as possible |

The offline burst therefore starts after the online stream has warmed up and finishes
before it ends, so roughly 50 of the 64 online requests are in flight while the burst is
resident. Output lengths are exact (`ignore_eos`, `max_tokens`), and the harness rejects any
response whose server-reported `completion_tokens` differs from the requested count; TTFT is
taken at the first content token, never at a role or empty delta.

**Arms.**

| arm | scheduler | notes |
|---|---|---|
| fcfs | `--scheduling-policy fcfs` (vLLM default) | no priorities sent |
| priority | `--scheduling-policy priority` | online requests `priority=0`, offline `priority=10` (lower wins) |
| priority-nochunk | priority | `--no-enable-chunked-prefill` |
| priority-halfkv | priority | `gpu_memory_utilization 0.77`: the weights take 14.3 GiB, so this leaves roughly half the KV blocks of the 0.9 setting |

fcfs and priority use three paired seeds (101, 102, 103); the seed fixes both the prompts
and the arrival times, so the two arms see identical request sequences. The two ablations
use seed 101.

**Metrics.** Online: p50/p99 TTFT, p99 E2E, the fraction of requests meeting both SLOs
("SLO-good"), and time-per-output-token (TPOT). Offline: makespan from burst submission to
the last completion, and output tokens per second over that window. Medians over seeds are
reported; per-seed values are in `results/mixed/<arm>/seed-*/summary.json`.

## 3. Results

Medians over seeds (fcfs and priority: 3 paired seeds; ablations: 1).

| arm | seeds | online p50 TTFT | online p99 TTFT | online p99 E2E | online SLO-good | online errors | offline makespan | offline tok/s | offline errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 3 | 0.759 s | 8.678 s | 17.65 s | 44% | 0 | 20.5 s | 1196 | 0 |
| priority | 3 | 0.138 s | 1.060 s | 11.96 s | 81% | 0 | 22.0 s | 1117 | 0 |
| priority-nochunk | 1 | 0.121 s | 0.935 s | 12.09 s | 86% | 0 | 22.4 s | 1096 | 0 |
| priority-halfkv | 1 | 0.157 s | 1.063 s | 7.33 s | 95% | 0 | 27.0 s | 909 | 0 |

**Admission is fixed; decode interference is not.** Under FCFS, 31 of the 64 online
requests in seed 101 miss the TTFT objective and their p99 TTFT during the burst is 8.5 s: the
online request simply waits for the burst's prefill. Priority scheduling reduces TTFT misses
to 6 (p99 1.05 s). The misses that remain are end-to-end misses with an acceptable TTFT:
online TPOT rises from 20 ms with the GPU to itself to 35 ms while the burst is resident,
because once admitted, online and offline sequences share every decode step regardless of
priority. A 128-token response at 35 ms/token is 4.5 s of decode on top of queueing, and a
handful of requests that arrive at the worst moment exceed the 10 s E2E budget. Priority
ordering is an admission-control mechanism; protecting decode-phase latency would need
either a per-step token budget reserved for the online class or a cap on offline
concurrency.

**What the batch job pays.** The offline burst finishes in 22.0 s under priority against
20.5 s under FCFS (+7%), and its throughput drops from about 1200 to about 1110 output
tokens/s (−8%). Both are the cost of repeatedly yielding the prefill budget to newly arrived
online requests; the batch job still completes with zero errors.

**Chunked prefill.** Disabling chunked prefill under priority scheduling did not degrade
online TTFT; it removed the remaining TTFT misses (0 of 64, p99 0.94 s) at the price of two
more E2E misses. With a 4096-token budget and 1024-token offline prompts, each offline
prefill fits in one step either way, so chunking mostly changes how online prefills
interleave with in-flight decodes rather than how long they wait. This is one configuration
on one GPU; the general claim that chunked prefill trades TTFT for ITL stability
(Sarathi-Serve) is not contradicted, but it did not show up here.

**KV capacity.** Halving the KV pool (0.77 utilisation) under priority scheduling produced the best online
numbers of the study — 95% SLO-good and p99 E2E 7.3 s against 12.0 s at full KV — and the
worst offline numbers: makespan 27.0 s (+32% over priority at full KV) and 909 output
tokens/s. With fewer blocks the scheduler cannot keep 96 offline sequences resident at once,
so fewer of them share each decode step with the online requests, which is exactly the
decode-phase interference that priority ordering alone does not remove. The batch job pays
for it in wall-clock time. This is a single-seed observation, but the direction is the one
the interference explanation predicts, and it suggests that capping offline concurrency is
a cheaper lever than shrinking KV for the same effect.

## 4. Threats to validity

- One GPU, one model, one synthetic workload. The offline burst size (96 × 1024 tokens) was
  chosen to overlap most of the online stream; a larger burst would lengthen the FCFS queue
  and widen the gap, a smaller one would narrow it.
- Poisson arrivals rather than a production trace. The Azure LLM inference traces and
  BurstGPT are the natural next step.
- Three seeds per main arm, one per ablation. The main-arm results are separated by far more
  than their seed-to-seed spread (SLO-good 27–29 vs 52–53 of 64), the ablations are
  single-run observations.
- Formal repetitions were run back to back on a rented pod; thermal drift is uncontrolled.
- The vLLM version is pinned (0.29.0); the priority policy's behaviour has changed across
  releases and the harness refuses to run on another version.

## 5. Relation to prior work

Sarathi-Serve introduced chunked prefill to bound decode stalls; Llumnix schedules across
instances; Andes and similar work optimise a quality-of-experience objective rather than
TTFT alone. vLLM's priority policy is the simplest deadline-agnostic mechanism available in
the engine. This study does not propose a new policy; it quantifies what the built-in one
buys on real hardware, which is the number an operator needs before deciding between
priority scheduling, a separate offline pool, or admission control at the gateway.

## 6. Future work

1. A deadline-aware policy (EDF over the online class with an offline token-budget cap),
   implemented against vLLM's scheduler interface and measured with the same harness.
2. Replace Poisson arrivals with a public production trace.
3. A second model size and a second GPU generation.
4. Per-step budget reservation for the online class, to attack the decode-phase misses
   that priority ordering leaves behind.

## Reproduce

```bash
pip install -e .              # on a machine with one RTX 4090 and vLLM 0.29.0
python -m slo_bench.mixed     # ~25 minutes; writes results/mixed/REPORT.md
```
