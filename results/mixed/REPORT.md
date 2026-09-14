# Mixed load: online stream + offline burst, FCFS vs priority scheduling

Median over seeds. Online: 64 requests at 2 req/s, 256 in / 128 out, SLO TTFT <= 1 s and E2E <= 10 s. Offline: 96 requests of 1024 in / 256 out submitted at t = 5 s. Qwen2.5-7B-Instruct bf16 on one RTX 4090, vLLM 0.29.0, max_num_seqs 128, 4096-token budget, prefix caching off.

| arm | seeds | online p50 TTFT | online p99 TTFT | online p99 E2E | online SLO-good | online errors | offline makespan | offline tok/s | offline errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 3 | 0.759 s | 8.678 s | 17.65 s | 44% | 0 | 20.5 s | 1196 | 0 |
| priority | 3 | 0.138 s | 1.060 s | 11.96 s | 81% | 0 | 22.0 s | 1117 | 0 |
| priority-nochunk | 1 | 0.121 s | 0.935 s | 12.09 s | 86% | 0 | 22.4 s | 1096 | 0 |
| priority-halfkv | 1 | 0.157 s | 1.063 s | 7.33 s | 95% | 0 | 27.0 s | 909 | 0 |

Per-seed records: `<arm>/seed-*/requests.jsonl`; server command and package pins: `<arm>/environment.json`.
