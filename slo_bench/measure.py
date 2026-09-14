"""Open-loop completions benchmark. Never infer token counts from SSE chunks."""
import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np


class StreamMeasurement:
    def __init__(self):
        self.first = None
        self.last = None
        self.usage = None
        self.done = False
        self.finish = None

    def consume(self, payload, elapsed):
        if payload == "[DONE]":
            self.done = True
            return
        item = json.loads(payload)
        if item.get("error"):
            raise ValueError(str(item["error"]))
        if item.get("usage") is not None:
            self.usage = item["usage"]
        for choice in item.get("choices", []):
            text = choice.get("text") or choice.get("delta", {}).get("content")
            if text:
                if self.first is None:
                    self.first = elapsed
                self.last = elapsed
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]

    def result(self, expected):
        if not self.done or not self.usage or self.first is None or not self.finish:
            raise ValueError("Incomplete stream: content, final usage, finish reason and DONE required")
        count = self.usage.get("completion_tokens")
        if not isinstance(count, int) or count != expected:
            raise ValueError(f"Expected {expected} exact output tokens; received {count}")
        return {"prompt_tokens": self.usage["prompt_tokens"], "completion_tokens": count,
                "ttft_s": self.first, "tpot_s": (self.last - self.first) / (count - 1) if count > 1 else None}


def summarize_offline(records, burst_start, wall_s):
    """The batch job's view: did it finish, how long did it take, what did it get."""
    offline = [r for r in records if r.get("class") == "offline"]
    if not offline:
        return None
    valid = [r for r in offline if r["ok"]]
    finish = [r["submitted_at_s"] + r["e2e_s"] for r in valid]
    makespan = (max(finish) - burst_start) if finish else None
    return {"attempted": len(offline), "succeeded": len(valid), "errors": len(offline) - len(valid),
            "burst_start_s": burst_start, "makespan_s": makespan,
            "output_tokens_s": (sum(r["completion_tokens"] for r in valid) / makespan) if makespan else None,
            "p50_e2e_s": float(np.percentile([r["e2e_s"] for r in valid], 50)) if valid else None,
            "p99_e2e_s": float(np.percentile([r["e2e_s"] for r in valid], 99)) if valid else None}


def summarize(records, wall_s, ttft_slo, e2e_slo):
    records = [r for r in records if r.get("class", "online") == "online"]
    valid = [r for r in records if r["ok"]]
    good = [r for r in valid if r["ttft_s"] <= ttft_slo and r["e2e_s"] <= e2e_slo]
    result = {"attempted": len(records), "succeeded": len(valid), "errors": len(records)-len(valid),
              "wall_s": wall_s, "output_tokens_s": sum(r["completion_tokens"] for r in valid)/wall_s,
              "requests_s": len(valid)/wall_s, "goodput_rps": len(good)/wall_s,
              "good_requests": len(good), "ttft_slo_s": ttft_slo, "e2e_slo_s": e2e_slo}
    for metric in ("ttft_s", "tpot_s", "e2e_s", "dispatch_lateness_s"):
        values = [r[metric] for r in valid if r.get(metric) is not None]
        for p in (50, 95, 99):
            result[f"p{p}_{metric}"] = float(np.percentile(values, p)) if values else None
    return result


async def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # Printable token IDs from the tokenizer vocabulary, not natural-language quality prompts.
    prompts = [[rng.randrange(100, 10000) for _ in range(args.input_tokens)] for _ in range(args.requests)]
    digest = hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
    manifest = vars(args) | {"workload_sha256": digest, "started_unix": time.time(), "clock": "perf_counter",
                             "measurement_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    records = []
    t0 = time.perf_counter()
    offline_rng = random.Random(args.seed + 200000)
    offline_prompts = [[offline_rng.randrange(100, 10000) for _ in range(args.offline_input_tokens)]
                       for _ in range(args.offline_requests)]
    async with httpx.AsyncClient(timeout=args.offline_timeout, limits=httpx.Limits(max_connections=1024)) as client:
        async def request(index, prompt, scheduled, *, cls="online", output_tokens=None,
                          input_tokens=None, priority=None, timeout=None):
            output_tokens = output_tokens or args.output_tokens
            input_tokens = input_tokens or args.input_tokens
            start = time.perf_counter()
            result = {"index": index, "class": cls, "ok": False, "dispatch_lateness_s": max(0, start-scheduled),
                      "submitted_at_s": start - t0}
            parser = StreamMeasurement()
            payload = {"model": args.model, "prompt": prompt, "max_tokens": output_tokens,
                       "temperature": 0, "seed": args.seed, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}}
            if priority is not None:
                payload["priority"] = priority   # honoured only with --scheduling-policy priority
            try:
                async with asyncio.timeout(timeout or args.timeout), client.stream("POST", args.base + "/v1/completions", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            parser.consume(line[5:].strip(), time.perf_counter()-start)
                result.update(parser.result(output_tokens))
                if result["prompt_tokens"] != input_tokens:
                    raise ValueError("Server prompt token count differs from workload")
                result["ok"] = True
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["e2e_s"] = time.perf_counter()-start
            return result

        for i in range(args.warmup):
            warmed = await asyncio.gather(*(request(-1, prompts[(i+j) % len(prompts)], time.perf_counter())
                                           for j in range(args.warmup_concurrency)))
            for warm in warmed:
                if not warm["ok"]:
                    raise RuntimeError(f"Warmup failed: {warm}")
        async def metrics():
            try:
                response = await client.get(args.base + "/metrics")
                response.raise_for_status()
                return response.text
            except Exception as exc:
                return f"# unavailable: {type(exc).__name__}\n"
        (root / "metrics-before.prom").write_text(await metrics())
        measured_start_unix = time.time()
        start = time.perf_counter()
        offsets = []
        offset = 0.0
        arrivals = random.Random(args.seed + 100000)
        for _ in prompts:
            offsets.append(offset)
            offset += arrivals.expovariate(args.rate)
        t0 = start
        def keep(result):
            records.append(result)
            with (root / "requests.jsonl").open("a") as out:
                out.write(json.dumps(result) + "\n")
        async def scheduled_request(index, prompt):
            scheduled = start + offsets[index]
            await asyncio.sleep(max(0, scheduled-time.perf_counter()))
            keep(await request(index, prompt, scheduled, priority=args.online_priority))
        burst_start = None
        async def offline_burst():
            nonlocal burst_start
            await asyncio.sleep(max(0, start + args.offline_at - time.perf_counter()))
            burst_start = time.perf_counter() - start
            # A batch job: everything submitted at once, no pacing.
            results = await asyncio.gather(*(
                request(10000 + i, p, time.perf_counter(), cls="offline", output_tokens=args.offline_output_tokens,
                        input_tokens=args.offline_input_tokens, priority=args.offline_priority,
                        timeout=args.offline_timeout)
                for i, p in enumerate(offline_prompts)))
            for r in results:
                keep(r)
        tasks = [scheduled_request(i, p) for i, p in enumerate(prompts)]
        if args.offline_requests:
            tasks.append(offline_burst())
        await asyncio.gather(*tasks)
        wall = time.perf_counter()-start
        measured_end_unix = time.time()
        (root / "metrics-after.prom").write_text(await metrics())
    summary = summarize(records, wall, args.ttft_slo, args.e2e_slo)
    summary.update(measured_start_unix=measured_start_unix, measured_end_unix=measured_end_unix)
    if args.offline_requests:
        summary["offline"] = summarize_offline(records, burst_start, wall)
        summary["online_priority"] = args.online_priority
        summary["offline_priority"] = args.offline_priority
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:18001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", required=True)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--input-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--warmup-concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--ttft-slo", type=float, default=1)
    p.add_argument("--e2e-slo", type=float, default=10)
    # Optional offline burst on top of the online stream (a batch job sharing the GPU).
    p.add_argument("--offline-requests", type=int, default=0)
    p.add_argument("--offline-at", type=float, default=5.0, help="seconds after the stream starts")
    p.add_argument("--offline-input-tokens", type=int, default=1024)
    p.add_argument("--offline-output-tokens", type=int, default=256)
    p.add_argument("--offline-timeout", type=float, default=600)
    p.add_argument("--online-priority", type=int, default=None, help="vLLM priority (lower wins); omitted = not sent")
    p.add_argument("--offline-priority", type=int, default=None)
    args = p.parse_args()
    if min(args.requests, args.rate, args.input_tokens, args.output_tokens, args.timeout, args.warmup_concurrency) <= 0:
        p.error("counts, rate and timeout must be positive")
    if (Path(args.output) / "requests.jsonl").exists():
        p.error("output already contains records; choose a fresh run directory")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
