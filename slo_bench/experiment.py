"""One bounded candidate: isolated loopback server, telemetry, guaranteed cleanup."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def sample(root, stop):
    with (root / "telemetry.jsonl").open("w") as out:
        while not stop.is_set():
            item = {"unix": time.time()}
            try:
                item["gpu"] = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
            except Exception as exc:
                item["gpu_error"] = type(exc).__name__
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=3) as response:
                    item["metrics"] = [line for line in response.read().decode().splitlines()
                                       if not line.startswith("#") and any(k in line for k in
                                       ("kv_cache_usage", "gpu_cache_usage", "num_preemptions", "num_requests_running", "num_requests_waiting"))]
            except Exception as exc:
                item["metrics_error"] = type(exc).__name__
            out.write(json.dumps(item) + "\n")
            out.flush()
            stop.wait(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", type=int, required=True)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--output", required=True)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=2)
    p.add_argument("--seeds", default="17")
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", "Qwen/Qwen2.5-7B-Instruct", "--host", "127.0.0.1", "--port", "8001",
               "--dtype", "bfloat16", "--max-model-len", "4096", "--gpu-memory-utilization", "0.9",
               "--max-num-seqs", str(args.seqs), "--max-num-batched-tokens", str(args.tokens),
               "--no-enable-prefix-caching", "--enable-chunked-prefill", "--seed", "17"]
    environment = {"python": platform.python_version(), "command": command,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
                   "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True)}
    (root / "environment.json").write_text(json.dumps(environment, indent=2))
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()
        sampler = None
        try:
            deadline = time.monotonic()+600
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server exited; inspect server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server not ready in 600s")
                time.sleep(2)
            print(f"READY seqs={args.seqs} tokens={args.tokens}", flush=True)
            sampler = threading.Thread(target=sample, args=(root, stop), daemon=True)
            sampler.start()
            for seed in args.seeds.split(","):
                subprocess.run([sys.executable, "-m", "slo_bench.measure", "--output", str(root / f"seed-{seed}"),
                                "--seed", seed, "--requests", str(args.requests), "--rate", str(args.rate)], check=True)
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=10)
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            text = (root / "server.log").read_text(errors="replace")
            (root / "server-diagnostics.json").write_text(json.dumps({
                "oom_log_lines": [line for line in text.splitlines() if "out of memory" in line.lower()],
                "exit_code": server.returncode,
                "note": "Negative exit code may reflect intentional experiment cleanup, not request failure."}, indent=2))


if __name__ == "__main__":
    main()
