"""Mixed-load study: an online stream with a TTFT SLO plus an offline batch burst on one GPU.

    python -m slo_bench.mixed                 # full matrix into results/mixed/
    python -m slo_bench.mixed --only fcfs priority

Arms (all: Qwen2.5-7B-Instruct bf16, RTX 4090, max_num_seqs 128, 4096-token budget,
online 64 requests at 2 req/s of 256 in / 128 out, SLO TTFT <= 1 s and E2E <= 10 s;
offline burst of 96 requests of 1024 in / 256 out submitted 5 s in):

  fcfs                vLLM default scheduling, no priorities sent
  priority            --scheduling-policy priority, online priority 0, offline priority 10
  priority-nochunk    priority, chunked prefill disabled
  priority-halfkv     priority, gpu-memory-utilization 0.45 (about half the KV blocks)

fcfs and priority get three paired seeds; the two ablation arms one seed each.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ONLINE = "--requests 64 --rate 2"
OFFLINE = "--offline-requests 96 --offline-at 5 --offline-input-tokens 1024 --offline-output-tokens 256"
PRIO = "--online-priority 0 --offline-priority 10"

ARMS = {
    "fcfs":             {"policy": "fcfs",     "seeds": "101,102,103", "extra": [], "measure": OFFLINE},
    "priority":         {"policy": "priority", "seeds": "101,102,103", "extra": [], "measure": f"{OFFLINE} {PRIO}"},
    "priority-nochunk": {"policy": "priority", "seeds": "101", "extra": ["--no-chunked-prefill"], "measure": f"{OFFLINE} {PRIO}"},
    "priority-halfkv":  {"policy": "priority", "seeds": "101", "extra": ["--gpu-memory-utilization", "0.45"], "measure": f"{OFFLINE} {PRIO}"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--root", default="results/mixed")
    args = p.parse_args()
    for name, arm in ARMS.items():
        if args.only and name not in args.only:
            continue
        out = Path(args.root) / name
        if out.exists():
            print(f"skip {name}: {out} exists", flush=True)
            continue
        cmd = [sys.executable, "-m", "slo_bench.experiment", "--seqs", "128", "--tokens", "4096",
               "--requests", "64", "--rate", "2", "--seeds", arm["seeds"],
               "--scheduling-policy", arm["policy"], "--output", str(out),
               "--measure-args", arm["measure"]] + arm["extra"]
        print("RUN", name, " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "-m", "slo_bench.mixed_report", "--root", args.root], check=True)


if __name__ == "__main__":
    main()
