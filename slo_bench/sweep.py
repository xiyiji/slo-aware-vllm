"""Bounded serial sweep; stop on launch/warmup failures, never overlap GPU engines."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="1,8,16,32,64")
    p.add_argument("--tokens", default="4096")
    p.add_argument("--prefix", default="screen")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--seeds", default="17")
    a = p.parse_args()
    for seq in a.seqs.split(","):
        for tokens in a.tokens.split(","):
            print(f"START seqs={seq} budget={tokens}", flush=True)
            subprocess.run([sys.executable, "-m", "slo_bench.experiment", "--seqs", seq, "--tokens", tokens,
                            "--requests", str(a.requests), "--seeds", a.seeds,
                            "--output", f"results/{a.prefix}-s{seq}-t{tokens}"], check=True)


if __name__ == "__main__":
    main()
