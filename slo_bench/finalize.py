"""Freeze candidates from screening, then run paired-seed formal comparisons."""
import json
from pathlib import Path
import subprocess
import sys


def choose(paths):
    candidates = []
    for path in paths:
        row = json.loads(path.read_text())
        env = json.loads((path.parent.parent / "environment.json").read_text())
        cmd = env["command"]
        candidates.append({"seqs": int(cmd[cmd.index("--max-num-seqs")+1]),
                           "tokens": int(cmd[cmd.index("--max-num-batched-tokens")+1]),
                           "goodput": row["goodput_rps"], "errors": row["errors"], "source": str(path)})
    eligible = [c for c in candidates if c["errors"] == 0]
    if not eligible:
        raise RuntimeError("No error-free candidate; inspect evidence instead of launching formal tests")
    best = max(c["goodput"] for c in eligible)
    # Within 5% of best observed goodput, prefer the smaller configuration.
    return min((c for c in eligible if c["goodput"] >= best*.95), key=lambda c: (c["seqs"], c["tokens"]))


def main():
    root = Path("results")
    chosen_seq = choose(root.glob("screen-s*-t4096/seed-17/summary.json"))["seqs"]
    subprocess.run([sys.executable, "-m", "slo_bench.sweep", "--seqs", str(chosen_seq),
                    "--tokens", "2048,8192", "--prefix", "budget"], check=True)
    candidates = list(root.glob(f"screen-s{chosen_seq}-t4096/seed-17/summary.json"))
    candidates += list(root.glob(f"budget-s{chosen_seq}-t*/seed-17/summary.json"))
    chosen = choose(candidates)
    chosen.update({"formal_seeds": [101,102,103], "formal_requests":128,
                   "selection_rule":"zero errors; within 5% of max goodput; prefer smaller seqs then budget",
                   "order":"three baseline runs then three selected runs; grouped to limit rented-GPU startup cost"})
    (root / "selection.json").write_text(json.dumps(chosen, indent=2))
    for seq, tokens, label in [(1,4096,"baseline"), (chosen["seqs"],chosen["tokens"],"selected")]:
        subprocess.run([sys.executable, "-m", "slo_bench.experiment", "--seqs", str(seq), "--tokens", str(tokens),
                        "--requests", "128", "--seeds", "101,102,103", "--output", f"results/formal-{label}"], check=True)
    subprocess.run([sys.executable, "-m", "slo_bench.analyze"], check=True)


if __name__ == "__main__":
    main()
