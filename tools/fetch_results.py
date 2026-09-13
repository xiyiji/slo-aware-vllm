"""Retrieve generated artifacts through RunPod's PTY-only SSH gateway.

No inbound HTTP server, cloud credential, or public upload is needed.
"""
import argparse
import base64
import io
from pathlib import Path
import select
import subprocess
import tarfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="Existing user@ssh.runpod.io login")
    p.add_argument("--identity", required=True)
    p.add_argument("--destination", default=".")
    a = p.parse_args()
    command = ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-i", a.identity, a.host]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    start, end = b"SLO_ARTIFACT_BEGIN", b"SLO_ARTIFACT_END"
    script = "stty -echo; printf '\\nSLO_ARTIFACT_BEGIN\\n'; tar -C /workspace/slo-aware-vllm --exclude='server.log' -czf - results | base64 -w 76; printf '\\nSLO_ARTIFACT_END\\n'; exit\n"
    proc.stdin.write(script.encode())
    proc.stdin.flush()
    data = b""
    deadline = time.monotonic()+120
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], 1)
            if ready:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                data += chunk
                # Echoed shell commands contain quote characters; markers must occupy a whole line.
                normalized = data.replace(b"\r", b"")
                if b"\n"+end+b"\n" in normalized:
                    break
        normalized = data.replace(b"\r", b"")
        payload = normalized.split(b"\n"+start+b"\n", 1)[1].split(b"\n"+end+b"\n", 1)[0]
        archive = base64.b64decode(b"".join(payload.split()), validate=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(Path(a.destination), filter="data")
        print(f"Retrieved {len(archive)} compressed bytes into {a.destination}")
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
