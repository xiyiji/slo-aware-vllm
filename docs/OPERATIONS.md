# Operations and evidence boundaries

## Public demo vs experiment

The public Vercel console forwards to Render, which forwards to a RunPod Ray Serve deployment. Engine experiments deliberately bypass these hops. A local benchmark number must never be labeled public end-to-end latency.

The console's Admin controls register/promote version metadata and manipulate a gateway warm-pool record. They do not download model weights, free CUDA memory, or perform real GPU deployment. This is stated in the UI and walkthrough.

## Recovery checklist

1. Confirm the experimental process has ended and `nvidia-smi` shows GPU memory released.
2. Activate `/workspace/InferenceGateway/.venv`; set `HF_HOME=/workspace/.cache/huggingface`.
3. In `/workspace/InferenceGateway`, pull the tested integration commit.
4. With the existing Ray head running, `serve start --address auto --http-host 0.0.0.0 --http-port 8000`.
5. `serve run --address auto serve.app:deployment` (use a persistent process manager/nohup for the demo).
6. Wait for application readiness; verify `/healthz` and an actual generated answer, not just Ray's process status.
7. Verify Render and browser Chat. Do not use `http://127.0.0.1:8265` as a Ray GCS address.

No cloud account credential is required in this repository. A new Pod changes its public proxy URL; update the existing Render upstream setting deliberately. Do not publish a bearer token or a browser terminal session URL.

## Known exploratory failures

- Initial port8001 attempt reached the Pod template's nginx instead of vLLM. Warmup returned405. No performance results from this attempt are valid. Fixed by choosing loopback18001 and checking ownership before starting.
- Released sockets can remain in TIME_WAIT. The port preflight now uses SO_REUSEADDR, still rejecting an active listener.
- Initial serial-sequence screen used httpx's read timeout, not an absolute request deadline. Formal experiments use `asyncio.timeout` for a60s total deadline in both arms. Initial screening is retained as exploratory evidence, not the formal baseline.

## Video provenance

The walkthrough is assembled from screenshots captured during real browser actions: deployed homepage, Chat, model register/promote/load/unload, and GitHub CI. Edited pauses/captions are for explanation; playback timing is not performance measurement. No generated UI or simulated response is substituted for a live result.
