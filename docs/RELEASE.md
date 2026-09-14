# v0.1: scheduler evidence and application walkthrough

## Purpose
Deliver an inspectable single-GPU study and an accessible demonstration of the existing serving platform. The selected tuning candidate is not an improvement over the128-sequence reference; retain the reference for this workload.

## Delivered
- English and Chinese design and reproducible measurement contract.
-16 completed runs, including nine formal paired-seed runs; raw per-request records, telemetry, tables and plots.
- Simplified existing Chat/Admin UI; real browser evidence of chat and registry/warm-pool operations.
-95-second1080p H.264/AAC walkthrough, Chinese synthetic narration and subtitle track, assembled from actual screenshots. Not continuous recording and not a timing benchmark.

## Verification
-8 measurement tests pass; analyzer recomputed16 completed summaries; GitHub CPU CI green at results commit7f153cb.
- Restored Ray Serve/vLLM application; RunPodhealth200; Render actual nonstream answer with prompt/completion usage35/5; Render SSE terminated DONE; browser Chat returned a complete answer.
- ffprobe confirmed1920x1080,H.264,AAC,mov_text,94.896seconds; full ffmpeg decode completed without reported errors. Chat screenshot visually inspected; other screenshots captured during actual operations.

## Boundaries and operations
Single model/GPU and synthetic workload, grouped repeats; no significance or production cost claim. The original InferenceGateway HF-versus-vLLM concurrency1/4/16 comparison remains outstanding. Terraform was validated, not applied to AWS. Admin controls do not deploy GPU weights. Full deployment lifecycle upgrade is deferred. GPU remains running; no authorization to terminate it or delete storage. The public demo is on-demand, not an uptime promise.

This release changes documentation/evidence, not production scheduling. Rollback consists of reverting the release documentation commit or removing its media asset; raw historical evidence should remain available.
