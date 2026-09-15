---
name: oss-ego-pipeline
description: Build and run a reviewable OSS Ego data pipeline that streams recordings, extracts structured action chains, classifies task difficulty, and reports diversity across heterogeneous suppliers.
---

# OSS Ego Pipeline

Use this skill when developing or operating the `oss://ccm-ego/` Ego-data pipeline.

Read [PPU_CONTEXT.md](../../PPU_CONTEXT.md) before changing the pipeline. It contains the authoritative user decisions and corrections from the working session.

## Non-negotiable semantics

- Treat `recording` as the primary difficulty unit. Do not automatically split long recordings into independent difficulty samples.
- Count synchronized multi-camera views once for task duration; count independent repeated recordings separately.
- Preserve `unknown` segments with time ranges and reasons. They do not count toward N/H but do count toward pending duration.
- H requires evidence of interaction, conditional decision, or multi-thread coordination. Candidate evidence must be distinct from confirmed H.
- Do not infer raw fields from directory names. Label evidence as confirmed, file-name/structure-only, name hint, or pending.

## Runtime boundary

- Input is an OSS prefix or recording manifest.
- Use a short-lived signed URL and stream-decode video. Never persist the original video.
- Send only selected frames to the cloud VLM; never send OSS credentials or raw sensor files.
- Default sampling is 2 seconds, adaptive 0.5 seconds around changes, with a 300-frame per-recording cloud budget.
- Preferred cloud VLM is Alibaba Bailian/Tongyi Vision. Keep the adapter replaceable.

## Model roles

- Local YOLO/RT-DETR + ByteTrack: objects, tracks, changes, key-frame selection.
- VLM: fine-grained action segments plus whole-recording summary.
- Rules: canonical vocabulary mapping, segment merging, T/N/H calculation, evidence checks.

## Required output

Every semantic segment must retain `t_start`, `t_end`, canonical and raw action/object labels, confidence, and source evidence. Every difficulty result must retain `T`, `N`, `H`, H type, level, review status, and evidence intervals. Produce Markdown/HTML diversity reports and a machine-readable review queue.

## Initial difficulty rules

- High: confirmed H ≥ 1.
- Low: confirmed H=0, N=1–3, complete chain.
- Medium: confirmed H=0, N≥4 with a coherent dependent chain.
- Pending: missing/conflicting timing, task boundary, N/H evidence, or candidate-only high-difficulty evidence.

The initial target mix is low 30%, medium 50%, high 20%, but thresholds must be calibrated against human-labeled samples before being treated as formal acceptance gates.
