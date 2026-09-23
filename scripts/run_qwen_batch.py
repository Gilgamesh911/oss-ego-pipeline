#!/usr/bin/env python3
"""Resident Qwen3-VL batch runner: load the model once, process many videos."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_vl_pipeline import analyze_many, JsonlLogger, write_report_files


def read_videos(args):
    if args.video:
        return args.video
    data = json.loads(Path(args.manifest).read_text())
    if isinstance(data, dict) and "samples" in data:
        return [x["video"] for x in data["samples"]]
    if isinstance(data, dict) and "videos" in data:
        return list(data["videos"])
    if isinstance(data, list):
        return [x["video"] if isinstance(x, dict) else str(x) for x in data]
    raise ValueError("manifest must contain samples, videos, or a list")


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", action="append", help="repeatable local/OSS-mounted video path")
    source.add_argument("--manifest", help="JSON manifest containing samples or videos")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-cache-root")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--window-frames", type=int, default=16)
    ap.add_argument("--window-overlap", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--max-duration", type=float, default=600.0)
    ap.add_argument("--log", help="JSONL event log; defaults to <out>.log.jsonl")
    ap.add_argument("--report-dir", help="per-video report directory; defaults to <out stem>_reports")
    args = ap.parse_args()
    videos = read_videos(args)
    log_path = args.log or str(Path(args.out).with_suffix('.log.jsonl'))
    logger = JsonlLogger(log_path, run_id=Path(args.out).stem)
    results = analyze_many(videos, args.model, args.interval, args.max_frames,
                           args.window_frames, args.frames_cache_root,
                           args.max_duration, args.min_frames, args.window_overlap,
                           logger=logger)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir or (Path(args.out).parent / f"{Path(args.out).stem}_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_paths = []
    for index, result in enumerate(results):
        stem = Path(result.get("video", f"video_{index}")).stem or f"video_{index}"
        base = report_dir / f"{index:03d}_{stem}"
        paths = write_report_files(result, base.with_suffix('.json'))
        result["report_paths"] = paths
        report_paths.append(paths)
    Path(args.out).write_text(json.dumps({"videos": videos, "results": results,
                                          "log_path": log_path,
                                          "report_dir": str(report_dir)}, ensure_ascii=False, indent=2))
    logger.emit("batch_reports_written", report_dir=str(report_dir), report_count=len(report_paths))
    print(json.dumps({"videos": len(videos), "model_loads": 1, "out": args.out,
                      "log": log_path, "report_dir": str(report_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
