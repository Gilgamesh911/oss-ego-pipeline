#!/usr/bin/env python3
"""Resident Qwen3-VL batch runner: load the model once, process many videos."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_vl_pipeline import analyze_many


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
    args = ap.parse_args()
    videos = read_videos(args)
    results = analyze_many(videos, args.model, args.interval, args.max_frames,
                           args.window_frames, args.frames_cache_root,
                           args.max_duration, args.min_frames, args.window_overlap)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"videos": videos, "results": results}, ensure_ascii=False, indent=2))
    print(json.dumps({"videos": len(videos), "model_loads": 1, "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
