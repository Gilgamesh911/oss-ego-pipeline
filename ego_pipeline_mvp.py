#!/usr/bin/env python3
"""MVP for OSS Ego recording inspection and semantic report generation.

The MVP never downloads an original video. It consumes a JSON object manifest
and produces a reviewable report. Video/VLM access is intentionally an adapter
boundary so a signed URL can be supplied by an OSS service later.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ACTIONS = ["观察","接近","抓取","拿起","放下","移动","放置","打开","关闭","倒入","擦拭","折叠","装配","拆卸","按压","旋拧","交互","等待","恢复","未知"]
OBJECTS = ["容器","餐具","食材","衣物","工具","家具","电子设备","人体/手","清洁用品","其他"]
SCENES = ["居家","厨房","卧室","客厅","办公","酒店","商业","工业","户外","其他"]

def classify_file(key: str) -> str:
    k = key.lower()
    if k.endswith(('.mp4','.mov','.mkv','.avi')): return 'video'
    if k.endswith('.csv'): return 'csv'
    if k.endswith(('.json','.yaml','.yml','.md','.txt')): return 'metadata'
    if k.endswith(('.mcap','.bin','.idx','.hdf5','.zip','.tar','.gz')): return 'binary'
    return 'other'

def build_index(objects: list[dict]) -> dict:
    groups = {}
    for obj in objects:
        key = obj['key']; typ = classify_file(key)
        groups.setdefault(typ, []).append({**obj, 'type': typ})
    evidence = []
    for typ, items in groups.items():
        evidence.append({'type': typ, 'count': len(items), 'keys': [x['key'] for x in items]})
    return {'object_count': len(objects), 'by_type': evidence}

def make_report(prefix: str, objects: list[dict], config: dict) -> dict:
    idx = build_index(objects)
    missing = []
    names = [x['key'].lower() for x in objects]
    for required in ('meta.json','calibration.json'):
        if not any(x.endswith(required) for x in names): missing.append(required)
    videos = [x for x in objects if classify_file(x['key']) == 'video']
    return {
        'report_version': 'mvp-0.1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'input': {'prefix': prefix, 'object_count': len(objects), 'original_video_persisted': False},
        'config': config,
        'index': idx,
        'recording': {
            'recording_id': prefix.rstrip('/').split('/')[-1],
            'video_count': len(videos),
            'metadata_files': [x['key'] for x in objects if classify_file(x['key']) == 'metadata'],
            'binary_files': [x['key'] for x in objects if classify_file(x['key']) == 'binary'],
        },
        'semantic': {
            'status': 'pending_vlm' if videos else 'metadata_only',
            'summary': None,
            'action_segments': [],
            'canonical_action_vocabulary': ACTIONS,
            'object_vocabulary': OBJECTS,
            'scene_vocabulary': SCENES,
        },
        'difficulty': {'level': '待判定', 'T': None, 'N': None, 'H': None, 'reason': 'MVP 尚未调用 VLM 或缺少可解析语义证据'},
        'review_queue': [{'reason': 'metadata_missing', 'field': x} for x in missing] + ([{'reason': 'vlm_pending'}] if videos else []),
        'limitations': ['本 MVP 不读取或保存原始视频', 'CSV/JSON/YAML 内容解析和 VLM 调用由适配器接入', '没有语义证据时不把缺失当作 0'],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--manifest', required=True, help='JSON: {"objects":[{"key":...,"size":...}]}')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.manifest).read_text())
    config = {'frame_interval_s': 2.0, 'adaptive_interval_s': 0.5, 'max_vlm_frames': 300, 'confidence_review_threshold': 0.75, 'vlm_provider': 'aliyun-bailian/tongyi-vision'}
    report = make_report(args.prefix, data['objects'], config)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'status': 'ok', 'out': args.out, 'object_count': report['input']['object_count'], 'semantic_status': report['semantic']['status']}, ensure_ascii=False))

if __name__ == '__main__': main()
