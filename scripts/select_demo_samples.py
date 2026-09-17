"""Reproducible source-stratified sampling from a saved, bounded inventory."""
import json,random,subprocess,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
seed=20260917; rng=random.Random(seed)
inv=json.loads((root/'work/demo_sampling/inventory.json').read_text())
# Exclude derived renders, reviews and supplementary cameras. This is a source
# stratified demo, not a bucket-wide unbiased random sample.
def eligible(p):
    low=p.lower()
    return not any(t in low for t in ['/review/','trajectory','exo_cam','wrist','left_camera','right_camera','cam1','depth']) and not Path(p).name.startswith('._') and not ('/擎羽/' in p and '/videos/left_cam_left.mp4' not in p)
pools={k:[p for p in v['paths'] if eligible(p)] for k,v in inv.items()}
for paths in pools.values():rng.shuffle(paths)
source_order=sorted(k for k,v in pools.items() if v); rng.shuffle(source_order)
selected=[];probes=[];seen=set()
def choose(source):
    for _ in range(min(len(pools[source]),20)):
        p=pools[source].pop(); parent=str(Path(p).parent)
        if parent in seen:continue
        entry={'video':p,'source':source}
        try:
            r=subprocess.run([sys.executable,str(root/'scripts/demo_probe.py'),p],capture_output=True,text=True,timeout=20)
            if r.returncode:raise ValueError(r.stderr[-600:])
            info=json.loads(r.stdout); duration=info['duration_s']
            entry.update(info)
            entry['eligible']=duration is not None and 0<duration<=600
        except Exception as e:entry.update(eligible=False,error=str(e))
        probes.append(entry)
        if entry['eligible']:
            sid=f'sample_{len(selected)+1:02d}'
            entry.update(id=sid,oss_uri='oss://ccm-ego/'+str(Path(p).relative_to('/mnt/oss')),status='pending',report=f'outputs/demo_20260917/reports/{sid}.json')
            selected.append(entry.copy());seen.add(parent)
            print(sid,source,round(duration,2),p,flush=True);return
for s in source_order:choose(s)
while len(selected)<10:
    remaining=[s for s in source_order if pools[s]]
    if not remaining:break
    rng.shuffle(remaining)
    for s in remaining:
        choose(s)
        if len(selected)>=10:break
rng.shuffle(selected)
for i,s in enumerate(selected):
    s['id']=f'sample_{i+1:02d}';s['report']=f"outputs/demo_20260917/reports/{s['id']}.json"
manifest={'seed':seed,'method':'source-stratified random sampling from bounded inventory; one eligible video per source first, then random source fill; maxdepth=9; no manual selection by semantic result',
 'not_bucket_uniform':True,'inventory':{k:{'complete':v['complete'],'listed_count':len(v['paths'])} for k,v in inv.items()},
 'max_duration_s':600,'parameters':{'interval':2,'max_frames':300,'window_frames':16,'max_duration':600},
 'pipeline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'pipeline_sha256':hashlib.sha256((root/'qwen_vl_pipeline.py').read_bytes()).hexdigest(),
 'repeat_ids':sorted(rng.sample([s['id'] for s in selected],3)), 'samples':selected}
(root/'demo_sample_list.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(root/'outputs/demo_20260917/selection_probes.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2))
(root/'outputs/demo_20260917/inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2))
assert len(selected)==10
