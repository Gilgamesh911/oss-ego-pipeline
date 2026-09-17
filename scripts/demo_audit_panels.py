"""Export deterministic first/middle/last action evidence panels for review."""
import json,math
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1];out=root/'outputs/demo_20260917'
for rp in sorted((out/'reports').glob('sample_[0-9][0-9].json')):
 r=json.loads(rp.read_text());segments=r.get('semantic',{}).get('action_segments',[])
 sid=rp.stem; cache=root/'work/demo_20260917'/sid
 if not segments and (cache/'windows.json').exists():
  raw=json.loads((cache/'windows.json').read_text())
  segments=[s for w in raw for s in w.get('semantic',{}).get('action_segments',[])]
 if not segments:continue
 fp=cache/'frames.json'
 if not fp.exists():continue
 records=json.loads(fp.read_text())['frames']; times=[x['timestamp'] for x in records]
 picks=sorted(set([0,len(segments)//2,len(segments)-1])); audit=[]
 for i in picks:
  a=segments[i];start=a.get('start_s',0);end=a.get('end_s',start)
  if not isinstance(start,(float,int)) or not isinstance(end,(float,int)):continue
  targets=[start,max(start,(start+end)/2),end]
  indices=sorted(set(min(range(len(times)),key=lambda j:abs(times[j]-t)) for t in targets))
  indices=sorted(set(indices+[max(0,indices[0]-1),min(len(times)-1,indices[-1]+1)]))
  canvas=Image.new('RGB',(320*len(indices),230),'#eeeeee');d=ImageDraw.Draw(canvas)
  for k,j in enumerate(indices):
   item=records[j]
   with Image.open(cache/item['file']) as im:
    im.thumbnail((316,195));canvas.paste(im,(k*320,0));d.text((k*320+3,201),f'{times[j]:.3f}s / frame {j}',fill='black')
  name=f'{sid}_action_{i:03d}.jpg';canvas.save(out/'previews'/name,quality=94)
  audit.append({'segment_index':i,'claim':a,'evidence_panel':f'outputs/demo_20260917/previews/{name}','checked_times':[times[j] for j in indices]})
 (out/'reports'/f'{sid}.audit_selection.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 print(sid,[a['segment_index'] for a in audit])
