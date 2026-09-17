"""Execute the frozen demo sample list and retain outputs, checks and repeat runs."""
import ast, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/demo_20260917'
MODEL='/mnt/workspace/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master'
MANIFEST=ROOT/'demo_sample_list.json'
# Read the literal vocabulary without importing torch in this orchestration process.
TREE=ast.parse((ROOT/'qwen_vl_pipeline.py').read_text())
ACTIONS=set(next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ATOMIC_ACTIONS' for t in n.targets)))

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def num(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def union_length(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)
def metrics(r,duration):
    s=r.get('semantic',{}); ws=s.get('windows',[]);ts=r.get('timestamps_s',[])
    total=valid_vocab=valid_ref=valid_schema=parsed=0; intervals=[]; errors=[]
    for w in ws:
        v=w.get('semantic',{})
        good=isinstance(v,dict) and isinstance(v.get('action_segments'),list) and 'raw_output' not in v
        parsed+=int(good)
        for i,a in enumerate(v.get('action_segments',[]) if good else []):
            total+=1
            if not isinstance(a,dict):errors.append({'window':w['window_index'],'segment':i,'reason':'not_object'});continue
            valid_vocab+=int(a.get('canonical_action') in ACTIONS)
            valid_schema+=int(all(k in a for k in ('start_s','end_s','canonical_action','raw_action','object','confidence','evidence_times')))
            start,end=a.get('start_s'),a.get('end_s');e=a.get('evidence_times')
            ok=num(start) and num(end) and 0<=start<end<=duration+.05 and w['time_start']-.05<=start and end<=w['time_end']+.05
            ok=ok and isinstance(e,list) and bool(e) and all(num(t) and start-.05<=t<=end+.05 and any(abs(t-x)<=.05 for x in ts[w['frame_start']:w['frame_end']]) for t in e)
            valid_ref+=int(ok)
            if ok:intervals.append((max(0,start),min(duration,end)))
            else:errors.append({'window':w['window_index'],'segment':i,'reason':'invalid_time_or_frame_reference','segment_data':a})
    covered=union_length(intervals)
    return {'windows':len(ws),'parsed_windows':parsed,'segments':total,'vocabulary_valid':valid_vocab,
      'schema_complete':valid_schema,'time_and_reference_valid':valid_ref,'validated_interval_coverage_s':round(covered,3),
      'validated_interval_coverage_ratio':covered/duration if duration else 0,'errors':errors,
      'level':s.get('difficulty',{}).get('level'),'N':s.get('difficulty',{}).get('N'),'H':s.get('difficulty',{}).get('H'),
      'review_status':s.get('difficulty',{}).get('review_status'),'pipeline_status':r.get('status'),
      'note':'Schema/time checks are engineering checks, not semantic accuracy. Model H=0 is not proof of absence.'}
def mdclean(s):return str(s).replace('|','／').replace('\n',' ')
def render_manifest(m):
    rows=[]
    for s in m['samples']:
        r=s.get('result',{}); v=r.get('metrics',{}); review=s.get('review',{})
        conclusion=review.get('conclusion',r.get('summary','待运行'))
        rows.append(f"| {s['id']} | {mdclean(s['source'])} | {s['duration_s']:.2f} | {s['status']} | {v.get('level','—')} / {v.get('N','—')} / {v.get('H','—')} | {mdclean(conclusion)} | [报告](outputs/demo_20260917/reports/{s['id']}.md) · [JSON]({s['report']}) |")
    text='''# demo_sample_list

固定随机种子：20260917。10个样本来自已盘点目录的来源分层随机抽样，不是全桶等概率抽样；单文件必须0 < 时长 ≤ 600秒。过滤派生轨迹、review视频和补充机位；同一目录不重复取样。已存档候选范围和全部时长探测记录。

本轮冻结 pipeline 提交与SHA，统一2秒/帧、300帧预算、16帧/窗、最长600秒。所有首轮及复跑均重新读取视频、重新推理；仅将本次抽帧存为复核证据，不读取历史推理结果。标注只用于事后比较，不送入提示词。

难度列是**原始规则初判**，不是人工真值。工程成功率、词表合规率、时间引用合规率、重复运行一致率与视觉语义支持率分别统计，不混作准确率。没有人工真值的样本不计算正式level准确率。

| ID | 来源 | 时长/秒 | 状态 | 原始level / N / H | 最重要结论 | 文件 |
|---|---|---:|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n## 输入路径\n\n'+ '\n'.join(f"- **{s['id']}**：`{s['oss_uri']}`" for s in m['samples'])
    text+='\n\n## 复跑样本\n\n'+', '.join(f'[{sid}](outputs/demo_20260917/reports/{sid}_repeat.md)' for sid in m['repeat_ids'])+'；每个再运行一次。重复一致只说明可复现，不说明语义正确。\n'
    text+='\n## 问题解决方案与实施难度\n\n难度是研发工作量和联调风险的估计，不是模型识别难度：低=半天至1天，中=1–3天，高=3天以上或需要重新标注/评测。执行顺序按 P0 → P1 → P2；P0 解决结果不可用，P1 提升语义与 level，P2 建立正式评测和性能基线。\n\n| 优先级 | 具体解决方案 | 难度 | 预期收益 | 验收标准 |\n|---|---|---|---|---|\n'
    for p in m.get('priority_plan',[]):
        text+=f"| **{p['priority']}** | {mdclean(p['solution'])} | {p['difficulty']} | {mdclean(p['benefit'])} | {mdclean(p['acceptance'])} |\n"
    text+='\n[评估总报告](outputs/demo_20260917/evaluation_summary.md) · [机器可读清单](demo_sample_list.json) · [汇总指标](outputs/demo_20260917/summary.json) · [问题与修正优先级](outputs/demo_20260917/findings.md)\n'
    (ROOT/'demo_sample_list.md').write_text(text)
def preview(cache,name):
    ip=cache/'frames.json'
    if not ip.exists():return None
    records=json.loads(ip.read_text())['frames'];n=min(12,len(records))
    if not n:return None
    ids=sorted(set(round(i*(len(records)-1)/max(1,n-1)) for i in range(n)))
    canvas=Image.new('RGB',(1280,225*math.ceil(len(ids)/4)),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for k,i in enumerate(ids):
        item=records[i]
        with Image.open(cache/item['file']) as image:
            image.thumbnail((316,195));x=k%4*320;y=k//4*225;canvas.paste(image,(x,y));draw.text((x+3,y+198),f"{item['timestamp']:.3f}s | frame {i}",fill='black')
    path=OUT/'previews'/f'{name}.jpg';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=92)
    return str(path.relative_to(ROOT))
def render_report(sample,r,checks,name):
    s=r.get('semantic',{});ws=s.get('windows',[])
    text=f"# {name} 测试报告\n\n视频：`{sample['oss_uri']}`\n\n实测时长 {sample['duration_s']:.3f} 秒；分辨率 {sample['width']}×{sample['height']}；帧率 {sample['fps']:.3f}。\n\n"
    text+=f"pipeline 状态：`{r.get('status')}`；帧数 {r.get('frame_count',0)}；窗口 {checks['parsed_windows']}/{checks['windows']} 个解析成功。\n\n"
    text+=f"原始规则初判：**{checks['level']}**，N={checks['N']}，H={checks['H']}，复核状态 `{checks['review_status']}`。这些不是已校准真值。\n\n"
    text+=f"工程检查：动作 {checks['segments']} 段；词表合规 {checks['vocabulary_valid']} 段；schema齐全 {checks['schema_complete']} 段；时间与帧引用合规 {checks['time_and_reference_valid']} 段；有效动作区间并集覆盖 {checks['validated_interval_coverage_ratio']:.1%}。\n\n"
    text+='## 窗口摘要\n\n| 窗口 | 时间/秒 | 模型摘要 |\n|---|---|---|\n'
    for w in ws:text+=f"| {w['window_index']+1} | {w['time_start']:.3f}–{w['time_end']:.3f} | {mdclean(w['semantic'].get('summary','JSON失败，见原始输出'))} |\n"
    text+='\n## 动作候选\n\n| 时间/秒 | 原子动作 | 原始描述 | 对象 | 证据帧/秒 |\n|---|---|---|---|---|\n'
    for a in s.get('action_segments',[]):
        if not isinstance(a,dict):continue
        text+=f"| {a.get('start_s')}–{a.get('end_s')} | {mdclean(a.get('canonical_action'))} | {mdclean(a.get('raw_action'))} | {mdclean(a.get('object'))} | {a.get('evidence_times')} |\n"
    text+='\n## 高难候选\n\n```json\n'+json.dumps(s.get('high_difficulty_candidates',[]),ensure_ascii=False,indent=2)+'\n```\n'
    text+='\n## 证据总览\n\n'+f'![抽帧总览](../previews/{name}.jpg)\n\n'
    text+=f'[原始输出]({name}.json) · [独立工程核查]({name}.checks.json)\n'
    (OUT/'reports'/f'{name}.md').write_text(text)
def run_one(sample,name,m):
    cache=ROOT/'work/demo_20260917'/name
    if cache.exists():raise RuntimeError(f'Fresh run requires new cache: {cache}')
    output=OUT/'reports'/f'{name}.json';log=OUT/'logs'/f'{name}.log'
    cmd=[sys.executable,str(ROOT/'qwen_vl_pipeline.py'),'--model',MODEL,'--video',sample['video'],'--out',str(output),'--interval','2','--max-frames','300','--window-frames','16','--max-duration','600','--frames-cache',str(cache)]
    started=time.monotonic();error=None
    with log.open('w') as f:
        try: code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
        except subprocess.TimeoutExpired:code=-1;error='timeout_1800s'
    if output.exists():r=json.loads(output.read_text())
    else:
        r={'status':'failed','error':error or f'exit_code_{code}','video':sample['video'],'semantic':{},'review_queue':[{'reason':'pipeline_failed','log':str(log.relative_to(ROOT))}]};dump(output,r)
    checks=metrics(r,sample['duration_s']);dump(OUT/'reports'/f'{name}.checks.json',checks)
    p=preview(cache,name);render_report(sample,r,checks,name)
    result={'exit_code':code,'wall_seconds':round(time.monotonic()-started,2),'metrics':checks,
        'summary':r.get('semantic',{}).get('summary','失败，见日志')[:400],'preview':p,'report':str(output.relative_to(ROOT))}
    print(name,code,r.get('status'),f"windows={checks['parsed_windows']}/{checks['windows']}",f"level={checks['level']}",flush=True)
    return result

def main():
    m=json.loads(MANIFEST.read_text())
    if hashlib.sha256((ROOT/'qwen_vl_pipeline.py').read_bytes()).hexdigest()!=m['pipeline_sha256']:raise RuntimeError('Pipeline changed since selection')
    render_manifest(m)
    for sample in m['samples']:
        if sample['status'] in ('complete','failed'):continue
        sample['status']='running';dump(MANIFEST,m);render_manifest(m)
        sample['result']=run_one(sample,sample['id'],m)
        sample['status']='complete' if sample['result']['exit_code']==0 else 'failed'
        dump(MANIFEST,m);render_manifest(m)
    for sid in m['repeat_ids']:
        sample=next(s for s in m['samples'] if s['id']==sid)
        if 'repeat' in sample:continue
        sample['repeat']=run_one(sample,sid+'_repeat',m);dump(MANIFEST,m);render_manifest(m)
    print('ALL_RUNS_COMPLETE',flush=True)
if __name__=='__main__':main()
