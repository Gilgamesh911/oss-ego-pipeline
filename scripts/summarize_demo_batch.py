"""Summarize operational reliability, repeatability and explicitly scoped audits."""
import collections, json, math, sys
from pathlib import Path
from run_demo_batch import ROOT, OUT, MANIFEST, dump, metrics, render_manifest

def rate(n,d):return {'numerator':n,'denominator':d,'ratio':n/d if d else None}
def jaccard(a,b):
    a,b=set(a),set(b);return len(a&b)/len(a|b) if a|b else 1.0

def main():
 m=json.loads(MANIFEST.read_text()); reviews=json.loads((OUT/'visual_reviews.json').read_text())
 comparisons=[];w_total=w_good=seg_total=vocab=schema=refs=0
 for s in m['samples']:
  if s['id'] in reviews['samples']:s['review']=reviews['samples'][s['id']]
  rp=ROOT/s['report'];r=json.loads(rp.read_text()) if rp.exists() else {}
  # Preserve raw VLM windows even when recording aggregation crashed.
  wp=ROOT/'work/demo_20260917'/s['id']/'windows.json'
  fp=wp.parent/'frames.json'
  if wp.exists():
   ws=json.loads(wp.read_text());ts=[x['timestamp'] for x in json.loads(fp.read_text())['frames']]
   raw={'timestamps_s':ts,'semantic':{'windows':ws}}
   vm=metrics(raw,s['duration_s']);w_total+=vm['windows'];w_good+=vm['parsed_windows'];seg_total+=vm['segments'];vocab+=vm['vocabulary_valid'];schema+=vm['schema_complete'];refs+=vm['time_and_reference_valid']
   dump(OUT/'reports'/f"{s['id']}.window_checks.json",vm)
   dump(OUT/'reports'/f"{s['id']}.windows.raw.json",ws)
  else:vm={}
  s['window_checks']=vm
  review=s.get('review',{})
  report_path=OUT/'reports'/f"{s['id']}.md"
  if report_path.exists():
   old=report_path.read_text().split('\n## 独立视觉复核')[0]
   old+='\n## 独立视觉复核\n\n'+review.get('conclusion','未复核')+'\n\n'
   if review.get('reference_label'):
    old+='原始文件任务文字：'+review['reference_label']+'；这是标签一致性参照，未经专家确认时不当作正式动作/难度真值。\n\n'
   old+=review.get('visual_evidence','')+'\n\n'
   old+='| 抽查动作序号（从0开始） | 视觉支持判断 | 理由 | 证据 |\n|---|---|---|---|\n'
   for a in review.get('audited_actions',[]):
    old+=f"| {a['segment_index']} | {a['verdict']} | {a['reason']} | [邻帧](../previews/{s['id']}_action_{a['segment_index']:03d}.jpg) |\n"
   if r.get('status')=='failed':
    old+=f"\n## 失败诊断与已保留的窗口\n\n退出码：{s['result']['exit_code']}；[运行日志](../logs/{s['id']}.log)。报告顶部的0帧/0窗口来自失败占位输出，不代表未抽帧。\n\n"
    old+=f"实际保留 {len(ts)} 帧，{vm.get('parsed_windows',0)}/{vm.get('windows',0)} 个可解析窗口、{vm.get('segments',0)} 个原始动作。聚合失败前的模型输出：[窗口原文]({s['id']}.windows.raw.json)。不能把可用窗口误记为整条管线成功。\n\n"
    for w in ws:old+=f"- {w['time_start']}–{w['time_end']}秒：{w['semantic'].get('summary','未解析')}\n"
   report_path.write_text(old)
  if s.get('repeat'):
   rr=json.loads((ROOT/s['repeat']['report']).read_text());a=r.get('semantic',{});b=rr.get('semantic',{})
   ad=a.get('difficulty',{});bd=b.get('difficulty',{})
   cmp={'id':s['id'],'both_runs_succeeded':s['result']['exit_code']==0 and s['repeat']['exit_code']==0,
      'frame_times_identical':r.get('timestamps_s')==rr.get('timestamps_s'),
      'semantic_json_identical':a==b,'level_identical':ad.get('level')==bd.get('level'),
      'N_identical':ad.get('N')==bd.get('N'),'H_identical':ad.get('H')==bd.get('H'),
      'N_values':[ad.get('N'),bd.get('N')], 'level_values':[ad.get('level'),bd.get('level')],
      'action_class_jaccard':jaccard([x.get('canonical_action') for x in a.get('action_segments',[])],[x.get('canonical_action') for x in b.get('action_segments',[])])}
   comparisons.append(cmp)
 actions=[a for s in m['samples'] for a in s.get('review',{}).get('audited_actions',[])]
 verdicts=collections.Counter(a['verdict'] for a in actions)
 labeled=[s for s in m['samples'] if s.get('review',{}).get('reference_agreement') is not None]
 totals={'sample_count':len(m['samples']),'total_video_seconds':sum(s['duration_s'] for s in m['samples']),
  'source_count':len(set(s['source'] for s in m['samples'])),
  'first_run_process_success':rate(sum(s.get('result',{}).get('exit_code')==0 for s in m['samples']),len(m['samples'])),
  'first_run_end_to_end_success':rate(sum(s.get('result',{}).get('exit_code')==0 and s['result']['metrics'].get('pipeline_status')=='candidate_semantics' for s in m['samples']),len(m['samples'])),
  'window_json_parse_success':rate(w_good,w_total),'atomic_vocabulary_compliance':rate(vocab,seg_total),
  'required_action_fields_present':rate(schema,seg_total),'time_and_frame_reference_valid':rate(refs,seg_total),
  'repeat_count':len(comparisons),'repeat_level_consistency':rate(sum(x['level_identical'] for x in comparisons),len(comparisons)),
  'repeat_semantic_json_consistency':rate(sum(x['semantic_json_identical'] for x in comparisons),len(comparisons)),
  'repeat_comparisons':comparisons,
  'first_run_level_distribution':dict(collections.Counter(s.get('result',{}).get('metrics',{}).get('level') or '失败/无结果' for s in m['samples'])),
  'assistant_visual_audit':{'method':reviews['method'],'counts':dict(verdicts),'strict_supported':rate(verdicts['supported'],len(actions)),
    'formal_expert_accuracy':None,'selection':'first/middle/last generated action per sample; failed pipelines use raw checkpoint windows; not a random action sample'},
  'task_label_agreement_proxy':rate(sum(s['review']['reference_agreement'] is True for s in labeled),len(labeled)),
  'formal_level_accuracy':None,
  'limitations':['来源分层的有限候选目录抽样，不代表全桶分布。','没有独立人工难度真值，不能报告正式level准确率。','参考任务标注可能是自动生成；只报告任务文字一致性。','视觉抽查是助手基于选中帧核查，非专家逐帧验收。','重复运行一致不等于正确；do_sample=False仍须实测。'],
  'pipeline_commit':m['pipeline_commit'],'pipeline_sha256':m['pipeline_sha256']}
 dump(OUT/'summary.json',totals);dump(MANIFEST,m);render_manifest(m)
 def f(x):return f"{x['numerator']}/{x['denominator']}（{x['ratio']:.1%}）" if x['ratio'] is not None else '无可用样本'
 table=[('完整管线成功率',f(totals['first_run_end_to_end_success'])),('窗口JSON解析成功率',f(totals['window_json_parse_success'])),('原子词表合规率',f(totals['atomic_vocabulary_compliance'])),('动作字段齐全率',f(totals['required_action_fields_present'])),('时间/帧引用合规率',f(totals['time_and_frame_reference_valid'])),('复跑level一致率',f(totals['repeat_level_consistency'])),('复跑完整语义JSON一致率',f(totals['repeat_semantic_json_consistency'])),('有参考任务文字的一致性',f(totals['task_label_agreement_proxy'])),('抽查动作严格视觉支持率（助手核查）',f(totals['assistant_visual_audit']['strict_supported']))]
 text='# Demo 批量测评结果\n\n'+f"共 {totals['sample_count']} 个视频、{totals['source_count']} 个来源，总视频时长 {totals['total_video_seconds']/60:.2f} 分钟。每个≤600秒；额外重复运行3个。\n\n"
 text+='**判断：当前输出适合作为待复核的任务摘要与动作候选，尚不能作为自动验收的精细动作标注或难度真值。** 本批已发现聚合崩溃、未提供帧的证据引用、动作混淆以及新闻视频被分配任务难度等问题。\n\n'
 text+='| 指标 | 结果 |\n|---|---|\n'+'\n'.join(f'| {k} | {v} |' for k,v in table)
 text+='\n\n**以上工程合规与一致性不是正式语义准确率。** 正式level准确率暂不能计算：没有独立人工真值。视觉支持率仅针对每条视频第一/中间/最后动作的抽查，不能外推全桶。\n\n'
 text+='视觉复核分类：'+json.dumps(dict(verdicts),ensure_ascii=False)+'。任务文字一致性仅比较主任务/包装目标，不代表具体对象、动作顺序或完成状态正确。\n\n'
 text+='## 稳定性复跑\n\n| 样本 | level两次 | N两次 | 语义JSON相同 | 原子动作集合Jaccard |\n|---|---|---|---|---|\n'
 for c in comparisons:text+=f"| {c['id']} | {c['level_values']} | {c['N_values']} | {c['semantic_json_identical']} | {c['action_class_jaccard']:.3f} |\n"
 text+='\n## 每条结论\n\n'+'\n'.join(f"- **{s['id']}**：{s.get('review',{}).get('conclusion','未完成复核')}" for s in m['samples'])
 text+='\n\n原始level分布：'+json.dumps(totals['first_run_level_distribution'],ensure_ascii=False)+'。没有独立确认的高难真值，因此不能评价高难事件召回率。\n'
 text+='\n\n## 当前限制\n\n'+'\n'.join('- '+x for x in totals['limitations'])
 text+='\n\n[总样本清单](../../demo_sample_list.md) · [完整指标JSON](summary.json) · [问题与修正优先级](findings.md) · [运行与指标说明](README.md)\n'
 (OUT/'evaluation_summary.md').write_text(text)
 print(json.dumps(totals,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
