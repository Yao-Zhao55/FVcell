"""Read-only project audit; freezes a supplemental evaluation specification.

No training, prediction, outcome selection, or scientific inference is performed.
The existing test summaries were already observed; this is not preregistration.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv, hashlib, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PROJECT=BASE.parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return json.loads((BASE/p).read_text(encoding='utf-8'))
PAPERS={
 'B03':{'title':'The recoverable resolution of cellular perturbation-response prediction','version':'bioRxiv v1, posted 2026-10-01','url':'https://doi.org/10.64898/2026.09.30.755756','local':'literature_20261026/sources/B03_0.txt','sections':'Introduction; Primary human CD4+ T-cell CGC analysis; Discussion'},
 'B02':{'title':'Mechanisms Matter: Transportability of Cellular Perturbation Effects','version':'bioRxiv v3, posted 2026-10-08','url':'https://doi.org/10.64898/2026.05.08.723625','local':'literature_20261026/sources/B02_0.txt','sections':'2.1-2.3; 4.2; 4.3.3; 5'},
 'PIE':{'title':'PIE: Generalizing perturbation effects across unseen perturbations, contexts and datasets','version':'bioRxiv v1, posted 2026-10-05','url':'https://doi.org/10.64898/2026.10.02.756297','local':'literature_20261026/sources/M03_0.txt','sections':'3.1-3.6; 4.1-4.4'},
}
MATRIX=[
 ('研究目标','原代T细胞条件终点预测','共享与背景特异响应的恢复','跨背景因果可迁移条件','群体效应及DE预测'),
 ('主要数据','GWT原代CD4+ T；另有癌细胞外部测试','包含同类四供者、三刺激状态','包含Zhu25四供者、三状态','以多癌细胞系为主，另有跨数据集任务'),
 ('输出尺度','完整受测基因效应；供者响应与合并DE分开','响应、背景差异及较粗分辨率','均值及分布指标','DE概率、LFC、群体表达差值'),
 ('共享/残差','低秩门控响应＋分层偏移','明确分解共享与背景特异效应','以背景机制变化分析迁移','多来源知识和基因级证据解码'),
 ('预测时信息','按P0/P1/P2分层','允许基线、参考背景及指定锚点','明确源/目标背景的可用观测','对照、同扰动及同背景训练证据'),
 ('先验与收缩','Reactome表示、图正则、Gaussian MAP','已有锚点与低秩比较','有DESeq2/ash效应估计收缩','功能/互作/依赖等多来源知识'),
 ('缺失数据','人工嵌套训练标签掩码；不计入生物结果','完整案例及观测条目设计，不混为零','本轮未核实等价缺失标签实验','缺失知识源的处理不等于标签缺失'),
 ('泛化评测','供者、组合、全局靶点、整研究分开','跨背景、锚点及完整度分析','匹配的背景内/跨背景比较','背景/干预/数据集及联合未见'),
 ('统计与不确定性','四供者描述性推断；保留原校准','重复与恢复度不确定性','多指标和模拟机制验证','验证AUPRC选模型；按折汇总'),
 ('功能证据','同研究9靶点、额外供者；关联不稳','包含功能/类器官分析','主线为转录效应迁移','主线为扰动及DE泛化'),
]
CLAIMS=[
 {'claim':'首次提出共享扰动＋背景残差','decision':'删除首次/原创组件表述','reason':'B03存在直接概念先例；条件低秩方法也有广泛先例。','supported_now':False},
 {'claim':'首次跨供者评测原代T细胞','decision':'删除','reason':'B03/B02均有直接相关评测和相近数据。','supported_now':False},
 {'claim':'独创同靶点证据通道','decision':'删除','reason':'PIE已使用同扰动/同背景训练证据；权限分类本身也不是算法贡献。','supported_now':False},
 {'claim':'首次利用缺失掩码或Gaussian收缩','decision':'删除','reason':'均为成熟统计/学习工具；不同实现不自动等于新方法。','supported_now':False},
 {'claim':'本实现提供严格证据权限控制与完整审计','decision':'可作实现/设计贡献，但单独不足以支撑RECOMB方法创新','reason':'严格版本P0/P1/P2已跑；初始输入级版本保留并须区分。','supported_now':True},
 {'claim':'分层残差明显优于强基线','decision':'当前结果不支持','reason':'主任务落后简单收缩加和；相对固定先验四供者差值区间跨零。','supported_now':False},
 {'claim':'实证揭示本任务适用范围和失败边界','decision':'可以如实报告，仍需说明超出先行评测的增量','reason':'数据/评价近邻很强；失败结果本身不构成全新科学发现。','supported_now':True},
 {'claim':'实现完整State复现或超越State系统','decision':'仅称State官方伪bulk骨干的终点适配器比较','reason':'未使用原始SE权重、集合分布训练目标和原文完整协议。','supported_now':False},
 {'claim':'实现器官模拟、真实时间动力学或临床预测','decision':'移出当前标题/摘要/主结果','reason':'现有实测数据与终点评测不能支撑。','supported_now':False},
]

def validate_permissions(level,global_unseen,same_target_response=False,target_context_response=False,target_condition_response=False):
    if level not in {'P0','P1','P2'}:raise ValueError('unknown permission')
    if target_condition_response:raise ValueError('held target-condition outcome leak')
    if target_context_response:raise ValueError('held-out context response leak; only seen training-context responses are allowed')
    if level in {'P0','P1'} and same_target_response:
        raise ValueError('strict permission does not permit held-entity response evidence')
    if global_unseen and (level=='P2' or same_target_response):
        raise ValueError('global unseen targets and P2 are incompatible')
    return True

def equal_donor_mae(frame):
    """Equal gene -> target -> context -> donor; seed is not a donor."""
    required={'donor','context','target','mae'}
    if not required<=set(frame):raise ValueError('missing aggregation columns')
    if not np.isfinite(frame.mae.to_numpy()).all():raise ValueError('nonfinite error')
    units=frame.groupby(['donor','context','target'],sort=True).mae.mean()
    return float(units.groupby(['donor','context']).mean().groupby('donor').mean().mean())

def audit_manifests():
    records=[]
    specs=[(f'training_completion/runs/hierarchical_donor_fold{i}/rows.csv','donor') for i in range(4)]
    specs +=[(f'full_experiments/runs/E2_strict_{p}/rows.csv',p) for p in ('P0','P1','P2')]
    specs +=[('full_experiments/runs/E1_global_target_P1/rows.csv','global'),('full_experiments/runs/E1_external_study_P1/rows.csv','study')]
    strict={}
    for rel,kind in specs:
        p=BASE/rel;m=pd.read_csv(p)
        assert not m['index'].duplicated().any(),rel
        a=m[m.partition.eq('train')];v=m[m.partition.eq('validation')];t=m[m.partition.eq('test')]
        assert min(len(a),len(v),len(t))>0,rel
        assert not set(a['index'])&set(v['index']) and not set(a['index'])&set(t['index']) and not set(v['index'])&set(t['index']),rel
        assert set(v.selection_or_calibration)<={'selection','calibration'},rel
        if kind=='donor':
            assert len(t.donor_id.unique())==1
            assert not set(a.donor_id)&set(t.donor_id) and not set(v.donor_id)&set(t.donor_id)
            assert not set(a.donor_id)&set(v.donor_id)
        if kind in {'P0','P1','P2','global'}:
            assert not set(a.target_contrast)&set(pd.concat([v,t]).target_contrast),rel
        if kind in {'P0','P1','P2'}:strict[kind]=(set(a['index']),set(v['index']),set(t['index']))
        records.append({'path':rel,'sha256':sha(p),'train':len(a),'validation':len(v),'test':len(t),'evidence_only':int(m.partition.eq('evidence_only').sum()),'kind':kind,'row_disjoint':True})
    assert strict['P0']==strict['P1']==strict['P2'],'permission comparison changes supervised support'
    return records

def main():
    ROOT.mkdir(exist_ok=True)
    now=datetime.now(timezone(timedelta(hours=8)))
    original=load('prepared/FREEZE_LOCK.json')
    for rel,value in original['sha256'].items():assert sha(BASE/'prepared'/rel)==value,rel
    paper_hashes={k:{**v,'sha256':sha(BASE/v['local'])} for k,v in PAPERS.items()}
    for v in paper_hashes.values():assert len((BASE/v['local']).read_text(encoding='utf-8'))>50000
    records=audit_manifests()
    checkpoints=[]
    for tree,manifest,col in [('full_experiments','analysis/model_checkpoint_manifest.csv','checkpoint'),('training_completion','analysis/model_manifest.csv','model_path')]:
        table=pd.read_csv(BASE/tree/manifest)
        for row in table.to_dict('records'):
            p=BASE/tree/row[col]
            assert p.is_file(),str(p)
            assert sha(p)==row['sha256'],str(p)
        checkpoints.append({'tree':tree,'verified_model_files':len(table),'all_sha256_match':True})
    protected=load('protected_files_before.json');changed=[]
    for r in protected:
        p=PROJECT/r['path']
        if not p.exists() or sha(p)!=r['sha256']:changed.append(r['path'])
    assert not changed,changed
    protocol={
      'version':'FVcell-supplement-after-literature-v1',
      'status':'specification_frozen_execution_not_claimed',
      'frozen_at':now.isoformat(),
      'prior_exposure':'All earlier test summaries were observed. This supplement is exploratory, not preregistered confirmation. Independent confirmatory data are not yet secured.',
      'original_primary_endpoint':'signed response MAE, equal target within context then equal context within donor then equal donor; retain old FV-versus-concat P2 contrast and original results',
      'label_spaces':{
       'pooled':'published DESeq2 log2FC, published fixed gene universe; inherited feature-curation limitations disclosed',
       'donor':'delta_log2(1+CPM), normalization over original 18129 features, fixed donor_genes.csv; no cross-unit pooling',
       'missingness':'true observed mask only; zeros are actual values, unknown labels not zero; no effect/significance filtering'},
      'frozen_inputs':[{'path':r['path'],'sha256':r['sha256']} for r in records],
      'permissions':{
       'P0':{'baseline_expression':False,'allowed':'fixed target function annotations and literal context metadata','strict_training':'held validation/test target outcomes removed in all contexts'},
       'P1':{'baseline_expression':True,'allowed':'P0 plus allowed NTC profiles; no direct held-target response evidence','strict_training':'same supervised labels as P0'},
       'P2':{'baseline_expression':True,'allowed':'P1 plus same-target response in other training contexts, only frozen evidence_only/train pool','strict_training':'same supervised labels as P0/P1'},
       'global_unseen_target':'P0/P1 only; no same-target response in any source, teacher or auxiliary corpus',
       'target_condition_response':'always forbidden',
       'same_context_other_perturbations':'permitted only when that context belongs to training; for an unseen context no perturbation outcomes are allowed',
       'extra_knowledge':'function annotations separate from measured DepMap/L1000/etc.; any added measured-response source becomes a different information stratum'},
      'matched_comparisons':[
       {'name':'strict information access','tasks':['E2_strict_P0','E2_strict_P1','E2_strict_P2'],'families':['ridge','concat','fv'],'equal':'supervised labels, tests, gene list, search budget; only declared response access differs'},
       {'name':'hierarchical component','tasks':[f'hierarchical_donor_fold{i}' for i in range(4)],'families':['hierarchical','fixed_gaussian','unshrunk_offsets','fv_continued'],'equal':'own originally selected warm-start plus <=1000 corrective steps; frozen selections; no retuning', 'limitation':'warm-start histories and wall time reported; older 3000-step models are not equal-compute continuation controls'},
       {'name':'simple baseline check','families':['zero','training_mean','same_target_mean_if_permitted','shrunk_additive','ridge'],'equal':'identical evaluable rows and genes; additive must have required support; no global-unseen P2 fallback disguised as true same-target evidence'}],
      'new_selection_budget':{'training':0,'new_hyperparameter_trials':0,'checkpoint_selection':'reuse existing validation-selected frozen checkpoints; no test-based switches','random_seeds':[11,29,47],'bootstrap_seed':20261026,'bootstrap_draws':10000},
      'supplemental_metrics':[
       {'metric':'signed_Pearson_delta','rule':'within condition, across fixed genes; center both vectors; zero norm -> NA; report NA count; equal donor/context/target summary'},
       {'metric':'truth_aligned_amplitude','rule':'dot(pred,true)/dot(true,true); denominator<=1e-12 -> NA; diagnostic, not mechanism evidence'},
       {'metric':'direction_accuracy','rule':'evaluate abs(true)>0.1 in label-specific log2 units; report denominator and coverage; predicted zero counts as incorrect'},
       {'metric':'DE_AP','rule':'pooled published adjp<0.05 and abs(log2FC)>0.5; score abs(pred); report AP prevalence and no-positive exclusions; no donor DE labels invented'},
       {'metric':'perturbation_retrieval','rule':'within same donor/context and fixed test panel; cosine of raw delta vector; correct-target rank against all panel targets; average tied rank; normalized inverse rank=(N-rank)/(N-1); invalid zero vector -> NA; report N and exclusions'},
       {'metric':'context_specific_error','rule':'seen-target transfer only: subtract same frozen training-only target shared reference from prediction/truth; unavailable references -> NA; equal donor/context/target summary; do not call this replicate-adjusted CGC recovery'},
       {'metric':'function','rule':'reuse all fixed IL10/IL21 x three RNA contexts, 9 targets, extra-donor same-study panel; no selecting best context; no independent-study claim'}],
      'positive_controls':{
       'role':'metric diagnostic only; target-condition outcomes never enter learned predictions or selection',
       'construction':'donor-response space only: split targeting guides and NTC guides deterministically into disjoint halves within donor/culture/run using SHA256(prefix|guide); aggregate each half with equal-guide weights after original per-row count QC and log2(1+CPM); use replicate-disjoint controls where present',
       'eligibility':'at least 2 distinct targeting guides and 2 distinct matched NTC guides with eligible rows; all relevant blocks nonempty; report exclusions and exact intersection',
       'fallback':'if guide/NTC/replicate metadata do not support independence, mark biological positive control unavailable; never substitute y for its own prediction',
       'comparison':'score zero, training mean, target mean when permitted and positive control on same eligible panel; no retuning gene weights on test',
       'DRF':'only if denominator nonzero and metric directions oriented identically; diagnostic on available positive/negative controls, not clinical calibration',
       'oracle':'an exact-truth oracle may be used for software bounds only, never as biological positive control or reported model'},
      'negative_controls':{
       'zero':'predict delta=0', 'training_mean':'training-only per-gene mean',
       'wrong_target':'20 fixed within-donor/context target derangements, seeded 20261026; reassignment of frozen predictions; no realignment to improve outcomes; aggregation over common supported test panel',
       'wrong_context':'seen-target cross-context derangement only when all candidate contexts have matched support; report blocked cases',
       'role':'diagnose perturbation/context correspondence; not substitutes for strongest predictive baseline'},
      'statistics':{
       'units':'donor is independent biological unit for donor tasks; targets/genes/seeds are not substitute donors; pooled task has no donor generalization CI',
       'seed_summary':'first average each metric across three trained seeds, then compare four held donors; keep seed SD separately',
       'four_donor':'show all four paired differences and mean; 10000-draw donor bootstrap and 16 sign patterns descriptive only because n=4, sign symmetry and overlapping CV fits',
       'multiplicity':'fixed family: hierarchy vs continued FV, fixed Gaussian, unshrunk offsets, on MAE and Pearson (6 exploratory contrasts); Holm across all six, retain raw p; no confirmatory significance claim',
       'replicates':'guide/run/plate splits constrain truth reliability; report repeated measurements nested in donors',
       'confidence_not_posterior':'Gaussian regularization is MAP-like, not full posterior; empirical calibration has no OOD exchangeability guarantee'},
      'independent_confirmation_gate':{
       'current_status':'unmet for harmonized independent primary-T CRISPRi',
       'requirements':'new study, resolved donor/time/intervention/control, full response genes or raw counts, deduplicated subjects/assays, no use for model or metric selection',
       'exclusions':'same-study flow is independent assay/donors, not independent study; cancer line transfer is OOD check, not independent primary-T confirmation; KO and CRISPRi cannot silently be treated as identical',
       'timebox':'metadata feasibility decision by 2026-10-28 18:00+08:00; otherwise scope limitation and no new mass download'},
      'completion_boundary':'This freezes protocol and verifies existing manifests only. No new positive-control biological result, new modern-baseline run, or independent-study replication is declared complete.',
      'do_not_modify':'old source folders, frozen split files, old outcomes/checkpoints, Digital_Organ_Twin, retained documents, old submission baseline'
    }
    protocol_path=ROOT/'evaluation_protocol_frozen.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text(encoding='utf-8'))
        protocol['frozen_at']=previous['frozen_at']
        assert protocol==previous,'A frozen specification changed: create an explicit amendment instead'
    else:write('evaluation_protocol_frozen.json',protocol)
    source_paths=['prepared/FREEZE_LOCK.json','full_experiments/experiment_plan_frozen.json','full_experiments/E2_strict_protocol_amendment.json','full_experiments/E1_external_study_holdout_protocol.json','training_completion/protocol.json','training_completion/identifiability_correction_protocol.json','full_experiments/analysis/all_model_summary.csv','training_completion/analysis/all_results.csv','training_completion/analysis/auxiliary_metrics.csv','training_completion/analysis/joint_functional_results.csv','training_completion/analysis/hierarchical_donor_contrasts.json','full_experiments/analysis/model_checkpoint_manifest.csv','training_completion/analysis/model_manifest.csv','scripts/full_suite_core.py','scripts/completion_core.py']
    frozen_sources={p:sha(BASE/p) for p in source_paths}
    for p in ['full_experiments/prepared/primary_genes.csv','full_experiments/prepared/donor_genes.csv']:frozen_sources[p]=sha(BASE/p)
    for p in PAPERS.values():frozen_sources[p['local']]=sha(BASE/p['local'])
    lock={'version':protocol['version'],'frozen_at':protocol['frozen_at'],'protocol_sha256':sha(protocol_path),'source_sha256':frozen_sources,'old_split_lock_unchanged':True}
    if (ROOT/'FREEZE_LOCK.json').exists():assert load('readiness_20261026/FREEZE_LOCK.json')==lock
    else:write('FREEZE_LOCK.json',lock)
    write('literature_sources.json',paper_hashes);write('claim_ledger.json',CLAIMS)
    with (ROOT/'novelty_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['axis','FVcell_current','B03','B02','PIE']);w.writerows(MATRIX)
    completion=load('training_completion/delivery_status.json')
    deadline_aoe=timezone(timedelta(hours=-12));cst=timezone(timedelta(hours=8))
    deadlines={name:datetime(*parts,tzinfo=deadline_aoe).astimezone(cst).isoformat() for name,parts in [('abstract',(2026,10,28,23,59)),('full_paper',(2026,11,4,23,59))]}
    status={'snapshot_time':now.isoformat(),'priority_1':'complete: source-version-locked comparison and claim ledger; no new method novelty established','priority_2':'protocol and real-manifest preflight complete; biological execution and independent confirmation remain open','manifest_checks':records,'checkpoint_checks':checkpoints,'protected_original_files_checked':len(protected),'protected_original_files_changed':changed,'test_exposure_disclosed':True,'existing_training_status':completion['state'],'current_method_claim_ready_for_RECOMB':False,'abstract_registration':'not registered, confirmed by user','deadlines_Beijing':deadlines,'hours_remaining_at_snapshot':{k:(datetime.fromisoformat(v)-now).total_seconds()/3600 for k,v in deadlines.items()},'source_acquisition':'cached public full texts used; new journal-site requests blocked; versions not upgraded by inference','no_new_training_or_data_download':True}
    write('audit_status.json',status)
    print(json.dumps({k:v for k,v in status.items() if k!='manifest_checks'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
