"""Audit fixed-test outputs and describe the first experiment without retuning."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'training_first'
result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
valid=json.loads((OUT/'input_validation.json').read_text(encoding='utf-8'))
main='shared_plus_context_residual_P2'
frames={k:pd.read_csv(OUT/(k+'_test_rows.csv')).set_index('observation_id') for k in result['test_results']}
reference=frames[main]
assert reference.index.is_unique and reference.target_id.nunique()==len(reference)==2245
assert reference.observed_genes.eq(valid['genes']).all()
for k,f in frames.items():
    assert set(f.index)==set(reference.index)
    f=f.loc[reference.index]
    assert f.context.equals(reference.context)
    actual=float(f.groupby('context').mae.mean().mean())
    assert abs(actual-result['test_results'][k]['macro_context_target_mae'])<1e-12
comparisons={}
for baseline in ('training_global_mean','context_residual_ablation'):
    other=frames[baseline].loc[reference.index]
    rng=np.random.default_rng(20261026);draws=np.zeros(2000)
    for context in sorted(reference.context.unique()):
        mask=reference.context.eq(context)
        delta=(other.loc[mask,'mae']-reference.loc[mask,'mae']).to_numpy()
        draws+=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)/3
    base_mae=result['test_results'][baseline]['macro_context_target_mae'];main_mae=result['test_results'][main]['macro_context_target_mae']
    comparisons[baseline]={'mae_reduction':base_mae-main_mae,'relative_reduction_percent':100*(base_mae-main_mae)/base_mae,'paired_stratified_target_bootstrap_95_percentile_CI':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_replicates':2000,'seed':20261026,'uncertainty_scope':'held-out targets within each fixed context, conditional on this study and pooled donor effects; not independent donor/study generalization'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
root=BASE.parents[2]
protected=json.loads((BASE/'protected_files_before.json').read_text(encoding='utf-8'))
unchanged=all(sha(root/x['path'])==x['sha256'] for x in protected)
assert unchanged
audit={'test_output_checks':'passed','main_test_rows':len(reference),'test_targets':int(reference.target_id.nunique()),'source_gene_count':valid['genes'],'protected_files_checked':len(protected),'protected_files_unchanged':unchanged,'comparisons':comparisons,'fixed_result_sha256':sha(OUT/'results.json'),'input_validation_sha256':sha(OUT/'input_validation.json'),'model_sha256':sha(OUT/'model.npz'),'no_additional_model_fitting_or_test_based_selection':True}
(OUT/'result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
names={'zero_response':'零响应','training_global_mean':'训练集均值','same_target_training_context_mean_P2':'同靶点其他训练条件均值','shared_plus_context_residual_P2':'最小共享响应＋条件残差模型','context_residual_ablation':'去掉条件残差的消融'}
lines=['# 25—26日首轮训练与下载交付','', '日期：2026-10-26。此报告记录真实公开数据的首次实验，不把合成流程测试计入生物学结果。','', '## 下载及核验','', '两份文件全部下载并校验，合计61,352,897,247字节（61.353 GB）。最终大小、分块校验、源对象ETag及完整SHA256记录于download_status.json。没有使用云端或付费资源。','',f'真实效应层为log_fc，单位沿用作者DESeq2 log2FC。完整源文件33,983行×10,282基因；按冻结条件交集保留{valid["rows"]:,}行×{valid["genes"]:,}基因，{valid["finite_observations"]:,}个有限效应值。未按显著性过滤，零效应保留。','', '实际汇总计数有278,684行、18,129基因，CSR格式。靶点列实际为perturbed_gene_id，NTC标签与guide_type=non-targeting一致；对照使用作者keep_for_DE=true的记录，三条件×四供体均有覆盖。完整计数矩阵没有重新做差异表达，本轮使用作者已估计效应。','', '## 冻结任务与真实结果','', '任务为P2：留出靶点与条件的组合，允许使用同靶点在其他训练条件下的实测响应。训练28,876行，验证2,137行，测试2,245行。训练特征中的同靶点均值排除该训练行自身；模型参数来自训练集，16项预设小网格只用验证集选择，固定测试集不用于调参。','', '最小模型是共享响应与训练均值的收缩组合，加上三个条件的残差程序（残差秩至多3）。选中共享权重0.25、条件残差权重0.5。该模型是可运行起点，尚非完整FVcell方法或强基线比较。','', '主指标：逐靶点的全基因有符号log2FC MAE，先在条件内平均，再对三个条件等权平均。越低越好。','', '| 方法 | 测试MAE |','|---|---:|']
for k,v in result['test_results'].items():lines.append(f'| {names[k]} | {v["macro_context_target_mae"]:.6f} |')
lines+=['','与训练集均值相比，最小模型误差相对下降约'+f'{comparisons["training_global_mean"]["relative_reduction_percent"]:.2f}%'+ '；与去掉条件残差的消融相比，额外下降约'+f'{comparisons["context_residual_ablation"]["relative_reduction_percent"]:.2f}%'+'。这说明当前整体收益中收缩效应占有较大作用，不能仅凭整体改进宣称条件建模是主要贡献。','', '## 配对不确定性及限制','']
for k,v in comparisons.items():
    lo,hi=v['paired_stratified_target_bootstrap_95_percentile_CI'];lines.append(f'- 相对{names[k]}，MAE绝对下降{v["mae_reduction"]:.6f}；按条件分层的配对靶点bootstrap 95%区间为[{lo:.6f}, {hi:.6f}]（2,000次，种子20261026）。')
lines+=['', '区间只描述此固定研究和供体合并效应下的留出靶点变动，不代表独立供体、独立研究或器官外推的证据。作者上游基因筛选仍是限制。当前尚未完成同容量MLP、强可复现基线、全局新靶点、独立功能验证及完整方法实验，不能宣称SOTA或投稿证据链已闭合。','',f'真实逐行结果重算一致，{len(protected)}个受保护原文件SHA256均未改变。原Digital_Organ_Twin、旧投稿基线和保留文档保持原位。','', '## 可复查产物','', '- training_first/results.json：固定测试集汇总。','- training_first/*_test_rows.csv：逐条件、逐靶点误差及辅助指标。','- training_first/model.npz：首轮模型参数和训练靶点共享统计。','- training_first/input_validation.json、genes.csv、control_coverage.csv：源校验、基因清单和实际对照覆盖。','- training_first/result_audit.json：配对区间、结果一致性和原文件保护检查。','- prepared/FREEZE_LOCK.json：原冻结协议及划分哈希。','', '下载与首轮训练监控在完成本次交付后停用。后续应优先补齐匹配权限的强基线、模型容量对照和更充分消融，再评估核心方法是否带来额外收益。']
(BASE/'25-26日首轮训练与下载交付.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
