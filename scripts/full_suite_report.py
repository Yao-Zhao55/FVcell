"""Render the completed experiment registry, including negative results and scope gaps."""
from pathlib import Path
import json,sys,html,shutil,subprocess,os
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vendor_plot'))
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'full_experiments/tmp/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'full_experiments';AN=OUT/'analysis'
DEST=OUT/'output/pdf';DEST.mkdir(parents=True,exist_ok=True)
FIG=OUT/'output/figures';FIG.mkdir(parents=True,exist_ok=True)
PDF=DEST/'FVcell_大纲v2模型实验报告_20261026.pdf'
MD=OUT/'FVcell_大纲v2模型实验报告_20261026.md'
audit=json.loads((OUT/'completion_audit.json').read_text())
assert audit['state']=='all_registered_computational_runs_complete',audit.get('missing')
frame=pd.read_csv(AN/'all_model_summary.csv');paired=json.loads((AN/'paired_model_comparisons.json').read_text())
donor=pd.read_csv(AN/'donor_crossvalidation.csv');functional=pd.read_csv(AN/'functional_concordance.csv')
cal=json.loads((AN/'ensemble_calibration.json').read_text());risk=pd.read_csv(AN/'selective_prediction.csv')
cal['shrunk_additive']=json.loads((OUT/'runs/E1_P2_combo/shrunk_additive/result.json').read_text())['calibration']
fa=json.loads((AN/'functional_evidence_audit.json').read_text());dc=json.loads((AN/'donor_level_comparisons.json').read_text())
names={'zero':'零响应','training_mean':'训练均值','same_target_mean':'同靶点跨背景均值','ridge':'岭回归','shared':'共享网络','concat':'同容量条件拼接','independent':'独立背景网络','state_official':'STATE 官方端点适配','fv':'FV 共享＋残差','no_mechanism':'去机制特征与正则','randomized_mechanism':'度保持随机化机制','no_mechanism_regularizer':'仅去机制正则','no_shrink':'去残差收缩项','fixed_state':'固定随机状态映射','onehot_state':'仅背景 one-hot'}
names.update({'shrunk_additive':'收缩均值＋背景加性','shrunk_shared':'收缩同靶点均值'})
def get(task,family):return frame[(frame.task==task)&(frame.family==family)].iloc[0]
def number(v,n=6):return '-' if v is None or pd.isna(v) else f'{v:.{n}f}'
def result_rows(task,families=None):
    f=frame[frame.task==task].copy()
    if families is not None:f=f.set_index('family').loc[families].reset_index()
    return [[names.get(r.family,r.family),number(r.mae),number(r.seed_std),str(int(r.parameters)) if pd.notna(r.parameters) else '经验统计量',str(int(r.rows))] for r in f.itertuples()]
def reduce(task,baseline='concat'):
    a=get(task,'fv').mae;b=get(task,baseline).mae;return (b-a)/b*100
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
# Separate axes and units. All absolute-error axes begin at zero.
fig,ax=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
mainfamilies=['zero','training_mean','ridge','concat','state_official','shrunk_additive','fv']
ax[0].barh(np.arange(len(mainfamilies)),[get('E1_P2_combo',x).mae for x in mainfamilies],color=['#b7c2cd']*5+['#737db8','#147d92'])
ax[0].set_yticks(np.arange(len(mainfamilies)),['Zero','Train mean','Ridge','Concat','STATE adapter','Shrunk additive','FV']);ax[0].invert_yaxis();ax[0].set_xlabel('MAE (published DESeq2 log2FC)');ax[0].set_xlim(left=0);ax[0].set_title('Held combination, P2')
pv=donor.pivot(index='test_donor',columns='family',values='MAE')
for f,col in [('concat','#bc7730'),('state_official','#737db8'),('fv','#147d92')]:ax[1].plot(np.arange(4),pv[f],'o-',label=names[f] if False else {'concat':'Concat','state_official':'STATE adapter','fv':'FV'}[f],color=col)
ax[1].set_xticks(np.arange(4),[x[-4:] for x in pv.index]);ax[1].set_xlabel('Held donor (CE identifier suffix)');ax[1].set_ylabel('MAE (delta log2(CPM+1))');ax[1].set_ylim(bottom=0);ax[1].legend(fontsize=8);ax[1].set_title('Four donor folds, P1')
fig.savefig(FIG/'E1_primary_and_donors.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
for f,col in [('concat','#bc7730'),('fv','#147d92')]:ax[0].plot(range(3),[get('E2_strict_'+p,f).mae for p in ('P0','P1','P2')],'o-',color=col,label=f)
ax[0].set_xticks(range(3),['P0','P1','P2']);ax[0].set_ylim(bottom=0);ax[0].set_ylabel('MAE (DESeq2 log2FC)');ax[0].set_title('Strict evidence permissions');ax[0].legend()
for f,col in [('concat','#bc7730'),('fv','#147d92')]:ax[1].plot(range(2),[get('E3_external_'+p,f).mae for p in ('targeted','non_primary_T')],'o-',color=col,label=f)
ax[1].set_xticks(range(2),['T-cell auxiliary','K562/RPE1 auxiliary']);ax[1].set_ylim(bottom=0);ax[1].set_ylabel('MAE (delta log2(CPM+1))');ax[1].set_title('519 targets; sources confounded');ax[1].legend()
fig.savefig(FIG/'E2_E3_permissions_and_sources.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
for (fam,method),g in risk.groupby(['family','support_score']):ax[0].plot(g.actual_test_retention,g.retained_macro_context_MAE,'o-',label=fam+' / '+('seed SD' if method=='three_seed_disagreement' else 'distance'))
ax[0].set_xlabel('Actual test retention');ax[0].set_ylabel('Retained MAE (DESeq2 log2FC)');ax[0].set_title('Validation-only thresholds');ax[0].legend(fontsize=9)
for f,col,shift in [('fv','#147d92',-.21),('concat','#bc7730',0),('shrunk_additive','#737db8',.21)]:
    ctx=sorted(cal[f]['by_context']);ax[1].bar(np.arange(len(ctx))+shift,[cal[f]['by_context'][c]['simultaneous_response_vector_coverage'] for c in ctx],width=.20,color=col,label=f)
ax[1].axhline(.9,color='#505050',ls='--',lw=1);ax[1].set_xticks(range(len(ctx)),ctx);ax[1].set_ylim(0,1);ax[1].set_ylabel('Vector coverage');ax[1].set_title('Vector coverage (90% target)');ax[1].legend()
fig.savefig(FIG/'E4_risk_and_calibration.png');plt.close(fig)
points=pd.read_csv(AN/'functional_points.csv');fig,axes=plt.subplots(2,3,figsize=(9,5.4),layout='constrained')
for i,e in enumerate(('IL10','IL21')):
    for j,c in enumerate(('Rest','Stim8hr','Stim48hr')):
        g=points[(points.family=='fv')&(points.endpoint==e)&(points.RNA_context==c)];a=axes[i,j];a.scatter(g.RNA_prediction_log2FC,g.protein_delta_percentage_points,color='#147d92',s=22);a.axhline(0,lw=.5,color='grey');a.axvline(0,lw=.5,color='grey');a.set_title(e+' / '+c);a.set_xlabel('Predicted RNA log2FC');a.set_ylabel('Protein fraction delta (pp)')
fig.savefig(FIG/'E5_functional_all_contexts.png');plt.close(fig)
pdfmetrics.registerFont(TTFont('YaHei','C:/Windows/Fonts/msyh.ttc'));pdfmetrics.registerFont(TTFont('YaHeiB','C:/Windows/Fonts/msyhbd.ttc'))
BODY=ParagraphStyle('Body',fontName='YaHei',fontSize=9.2,leading=14.6,spaceAfter=7,wordWrap='CJK',textColor=colors.HexColor('#253544'))
H=ParagraphStyle('H',parent=BODY,fontName='YaHeiB',fontSize=16,leading=23,spaceAfter=14,textColor=colors.HexColor('#126579'))
SMALL=ParagraphStyle('Small',parent=BODY,fontSize=7.5,leading=10.5,spaceAfter=3)
CELL=ParagraphStyle('Cell',parent=SMALL,fontSize=7.4,leading=10.4)
story=[];md=['# FVcell 大纲 v2 模型实验报告','日期：2026-10-26。所有结果为本地真实公开数据训练；未发布或提交。','']
def title(t):story.append(Paragraph(html.escape(t),H));md.extend(['## '+t,''])
def para(t):story.append(Paragraph(html.escape(t),BODY));md.extend([t,''])
def table(headers,data,widths=None):
    contents=[[Paragraph(html.escape(str(v)),CELL) for v in r] for r in [headers]+data]
    tab=Table(contents,colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecf0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#85a5af'))]));story.extend([tab,Spacer(1,9)])
    md.append('| '+' | '.join(headers)+' |');md.append('| '+' | '.join(['---']*len(headers))+' |')
    for row in data:md.append('| '+' | '.join(str(v) for v in row)+' |')
    md.append('')
def picture(name,height=175):
    im=Image(str(FIG/name));ratio=im.imageWidth/im.imageHeight;w=min(485,height*ratio);story.extend([im.__class__(str(FIG/name),width=w,height=w/ratio),Spacer(1,7)]);md.extend(['![实验图](output/figures/'+name+')',''])
def newpage():story.append(PageBreak())
title('FVcell：大纲 v2 模型实验交付')
para('本轮定位：原代 T 细胞体外背景中的终点响应迁移。组织条件、跨器官运输及序贯动态缺少匹配真值，不能作为本次已经验证的贡献。')
para(f"计算登记表全部完成：{audit['task_settings']} 个任务设置、{audit['method_task_settings']} 个方法×任务设置、{audit['neural_checkpoints']} 个三随机种子神经网络检查点；另有 {audit['deterministic_baseline_settings']} 个确定性基线设置。每项结果均可追溯到固定清单、模型和逐条件误差。")
main=get('E1_P2_combo','fv');best=frame[frame.task=='E1_P2_combo'].sort_values('mae').iloc[0]
para(f"主组合任务：FV 的 MAE={main.mae:.6f}；相对同容量条件拼接变化为 {reduce('E1_P2_combo'):+.3f}%（正值表示误差降低）。本任务最低 MAE 为 {names[best.family]} 的 {best.mae:.6f}。必须同时比较简单基线与官方强基线适配，不能只相对零响应报提升。")
globaldiff=reduce('E1_global_target_P1')
para(f"全局新靶点：相对拼接模型变化 {globaldiff:+.3f}%。独立研究留出中，FV={get('E1_external_study_P1','fv').mae:.6f}，零响应={get('E1_external_study_P1','zero').mae:.6f}；复杂模型没有超过该简单基线。")
para('完成的是可执行计算实验，而非大纲全部生物学证据已经成立。E3 仅匹配条件与靶点，未完全匹配细胞数、供者类型与平台；E5 是同研究的独立蛋白读出与额外供者，不是独立研究的功能因果确认。')
table(['证据层','本轮实际范围'],[['E1','组合、全局靶点、四供者轮流留出、整外部研究留出'],['E2','相同监督目标下严格 P0/P1/P2；另保留输入特征消融'],['E3','50%/75%/100% 训练曲线；T 细胞与细胞系辅助数据对比'],['E4','六项重新训练消融；区间覆盖与选择性预测'],['E5','九靶点 IL10/IL21 蛋白阳性比例变化；三 RNA 背景全部报告']], [62,423])
newpage();title('1 数据、单位与固定划分')
table(['数据','实际用途与处理'],[['GWCD4i.DE_stats','33,258 个纳入条件，11,086 个靶点，10,282 个响应基因。直接使用完整 DESeq2 log2FC，未按显著性截断。'],['GWCD4i.pseudobulk_merged','278,684 个来源行。独立计数重建 129,671 个供者×背景×靶点响应，18,129 个参考特征（含外源 PuroR），缺失组不补零。'],['Replogle K562 / RPE1','310,385 / 247,914 个细胞；按靶点和对照汇总原始 UMI 计数。共同任务 519 靶点、6,795 基因。'],['蛋白流式','103 个原始测量行，13 个供者，9 靶点；供者和批次内 NTC 对照。'],['机制先验','2026-10-26 Reactome 人类基因-通路成员关系快照；保留下载与 SHA256 记录。']], [125,360])
para('主任务组合划分：训练 28,876、验证 2,137、测试 2,245 行。全局靶点划分：训练 19,572、验证 6,981、测试 6,705 行；同一靶点所有背景归入同一组。验证靶点按固定 SHA256 分成模型选择与校准两半，校准不参与早停。')
para('供者分析统一使用 delta log2(CPM+1)：逐靶点/供者/背景汇总计数，匹配同供者/背景 NTC；文库总数用完整参考特征。仅使用 n_cells>=5、非负有限计数及正的文库总数；不使用跨供者差异结果决定的 keep_for_DE、effective_guides 或 test_genes 过滤。')
para('两类响应的估计量、基因集和尺度不同，分别排名，不汇总为一个总 MAE。外部细胞系与原代 T 细胞的细胞数、实验类型、时间与保留基因文库仍存在差异。')
para('作者上游靶点整理及 pooled DE 特征选择仍是继承限制。本轮重做的供者标签不等于彻底消除原研究选择偏差。旧混合来源 NPZ、显著性筛选表、旧投稿权重均未充当真值或教师。')
newpage();title('2 E1：未见背景×干预组合')
table(['模型','MAE','种子 SD','参数/统计量','测试行'],result_rows('E1_P2_combo',mainfamilies[:2]+['same_target_mean','shrunk_shared','shrunk_additive','ridge','shared','concat','independent','state_official','fv']),[151,78,77,92,87])
para(f"既有简单方法在相同验证选择半集重跑：收缩均值＋背景加性 MAE={get('E1_P2_combo','shrunk_additive').mae:.6f}，FV={get('E1_P2_combo','fv').mae:.6f}。因此即使 FV 相对拼接有小幅收益，也不能宣称复杂神经结构超过最强简单基线。两组权重网格沿用首次训练，校准半集和测试不参与选择。")
contrast=paired['E1_P2_combo'];ci=contrast['target_cluster_bootstrap_95_CI']
para(f"预设主对照为 FV 对拼接模型。拼接 MAE 减 FV MAE={contrast['absolute_reduction']:.8f}，条件于当前队列的靶点簇 bootstrap 95% 区间 [{ci[0]:.8f}, {ci[1]:.8f}]。三种子先对逐条件损失求均值，区别于集成预测损失。此区间不代表新供者总体的不确定性；多项对比仅作描述，不作未经校正的显著性宣称。")
aux=[]
for f in ['zero','training_mean','concat','state_official','fv']:
    s=json.loads((OUT/'runs/E1_P2_combo'/f/'summary.json').read_text());runs=s.get('runs',[s]);m=[]
    for r in runs:
        a=r.get('auxiliary_metrics',{});t=r.get('test',{});m.append([t.get('signed_pearson'),a.get('macro_context_AP')])
    # The stored metric key is checked rather than invented when unavailable.
    vals=[r.get('auxiliary_metrics',{}).get('macro_context_AP') for r in runs];vals=[v for v in vals if v is not None]
    aux.append([names[f],number(np.mean(vals)) if vals else '未计算（保留 MAE 基线）'])
table(['模型','DEG 排序 AP（无正例行排除）'],aux,[240,245])
para('AP 以全部有限调整 p 值、FDR<=0.1 定义正例，排序分数为预测效应绝对值；不是 PR 曲线梯形面积。无正例行记为未定义而非零。方向准确率采用真值绝对效应>0.1 的死区；相关性对常数向量记未定义。完整辅助指标和扰动检索结果保存在各检查点结果中。')
para('这轮组合测试在先前最小加性模型中已被查看，因此扩展比较是探索性结果，不能称为从未查看的确认性测试。')
newpage();title('3 E1：供者、全局靶点与研究迁移')
picture('E1_primary_and_donors.png',185)
donorrow=[]
for d,r in pv.iterrows():donorrow.append([d]+[number(r[f]) for f in ['ridge','shared','concat','state_official','fv']])
donorrow.append(['四供者等权均值']+[number(pv[f].mean()) for f in ['ridge','shared','concat','state_official','fv']])
table(['测试供者','岭回归','共享','拼接','STATE 适配','FV'],donorrow,[105,76,76,76,76,76])
para('每折以两个供者训练、一个供者选择/校准、一个供者测试；四折训练集重叠。独立生物重复仅 4 个，不能把上万靶点或三随机种子当作独立供者。配对 16 种符号翻转检验与描述性供者 bootstrap 均单列保存。')
if 'concat' in dc:para(f"FV 对拼接的四供者平均 MAE 降低 {dc['concat']['paired_held_donor_MAE_reduction']:.6f}；四供者符号翻转双侧 p={dc['concat']['exact_16_sign_flip_two_sided_p']:.3f}。这不是大人群泛化保证。")
table(['任务 / 单位','拼接','STATE 适配','FV','零响应'],[['全局新靶点 / DESeq2',number(get('E1_global_target_P1','concat').mae),number(get('E1_global_target_P1','state_official').mae),number(get('E1_global_target_P1','fv').mae),'另见完整基线'],['整外部研究 / delta log2CPM',number(get('E1_external_study_P1','concat').mae),number(get('E1_external_study_P1','state_official').mae),number(get('E1_external_study_P1','fv').mae),number(get('E1_external_study_P1','zero').mae)]],[177,77,77,77,77])
para('整研究留出：只用训练 T 供者效应训练，以 T 供者验证选择；K562/RPE1 所有扰动响应全部作为测试，P1 仅允许它们未扰动对照。两种细胞系属于同一外部研究，不能计为两个独立研究。跨研究任务虽胜过部分学习模型，却未胜过零响应，泛化结论必须受此限制。')
newpage();title('4 E2：严格区分信息权限')
rows=[]
for p in ('P0','P1','P2'):
    for f in ('ridge','concat','fv'):rows.append([p,names[f],number(get('E2_strict_'+p,f).mae),number(get('E2_strict_'+p,f).seed_std)])
table(['权限','模型','MAE','种子 SD'],rows,[58,227,100,100])
para('严格 E2：所有权限使用相同 20,112 个监督训练行，移除验证/测试靶点在其他背景中的效应监督。P0 不输入目标背景观测；P1 允许目标背景 NTC；P2 再允许原训练来源的 8,764 个其他背景同靶点效应行作为 evidence_only 特征，不进入监督梯度和标签归一化统计。待预测条件的真值从不作为输入。')
para(f"FV 从严格 P1 到 P2 的 MAE 降低 {(get('E2_strict_P1','fv').mae-get('E2_strict_P2','fv').mae)/get('E2_strict_P1','fv').mae*100:.3f}%；其中应分别归因于新增证据与模型结构。相同权限内，严格 P2 的 FV 相对拼接降低 {reduce('E2_strict_P2'):.3f}%。")
para('原始 E2_P0/P1_combo 仅删除直接输入特征，训练目标仍含部分同靶点其他背景效应，因此只保留为操作性特征消融，不将其宣称为全来源严格撤权。所有严格协议补充均在读取其测试结果前冻结。')
para('这里 P0 的含义限定为不输入目标未扰动表达；同背景其他靶点仍可在监督训练中出现。因此即使撤去了同靶点监督，也不是整个背景从所有训练来源消失的归纳外推，不等同于未见组织或未见刺激背景的严格零样本测试。')
picture('E2_E3_permissions_and_sources.png',165)
newpage();title('5 E3：规模与辅助数据来源')
scale=[]
for label,t in [('50%','E3_internal_50'),('75%','E3_internal_75'),('100%','E1_P2_combo')]:
    m=pd.read_csv(OUT/'runs'/t/'rows.csv');scale.append([label,str(int(m.partition.eq('train').sum())),number(get(t,'fv').mae),number(get(t,'fv').seed_std)])
table(['训练比例','监督条件行','FV MAE','种子 SD'],scale,[100,145,120,120])
para('规模曲线保持相同测试组合与同靶点必需辅助证据，只按固定靶点哈希减少其余训练行。因此是本任务的受控监督量曲线，不是任意规模预训练规律，也不估计一亿细胞的收益。')
external=[]
for regime,label in [('targeted','原代 T 细胞辅助'),('non_primary_T','K562/RPE1 辅助')]:
    for f in ('ridge','concat','fv'):external.append([label,names[f],number(get('E3_external_'+regime,f).mae)])
table(['辅助来源','模型','MAE'],external,[175,190,120])
para('共同数据范围固定为 519 个靶点和 6,795 个基因；每组训练 3,114 个条件行，其中辅助来源各 1,038 行，验证和测试各 1,557 行。模型家族、调参规则及输出空间相同。')
para('仍未匹配：各条件细胞数、健康原代供者与癌细胞系的生物单位、实验室/引导 RNA/刺激时间及文库基因保留策略。目标 T 辅助数据与非 T 辅助数据效果差异只能作为有限探索，不能回答大纲中完全匹配广域预训练数据的严格因果比较。')
para('大规模状态图谱、Tahoe 或预训练编码器未被混入本次监督链路，避免在预训练暴露无法确认时宣称全局新靶点。相应广域预训练实验仍需另行构建来源与计算预算匹配的数据。')
newpage();title('6 E4：机制、残差和状态消融')
ab=['fv','concat','no_mechanism','randomized_mechanism','no_mechanism_regularizer','no_shrink','fixed_state','onehot_state']
table(['模型 / 消融','MAE','种子 SD','参数/统计量','测试行'],result_rows('E1_P2_combo',ab),[151,78,77,92,87])
para('每项消融重新训练三随机种子，学习率与早停规则相同。真实/随机先验使用同一 32 维通路哈希映射；随机化采用保持基因和通路度数的二部图交换，不接触响应标签。未映射基因使用固定 ID 随机向量。')
para('共享模型、拼接模型与独立背景模型参数量接近（主任务约 67 万至 69 万，差异小于 2%）。STATE 适配约 78.8 万，比 FV 约多 16%；岭回归参数更多，不能声称所有方法严格同容量。')
para('no_shrink 只移除残差潜变量 L2 项，保留 AdamW 权重衰减；不等于移除全部正则。fixed_state 是固定随机线性映射，不是预训练 STATE-SE。onehot_state 删除对照表达状态特征，只保留背景标签。')
para('当前实现是共享/条件低秩残差与潜变量 L2 收缩，不是完整层次贝叶斯模型；机制项为几何一致性正则，未引入不具备证据的有向激活/抑制符号。不能把候选大纲中更强的层次收缩或机制解释写成已实现。')
para(f"反证同样保留：仅背景 one-hot 的 MAE={get('E1_P2_combo','onehot_state').mae:.6f}，低于完整状态模型 {get('E1_P2_combo','fv').mae:.6f}；当前组合任务不能据此宣称加入基线表达状态带来收益。去机制正则与去残差收缩的差异也很小，不能仅凭结构存在就声称组件有效。")
newpage();title('7 E4：区间覆盖与选择性预测')
picture('E4_risk_and_calibration.png',195)
rr=[]
for f in ('concat','fv','shrunk_additive'):
    for c,v in cal[f]['by_context'].items():rr.append([names[f],c,str(v['calibration_rows']),number(v['simultaneous_response_vector_coverage'],3),number(v['coordinate_coverage'],6),number(v['mean_interval_width'],3)])
table(['模型','背景','校准行','整向量覆盖','坐标覆盖','平均宽度'],rr,[100,80,60,85,80,80])
para('区间以训练基因标准差缩放，校准每个响应向量最大标准化绝对误差的 90% 次序统计量。整向量覆盖和坐标覆盖分开；区间过宽也可取得较高覆盖，必须同时报告宽度。跨靶点依赖及供者偏移使交换性未经证明，故只报告经验覆盖。')
para('完整 FV 的区间平均宽度约 13.5-14.6 log2FC，远大于平均绝对误差，实用分辨率有限；简单加性基线区间还更窄。因此约 90% 的整向量经验覆盖不能单独支持高效、可信的不确定性结论。')
para('选择性预测比较三种子分歧和训练特征中心距离。20%、40%、60%、80%、100% 的阈值仅由模型选择验证子集确定；图中横轴是实际测试保留率，不以测试误差选择阈值。风险降低并不自动说明不确定性已校准，须对照简单距离筛选和样本组成变化。')
newpage();title('8 E5：独立蛋白读出与严格靶点留出')
para(f"独立测量审计：{fa['flow_rows']} 个来源行，{fa['matched_perturbed_measurements']} 个匹配对照后的扰动测量，{fa['flow_donors']} 个供者、{fa['gene_targets']} 个预选靶点；解析到 {fa['functional_donor_ids_resolved']} 个供者 ID，与主训练供者重叠为 {fa['donor_overlap_with_primary']}。统计单元先对同靶点/供者的引导 RNA 求均值，再汇总供者，不能按原始行数放大独立样本量。")
para('所有九靶点的 RNA 效应在全部背景中均从训练与选择剔除；新版本同时用全局靶点划分的验证集，避免用已见靶点选择模型来评价新功能靶点。另保留旧验证版本作探索对照。流式标签从不参与训练、调参或最佳 RNA 背景选择。')
ff=[]
for r in functional.itertuples():
    if getattr(r,'status',None) is not None and pd.notna(r.status):continue
    ff.append([names[r.family],r.RNA_context,r.endpoint,str(int(r.targets)),number(r.spearman_RNA_vs_protein_fraction_delta,3),'['+number(r.two_way_target_and_functional_donor_bootstrap_95CI_low,3)+', '+number(r.CI_high,3)+']',number(r.direction_agreement,3)])
table(['模型','RNA 背景','终点','靶点','Spearman','双向重采样 95% CI','方向'],ff,[90,72,43,34,62,125,59])
para('终点是 IL10/IL21 蛋白阳性细胞比例相对同供者/批次 NTC 的百分点变化，RNA 预测为 log2FC。这里只测跨测量排序与方向一致性，不是蛋白绝对量回归；时间与培养背景未严格匹配，同一研究预选九靶点亦有选择偏差。区间同时重采样靶点与功能供者，不能据小样本相关性宣称全基因组功能可预测。')
newpage();title('9 E5：全部背景散点与组织解释边界')
picture('E5_functional_all_contexts.png',300)
para('上图完整呈现两个预设终点与三个 RNA 背景，没有挑选最有利背景。离群靶点、小样本与 RNA/蛋白解耦应直接保留；阴性相关或方向不一致同样构成本次验证结果。')
demo=json.loads((AN/'composition_identity_demo.json').read_text())
para(f"组织组成分解只完成数学一致性验证：10,000 次合成抽样，线性尺度恒等式最大误差 {demo['max_absolute_identity_error']:.2e}。它验证计算公式，不能作为生物学预测精度。相同混合均值变化可由细胞内响应或组成变化产生，只有混合均值不能唯一分离。")
para('没有真实匹配的器官干预/组成或纵向流量观测，因此不报告跨器官迁移、疾病逆转、虚拟患者疗效和序贯干预正确性。本轮标题宜采用 Primary T Cells，而非暗示组织或淋巴器官泛化已经验证。')
newpage();title('10 实现、成本与复现实验')
para('FV 以 65 维动作特征与 35 维状态输入构建 32 维共享程序和低秩交互残差，FP32 训练。P2 动作特征增加同靶点其他允许训练条件的响应投影；训练特征去除当前响应，P0/P1 不输入该同靶点效应投影。只有训练监督标签用于输出均值/尺度与特征归一化。')
para('未观测条件整行从监督中排除，纳入向量须全部有限；当前管线会拒绝任意基因级 NaN，不会把它们补零。尚未实现任意基因级缺失掩码的多源训练，不能将行级不完整监督写成已经支持全部基因级缺失模式。')
para('神经网络种子 11/29/47，AdamW，SmoothL1(beta=0.1)，学习率候选 0.001/0.0003、最多 3,000 步，每 250 步验证，耐心 4、最小改进 1e-6；第一种子选择学习率，其余复用。岭回归 alpha=0.1/10/1000，同一验证选择规则。校准半集不参与这些决定。')
para('STATE 基线复用官方 PseudobulkPerturbationModel 的实际 forward：两层 64 维 GPT2、四注意力头、聚合终点序列长度 1；保留签名响应输出，并统一外部监督损失。这是可运行的官方端点适配，不是原论文单细胞 OT、预训练 SE 或完整预训练权重的复现。其 CC-BY-NC-SA-4.0 许可证与来源快照保留，GEARS 未被伪造为适配后的得分。')
fitsecs=frame.total_fit_seconds.sum()
para(f"所有选中模型拟合时间合计 {fitsecs/3600:.2f} 小时（求和，不是独占 GPU 小时）。该数字不含被舍弃的学习率候选、部分缓存构建和完整评估；家族墙钟耗时包含调参/恢复与并发资源争用。早期显存峰值是进程累计、后期才重置单次峰值，不能混成统一模型显存排名。")
para('机器：RTX 5060 Laptop 8 GB，24 逻辑 CPU、约 34 GB RAM。训练仅缓存训练行到 GPU，显存不足自动流式读取。大预测缓存只保留校准/功能分析需要的项目；源数据、全部模型、逐条件损失与校验清单保留。来源下载与额外依赖总量仍低于用户授权 100 GB，未使用云端或购买资源。')
table(['复核产物','相对 full_experiments 的位置'],[['总指标与逐任务明细','analysis/all_model_summary.csv；runs/<task>/<family>/<seed>/'],['成对与供者统计','analysis/paired_model_comparisons.json；donor_level_comparisons.json'],['校准与功能','analysis/ensemble_calibration.json；selective_prediction.csv；functional_concordance.csv'],['模型 SHA256','analysis/model_checkpoint_manifest.csv'],['完成与环境审计','completion_audit.json；software_and_code_manifest.json'],['固定设计及来源','experiment_plan_frozen.json；各补充协议；prepared 与 external_data 的 provenance']], [140,345])
newpage();title('11 结论、未完成证据与来源')
para('本轮应按真实结果写作：共享/残差结构在部分允许同靶点证据和供者条件下具有局部误差收益，但全局新靶点差异微小，整研究外推未超过零响应；因此不能写成通用虚拟细胞或稳定跨组织 SOTA。不同证据权限及负结果比单一提升百分比更重要。')
table(['仍缺少的证据','对论文结论的约束'],[['完全匹配的广域预训练比较','E3 未满足细胞数、供者、平台、计算量的全匹配，保留混杂。'],['独立原代 T 细胞 CRISPRi 完整研究测试','外部 Replogle 是非原代 T OOD。GSE278572 有原始计数，但为 KO/背景差异，未构建可比完整 gold；不能说所有原始数据不存在。'],['完整层次收缩/预训练 SE 对照','当前仅潜变量 L2 收缩和随机固定状态映射。若投稿强调更强实现，须补齐后重做。'],['独立组织干预与纵向数据','数学分解及合成一致性不能替代真实器官/动态验证。'],['确认性复现与更大供者队列','组合测试已查看；四供者和九预选功能靶点不支持广泛人群结论。']], [155,330])
para(f"文件保护审计通过：{audit['protected_original_files_unchanged']} 个原文件 SHA256 未改变，原始 FREEZE_LOCK 各固定文件哈希一致；新增产物全部位于 day25_26。Digital_Organ_Twin、旧投稿基线与保留文稿保持原位。没有因结果不利而删除模型或改动冻结的训练/验证/测试划分。")
for t,url in [('主数据 GSE314342','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342'),('作者数据说明与功能元数据','https://github.com/emdann/GWT_perturbseq_analysis_2025'),('STATE 官方代码与许可证','https://github.com/ArcInstitute/state'),('scPerturb 原始公开处理文件记录','https://zenodo.org/records/13350497'),('Reactome 基因-通路注释','https://reactome.org/download/current/Ensembl2Reactome.txt')]:
    story.append(Paragraph(html.escape(t)+'<br/><link href="'+url+'" color="#126579">'+html.escape(url)+'</link>',SMALL));md.extend([f'- [{t}]({url})',''])
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#cad7dd'));c.line(42,39,553,39);c.setFont('YaHei',7);c.setFillColor(colors.HexColor('#607680'));c.drawString(42,26,'FVcell · 2026-10-26 · 真实数据终点响应实验');c.drawRightString(553,26,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=52,title='FVcell 大纲v2模型实验报告',author='FVcell project')
doc.build(story,onFirstPage=footer,onLaterPages=footer);MD.write_text('\n'.join(md),encoding='utf-8')
import importlib.metadata as metadata
(OUT/'report_dependencies.json').write_text(json.dumps({n:metadata.version(n) for n in ['matplotlib','contourpy','fonttools','kiwisolver','pyparsing','reportlab','pypdf','Pillow']},indent=2),encoding='utf8')
from pypdf import PdfReader
reader=PdfReader(str(PDF));assert len(reader.pages)>=10
text='\n'.join(p.extract_text() or '' for p in reader.pages);assert 'E1' in text and '阴性' in text and '完整层次' in text
print(json.dumps({'pdf':str(PDF),'markdown':str(MD),'pages':len(reader.pages)},ensure_ascii=False))
