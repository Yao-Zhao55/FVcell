"""Create a separate addendum; the earlier report stays intact."""
from pathlib import Path
import os,sys,json,html,itertools
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'training_completion';AN=ROOT/'analysis';OUT=ROOT/'output';FIG=OUT/'figures';DEST=OUT/'pdf'
FIG.mkdir(parents=True,exist_ok=True);DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'vendor_plot'));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer,Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
AUD=json.loads((ROOT/'audit.json').read_text());assert AUD['state']=='all_registered_additional_runs_complete'
R=pd.read_csv(AN/'all_results.csv');D=pd.read_csv(AN/'donor_results.csv');M=pd.read_csv(AN/'missingness_results.csv');C=json.loads((AN/'hierarchical_donor_contrasts.json').read_text())
N={'hierarchical':'分层高斯收缩 MAP','fixed_gaussian':'固定方差高斯收缩','unshrunk_offsets':'去随机效应收缩','previous_fv':'原 FV','previous_concat':'原条件拼接','previous_state_official':'原 STATE 端点适配','concat':'条件拼接','fv':'FV','state_official':'STATE 端点适配','zero':'零响应','training_mean':'训练均值','ridge':'岭回归'}
N.update({'hierarchical_pooled':'联合分层 FV','fv_continued':'同预算续训 FV'})
def fmt(x,n=6):return '-' if pd.isna(x) else f'{x:.{n}f}'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
P=D.pivot(index='fold',columns='family',values='mae');fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
for family,label,color in [('fixed_gaussian','Fixed Gaussian','#bb843c'),('hierarchical','Hierarchical MAP','#147d92'),('unshrunk_offsets','Unshrunk offsets','#8475ac'),('fv_continued','FV continuation','#58995c'),('previous_fv','Previous FV','#99a3ad')]:ax.plot(P.index,P[family],'o-',label=label,color=color)
ax.set_xticks(range(4),['Fold 0','Fold 1','Fold 2','Fold 3']);ax.set_ylim(bottom=0);ax.set_ylabel('MAE (delta log2(CPM+1))');ax.legend(fontsize=9);fig.savefig(FIG/'hierarchical_donors.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
for j,p in enumerate(('P0','P1')):
    for family,color,label in [('zero','#aaaaaa','Zero'),('concat','#bb843c','Concat'),('fv','#147d92','FV'),('state_official','#8475ac','STATE adapter')]:
        ys=[R[R.task.eq('whole_context_'+c+'_'+p)&R.family.eq(family)].iloc[0].mae for c in ('Rest','Stim8hr','Stim48hr')];axes[j].plot(range(3),ys,'o-',label=label,color=color)
    axes[j].set_xticks(range(3),['Rest','Stim8hr','Stim48hr']);axes[j].set_ylim(bottom=0);axes[j].set_title(p+' / unseen context + target');axes[j].set_xlabel('Entire held context');axes[j].set_ylabel('MAE (published DESeq2 log2FC)');axes[j].legend(fontsize=8)
fig.savefig(FIG/'whole_context_permissions.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
for family,label,color in [('concat','Concat','#bb843c'),('fv','FV','#99a3ad'),('state_official','STATE adapter','#8475ac'),('hierarchical_pooled','Joint hierarchical FV','#147d92'),('fv_continued','FV continuation','#58995c')]:
    f=M[M.family.eq(family)].sort_values('observed_fraction');ax.errorbar(f.observed_fraction,f.mae,yerr=f.seed_sd,fmt='o-',label=label,color=color,capsize=3)
ax.set_xticks([.3,.7,1.],['30%','70%','100%']);ax.set_xlabel('Observed training labels');ax.set_ylabel('MAE (published DESeq2 log2FC)');ax.set_ylim(0,float((M.mae+M.seed_sd).max())*1.12);ax.legend();ax.set_title('Nested deletion; complete response reference');fig.savefig(FIG/'missingness.png');plt.close(fig)
pdfmetrics.registerFont(TTFont('YH','C:/Windows/Fonts/msyh.ttc'));pdfmetrics.registerFont(TTFont('YHB','C:/Windows/Fonts/msyhbd.ttc'))
B=ParagraphStyle('B',fontName='YH',fontSize=9.5,leading=15.4,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#253544'))
H=ParagraphStyle('H',parent=B,fontName='YHB',fontSize=16,leading=24,spaceAfter=13,textColor=colors.HexColor('#126579'))
S=ParagraphStyle('S',parent=B,fontSize=7.8,leading=11,spaceAfter=2)
story=[];md=['# 大纲 v2 训练补完与稿件准入核验','日期：2026-10-26。保留前一轮报告和全部模型，本报告是独立补充。','']
def title(t):story.append(Paragraph(html.escape(t),H));md.extend(['## '+t,''])
def p(t):story.append(Paragraph(html.escape(t),B));md.extend([t,''])
def table(header,data,widths):
    tab=Table([[Paragraph(html.escape(str(v)),S) for v in row] for row in [header]+data],colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecf0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f6f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([tab,Spacer(1,9)]);md.extend(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']);md.extend('| '+' | '.join(map(str,row))+' |' for row in data);md.append('')
def image(name,height=185):
    im=Image(str(FIG/name));w=min(485,height*im.imageWidth/im.imageHeight);story.extend([Image(str(FIG/name),width=w,height=w*im.imageHeight/im.imageWidth),Spacer(1,8)]);md.extend(['![实验图](output/figures/'+name+')',''])
def page():story.append(PageBreak())
title('大纲 v2：训练补完与稿件准入核验')
p('核查结论：此前“已完成”仅覆盖当时登记的计算批次，不能等同于大纲全部实现。本轮补做分层收缩、基因级缺失标签损失和整个刺激背景留出三项。它们分别回答收缩是否有效、不完整标签能否安全训练，以及未见背景的预测是否成立。')
p(f"本轮已完成 {AUD['new_tasks']} 个新增任务、{AUD['new_method_task_settings']} 组方法×任务比较、{AUD['new_neural_checkpoints']} 个三种子神经网络检查点与 {AUD['new_deterministic_baseline_settings']} 个确定性基线。补充管线 14 项检查通过，其中 8 项是新检查、6 项复核既有规则；不把合成冒烟测试计入生物学证据。")
p('结果结论：联合分层主任务 MAE=0.173392，同预算续训 FV=0.173386，简单加性基线=0.172030。分层并未超过这些对照；四供者相对固定收缩的平均变化仅约 4.15e-7，描述性区间跨零。六组分层模型蛋白读出相关区间也全部跨零。训练缺口已补，不代表方法优势已成立。')
table(['大纲要求','此前缺口','本轮完成'],[['4.2 分层收缩','原模型仅残差潜变量 L2','四供者留出：部分池化高斯 MAP / 固定高斯 / 去额外收缩，保持骨干一致'],['4.3 观测掩码','此前只能排除整条件，遇基因级缺失即报错','观测条目损失和训练内统计；70%/30% 嵌套缺失，完整实测测试真值'],['6.1 P0/P1','此前 P0 只删除基线表达输入','三背景分别全部留出，并同时留出全局新靶点；P0/P1 分开与强基线比较']], [92,150,243])
p('所有新增实验设计固定后再读取新测试结果。早期任务的测试摘要已经看过，因此本轮是补充探索性实验，不包装为全新确认性试验，也不按测试结果选择结构、超参数、靶点或最佳背景。')
p('是否适合顶会投稿仍取决于实际增益、可复现性和生物学证据。增加训练数量不代表算法已经成立；此前主任务的简单加性基线更好这一结果继续保留。')
page();title('1 分层收缩：实现与四供者对照')
p('共同骨干增加 32 维背景随机效应 b_c 与供者随机效应 v_d。残差为 A(u) ⊙ tanh(H(z_control)+b_c+v_d)，随机效应施加零和约束。共享 H 不允许 one-hot ID 旁路，所有背景 ID 系数仅由受先验约束的 b_c 表达；未见供者 v_d 精确为零。')
p('分层版令 b_c 的高斯尺度随背景变化，其 log 尺度向一个全局 log 尺度部分池化；供者尺度也由训练估计并加固定超先验。先验密度包含零和子空间的正确归一化项。固定版保持相同随机效应和固定尺度；去收缩版只删除额外随机效应先验，原残差 L2、机制项和 AdamW 仍保留。')
p('第一阶段学习率 0.001/0.0003，分层/固定版收缩系数 0.0003/0.003，仅以选择验证半集确定。审查后将 one-hot 系数等价移入 b_c 并阻断自由路径，从权重重置优化器再训练最多 1000 步，保留验证最优更新后模型；FV 控制也获相同预算。旧模型仅作训练内 warm start 并归档，不计最终修正检查点。分层与固定高斯预算相同；去额外收缩候选数较少，不称全部调参成本相等。')
data=[]
for i,r in P.iterrows():data.append([str(i)]+[fmt(r[f]) for f in ('fixed_gaussian','hierarchical','unshrunk_offsets','fv_continued','previous_fv','previous_state_official')])
data.append(['供者均值']+[fmt(P[f].mean()) for f in ('fixed_gaussian','hierarchical','unshrunk_offsets','fv_continued','previous_fv','previous_state_official')])
table(['供者折','固定高斯','分层 MAP','去额外收缩','续训 FV','原 FV','原 STATE'],data,[50,73,73,73,73,73,70])
image('hierarchical_donors.png',170)
c=C['fixed_gaussian'];bounds=c['descriptive_donor_bootstrap_95CI'];p(f"固定高斯 MAE 减分层 MAE 的四供者平均值 {c['absolute_MAE_reduction']:.2e}，描述性重采样区间 [{bounds[0]:.2e}, {bounds[1]:.2e}]；16 种符号翻转枚举 p={c['exact_sign_flip_p']:.3f}。枚举依赖符号对称假设，训练重叠时仅作描述性检验。种子在供者内平均，不能替代独立供者；n=4 不支持大人群效应保证。")
p('移除的是 one-hot 与随机效应之间的特定重复路径，并不证明整个非线性潜变量模型可辨识。供者任务使用全部 18,129 个参考特征（含外源 PuroR），单位为 delta log2(CPM+1)，与 pooled DESeq2 的 10,282 基因任务分开排名。')
page();title('2 整背景留出：P0 与 P1')
p('训练和选择阶段完全删除一个刺激背景的效应行；测试仅使用原冻结全局测试靶点在该背景中的响应。这些靶点也从其他背景监督中全部剔除。原全局验证靶点只在其余两个背景选择/校准，测试背景的响应从不用于早停或调参。')
p('P0 不输入任何目标未扰动表达；P1 仅允许该背景 NTC。两者采用同一训练监督与靶点/基因集合。刺激元数据用是否刺激、log1p(小时)/log49 和常数表示，避免靠未经训练的背景 ID 参数预测未知背景；此编码在新测试前固定。没有目标域编码器更新或测试效应预训练。')
data=[]
for ctx in ('Rest','Stim8hr','Stim48hr'):
    for fam in ('zero','training_mean','ridge','concat','fv','state_official'):
        vals=[R[R.task.eq('whole_context_'+ctx+'_'+q)&R.family.eq(fam)].iloc[0] for q in ('P0','P1')];data.append([ctx,N[fam],fmt(vals[0].mae),fmt(vals[1].mae)])
table(['留出背景','方法','P0 MAE','P1 MAE'],data,[95,184,103,103])
p('这是训练与显式输入层面的“未见背景＋未见靶点”联合外推。主 DESeq2 文件仍继承作者跨样本估计和 pooled 特征选择，不能称全预处理从未接触测试背景。其他四供者实验证据使用独立计数重建及完整参考基因集，二者分开解释。')
p('本节为基础 FV 与强基线的领域审计，不把这组分数冒作联合分层模型在全新背景上的性能。分层版本的主任务、新靶点、供者及功能实验在其他节单列；已有背景的新组合与全新背景不是同一问题。')
page();title('3 未见背景结果的解释边界')
image('whole_context_permissions.png',205)
for ctx in ('Rest','Stim8hr','Stim48hr'):
    text=[]
    for perm in ('P0','P1'):
        f=R[R.task.eq('whole_context_'+ctx+'_'+perm)];fv=float(f[f.family.eq('fv')].mae.iloc[0]);zero=float(f[f.family.eq('zero')].mae.iloc[0]);best=f.sort_values('mae').iloc[0];text.append(f'{perm}: FV={fv:.6f}, 零响应={zero:.6f}, 本组最低为 {N[best.family]}={best.mae:.6f}')
    p(ctx+'：'+'；'.join(text)+'。')
p('应逐背景报告失败，不用三个培养条件代表多个器官，也不挑选最有利背景写入摘要。Rest 与刺激后两个时点反映体外培养差异；是否能迁移至血液、淋巴结、脾脏等组织，仍需具有干预观测的真实组织测试。')
p('官方 STATE 对照复用可运行的聚合终点 forward，采用同样输入和监督损失。它是任务适配，未使用预训练 SE 或原论文单细胞分布损失；CC-BY-NC-SA-4.0 来源许可证保留。其分数不能称为完整 STATE 论文重现。')
page();title('4 基因级缺失监督：损失与实测评估')
p('对每个训练条目定义 M，只在已观测条目计算 SmoothL1：先用 mask 安全选择真值和预测，再求损失并除以观测数。隐藏 NaN 不参与算术，隐藏条目的梯度精确为零。响应均值/方差仅按训练中已观测条目计算；输入和参数归一化不使用被隐藏的值。')
p('P1 不输入同靶点数值效应，避免把缺失标签通过 P2 投影带回。掩码由观测 ID 的固定哈希生成，与响应大小、显著性无关；30% 是 70% 已观测集的子集。相同掩码用于所有模型和三个随机种子。完全没有训练观测的输出基因会明确报错，不能声称能学习未测输出。')
data=[]
for fam in ('concat','fv','state_official','hierarchical_pooled','fv_continued'):
    for rate in (1.,.7,.3):
        r=M[M.family.eq(fam)&M.observed_fraction.eq(rate)].iloc[0];data.append([N[fam],str(int(rate*100))+'%',fmt(r.mae),fmt(r.seed_sd)])
table(['模型','训练已观测标签','测试 MAE','种子 SD'],data,[173,105,104,103])
p('100% 是前轮完全相同全局靶点划分的完整标签对照；70%/30% 是受控人工删除训练测量。测试真值始终是完整实测 RNA 响应，未合成生物响应。这可检验缺失监督稳健性，但不代表真实非随机缺失、多平台差异或真实器官行为已经验证。')
p('少退化也可能来自输出接近全局均值，而非学到充分扰动特异性。须与前轮零响应/训练均值、方向和 DEG 排序指标结合解读，不能仅凭掩码实验宣称生物预测能力强。')
page();title('5 完整分层模型：主任务与功能联合检查')
p('分层模块还在 pooled DESeq2 主任务、全局新靶点、严格九功能靶点留出以及两档缺失监督中实际训练。pooled 数据将多个供者合成一个逻辑组，其供者随机效应恒为零，不能从中推断独立供者方差；背景项仍部分池化。额外先验系数预先固定为 0.001，仅选择两种学习率。')
old=pd.read_csv(BASE/'full_experiments/analysis/all_model_summary.csv');data=[]
for name,label,previous in [('hierarchical_combo_P2','组合 P2','E1_P2_combo'),('hierarchical_global_P1','全局新靶点 P1','E1_global_target_P1'),('hierarchical_functional_P1','严格功能靶点 P1','E5_functional_targets_strict_P1')]:
    a=R[R.task.eq(name)&R.family.eq('hierarchical_pooled')].iloc[0];b=R[R.task.eq(name)&R.family.eq('fv_continued')].iloc[0];c=old[old.task.eq(previous)&old.family.eq('fv')].iloc[0];data.append([label,fmt(a.mae),fmt(b.mae),fmt(c.mae)])
table(['任务','联合分层 FV','同预算续训 FV','前轮原 FV'],data,[176,103,103,103])
p('组合任务还需对照前轮简单加性 MAE=0.172030；不能因神经网络相对另一神经网络略好而忽略更强简单基线。续训结果与仅第一阶段 3000 步模型的训练预算不同，已分别标注；验证、校准和测试来源不变。')
J=pd.read_csv(AN/'joint_functional_results.csv');data=[]
for r in J.itertuples():data.append([N[r.family],r.RNA_context,r.endpoint,fmt(r.spearman,3),'['+fmt(r.CI_low,3)+', '+fmt(r.CI_high,3)+']',fmt(r.direction,3)])
table(['模型','RNA 背景','终点','Spearman','双向重采样95%区间','方向'],data,[100,65,40,65,145,70])
p('九个预选靶点、13 个额外功能供者，RNA 终点为 log2FC、蛋白读出为阳性比例百分点变化。两个终点与三个 RNA 背景全部报告，未挑选最佳背景。此为同研究额外测量，时间未严格配对；宽区间和方向失败须保留，不称独立研究或已校准蛋白量预测。')
page();title('6 新分层模型的供者外推区间')
p('对新增供者模型取三种子集成预测，沿用前轮固定的区间规则：误差除以训练基因标准差（下限 0.05），在专门保留的验证校准靶点上取全响应向量最大误差的 90% 次序统计量，再评价未见供者。校准半集没有参与梯度、早停或学习率/收缩系数选择。')
I=pd.read_csv(AN/'donor_interval_results.csv');data=[]
for (fold,fam),g in I.groupby(['fold','family']):
    extra=json.loads((ROOT/'calibration'/(str(fold)+'_'+fam+'.json')).read_text());data.append([str(fold),N[fam],fmt(g.simultaneous_response_vector_coverage.mean(),3),fmt(g.coordinate_coverage.mean(),6),fmt(g.mean_interval_width.mean(),3),fmt(extra['test_mae_of_ensemble'])])
table(['供者折','方法','整向量覆盖','坐标覆盖','平均宽度','集成 MAE'],data,[48,145,75,75,70,72])
p('表中按三个刺激背景等权汇总；逐背景数据保留在 donor_interval_results.csv。宽区间可机械地提高覆盖，不能只以接近 90% 为成功。集成 MAE 与前页三种子逐损失的均值不是同一统计量。')
p('每折校准与测试来自不同供者，且仅一个校准供者；靶点间也有依赖。故这里只报告经验覆盖，不声称严格交换性或人群覆盖保证。这里的区间是保留数据校准的预测区间，不是高斯 MAP 的贝叶斯可信区间；潜变量尺度也不直接等同于生物机制强度。')
page();title('7 最终检查点的辅助指标')
A=pd.read_csv(AN/'auxiliary_metrics.csv');data=[]
for (task,fam),g in A.groupby(['task','family']):data.append(['组合 P2' if task=='hierarchical_combo_P2' else '全局新靶点 P1',N[fam],fmt(g.signed_Pearson.mean(),4),fmt(g.direction_accuracy.mean(),4),fmt(g.macro_context_DEG_AP.mean(),5),fmt(g.DEG_positive_prevalence.mean(),5)])
table(['任务','模型','Pearson','方向','DEG AP','正例率'],data,[100,120,65,65,70,65])
p('Pearson 是逐条件有符号响应相关的平均；常数向量相关记未定义。方向只计算真值绝对效应>0.1 的基因，无可判定基因的行不计。AP 用全部有限调整 p 值、FDR<=0.1 定义正例，预测效应绝对值排序；无正例行记未定义而非零。这是 AP，不是梯形 PR-AUC。各种子、未定义行数及背景内扰动检索记录保存在 auxiliary 中。')
p('这些指标来自最终修正检查点的独立推理，没有再更新参数或选择超参数。不能只凭全基因 MAE 接近均值基线就认定恢复了扰动特异性，尤其需要同时查看方向、稀有 DEG 排序和独立读出。')
image('missingness.png',180)
p('上图对应前述嵌套缺失训练表，误差条仅是三个随机种子的变异，不是独立供者置信区间。100%/70%/30% 是训练标签观测比例，测试参照不变；少退化不自动等于生物学泛化强。')
page();title('8 大纲剩余事项与稿件可写范围')
table(['事项','核验结论'],[['分层收缩与缺失损失','实现、重新训练和对照齐全。分层为高斯 MAP，而非完整后验采样；缺失评测限定 P1。'],['P0 真正未见背景','模型训练和显式输入层面的三背景联合新靶点评测已补齐，上游来源选择限制保留。'],['预训练状态编码器','大纲允许小型骨干起步，非必选 foundation model；当前学习状态层与固定状态消融已有结果，不能声称完成预训练 SE。'],['标准误加权','本稿不使用 SE 权重，因此不需要把未实施的加权/不加权对照写入主方法。'],['E3 完全匹配广域预训练','现有数据缺少同时匹配细胞数、独立供者与平台的设计；已有来源对比保留混杂，不以额外跑训练冒充补齐。'],['独立原代 T 研究','现有跨研究测试为非原代 T OOD；GSE278572 有原始计数但 KO/CRISPRi 与背景不匹配，不把其改标签当相同任务。'],['E5 组织/功能','九靶点独立蛋白读出已完成且区间宽、验证不稳健。真实器官组成/干预与动态仍需相应数据；合成恒等式不替代。']], [146,339])
p('从稿件角度，已不再把上述三个实现缺口算成“数据不足”。剩余阻碍主要是数据可比性、独立验证强度与方法增益，而不是缺一个训练入口。现有实验即使为阴性，也应按结果写作；不能保证通过增加模型数量达到顶会要求。')
p(f"审计通过：{AUD['new_neural_checkpoints']} 个新增模型均有 SHA256 和逐条件测试记录；原 {AUD['protected_original_files_unchanged']} 个保护文件及原始冻结划分哈希不变。Digital_Organ_Twin、旧投稿基线、原大纲与前轮报告保留。新增均在 day25_26；补充结果在 training_completion，代码在 scripts，官方基线另在 full_experiments/state_runtime 生成进程隔离的运行副本。")
p('复核入口：analysis/all_results.csv、analysis/model_manifest.csv、analysis/donor_results.csv、analysis/hierarchical_donor_contrasts.json、analysis/missingness_results.csv；protocol.json、task_registry.json、audit.json 保留计划与审计。所有模型在 runs/<task>/<family>/<seed>/model.pt，可按选定训练参数复现。')
PDF=DEST/'FVcell_大纲训练补完核验_20261026.pdf';MD=ROOT/'FVcell_大纲训练补完核验_20261026.md'
def footer(c,d):
    c.setStrokeColor(colors.HexColor('#d1dde2'));c.line(42,38,553,38);c.setFont('YH',7);c.setFillColor(colors.HexColor('#637b87'));c.drawString(42,25,'FVcell · 训练补完 · 2026-10-26');c.drawRightString(553,25,str(d.page))
doc=SimpleDocTemplate(str(PDF),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=52,title='FVcell 大纲训练补完核验',author='FVcell project');doc.build(story,onFirstPage=footer,onLaterPages=footer);MD.write_text('\n'.join(md),encoding='utf8')
reader=PdfReader(str(PDF));assert len(reader.pages)>=6
print(json.dumps({'pdf':str(PDF),'pages':len(reader.pages),'markdown':str(MD)},ensure_ascii=False))
