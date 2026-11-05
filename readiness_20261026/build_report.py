from pathlib import Path
import hashlib,html,json,csv
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether
from complete_priorities import ROOT,BASE,MATRIX,CLAIMS,PAPERS

def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))

SECTIONS=[
 ('01  完成状态与当前判断',[
  '优先级一已完成：对照B03、B02与PIE的固定版本，比较研究对象、输出、输入证据、模型、数据划分和验证；生成逐项对照表及九条主张台账。结论是：现有组件和任务有明显先例，尚未建立足够独立的新方法贡献。',
  '优先级二已完成的部分：冻结补充评测规则，核验九份实际划分、三权限共同监督集、318个权重文件及98份受保护文件；复核48份已有供者级结果。尚未完成的部分：新的生物正对照实测、近期强近邻完整复现，以及独立原代T细胞研究确认。不能把协议完成写成独立验证完成。',
  '本轮没有新增训练；原有已注册训练已完成。新增分析是对已见测试输出的复核，不是一个全新的盲测。原始文档、旧投稿包、Digital_Organ_Twin、原权重与旧结果均保持原位。',
  '当前判断：可以开始写如实描述现有研究的稿件，但不应称为已具备充分证据的RECOMB方法论文。主要缺口是新颖性、公平比较下的可解释增量，以及与主张相匹配的独立证据。增加训练次数或更换标题不会自动补齐这些缺口。',
  '用户已确认尚未登记摘要。RECOMB 2027官方截止折算为北京时间：摘要10月29日19:59，全文11月5日19:59。本轮快照约剩70小时和238小时。摘要必须有实质内容且与全文基本一致，不能先空占名额再大改主题。'
 ]),
 ('02  对照如何限制创新主张',[
  '下面是相同问题维度的定向比较，不是按论文自报分数排榜。B03、B02、PIE均为本轮读到的预印本版本；本轮未重跑它们的全部代码。表中“未核实等价实验”表示证据不足，不能用作不存在先例的证明。',
  'B03最近邻主要是问题和数据：它已研究共享与背景特异响应，并使用相近原代T细胞系统。B02最近邻主要是迁移条件和公平设计。PIE最近邻主要是允许哪些训练证据以及如何建模群体效应。',
  '因此，不能把共享/残差公式、四供者测试、缺失不置零、收缩或增加证据通道单独列为首次贡献。剩余可争取的是一个有明确增量的计算/统计方案及其公平验证；当前尚未证实。'
 ]),
 ('03  固定后的信息权限与比较对象',[
  'P0：固定的靶点功能注释和可用背景描述，不读目标基线表达。严格版本还删除验证/测试靶点在所有背景中的监督响应。P1：在同一监督集上允许目标NTC基线；不直接输入被留出靶点的实测响应。P2：在共同监督集外，允许原训练池中同靶点的其他背景响应；目标条件结果始终封存。',
  '同背景的其他训练扰动可以作为已见背景中的学习数据；如果声称背景完全未见，则不能使用该背景的扰动响应。全局未见靶点与P2不兼容。外部DepMap/L1000等实测证据要单列，不能与纯功能注释混称同一先验。',
  '同信息权限主比较保留：严格P0/P1/P2各自的ridge、concat和FV，使用完全相同监督行、测试行和基因集。分层组件比较固定为hierarchical、fixed Gaussian、unshrunk offsets与matched continued FV，四个原冻结供者折全部报告。',
  '已存在的零响应、训练均值、允许时的同靶点均值、收缩加和与岭回归必须保留。若某方法需要的同靶点参考不存在，该结果记不适用；不能偷偷回退后仍宣称同靶点证据相同。',
  'State只报告官方伪bulk骨干终点适配器结果。PIE、B03等原始系统尚未在本机同一任务上完整比较，不拼接它们的论文分数。补跑时须匹配输入、标签、输出及预算；确实不适用的模型须说明原因。',
  '本次冻结的新超参数搜索预算为零：沿用原验证集选出的权重，不再拿已有测试挑学习率、残差维度或指标。早期3000步模型与追加1000步模型分表；模型重参数化是工程修正，不是全局可辨识性证明。'
 ]),
 ('04  固定后的指标与正负对照',[
  '原主终点不变：固定基因集上的有符号响应MAE；按靶点、培养背景、供者逐层等权汇总。合并DESeq2的log2FC与供者log2(1+CPM)差值分表，不能把不同单位合成一个分数。原FV对concat的P2比较及原表保留。',
  '固定辅助指标：有符号响应相关、预测沿真实响应方向的幅度、方向正确率、DE排序AP、扰动检索和背景残差误差。相关分母为零记NA并报告数量；方向只在|真实响应|>0.1的对应log2单位内计算并报告覆盖。不同模型必须使用相同可评估条件。',
  '源代码核查发现，已有DE AP定义为：所有有限的发表adj_p_value中，以<=0.1定义正例，用预测绝对log2FC排序，不附加效应大小阈值。已用公开修订单保留这一定义；最初拟议的0.05/0.5定义没有执行。原冻结文件保留，修订单另有哈希锁。',
  '扰动检索固定在同供者、同培养背景和固定测试面板内，使用响应向量余弦给真实靶点排名，平均处理并列；报告面板大小及零向量排除。背景残差仅在有合法训练参考的同靶点迁移中计算，不将普通残差误差冒称为重复噪声校正后的CGC恢复度。',
  '负对照包括零响应、训练均值，以及固定随机种子的20次靶点错配；条件满足时再做背景错配。错配只置换冻结预测，不重训或挑有利置换。负对照用于判断预测是否真的认得干预/背景，不能替代强预测基线。',
  '生物正对照规则已固定：在供者×培养×run内，以guide ID哈希拆分互不重叠的靶向guide和NTC，沿原计数质控和单位构建两份响应；有独立重复时优先保留重复分离。每组至少两个合格靶向guide及两个NTC，缺失支持的条件排除并记录。',
  '正对照只诊断指标能否识别真实信号，不输入训练或推理。若元数据不能支持独立拆分，就标记无法完成，不能用真值预测自己替代。该正对照尚未在真实数据执行；所有待执行指标也不能写成已有结果。'
 ]),
 ('05  统计规则与本轮实际核查',[
  '供者是供者泛化的生物学单位。先对同一条件的三个训练种子作汇总，再展示四个留出供者的成对差值。种子、靶点和基因数不能扩充为独立供者。四供者的重采样区间和16种符号置换仅作描述，不能因名义P值宣称充分人群证据。',
  '补充比较固定为：层级模型分别对continued FV、固定先验、无收缩模型，在MAE与响应相关上的六个探索性对比，统一Holm处理并保留原始值。不得从结果中再挑一个“主要比较”。这不覆盖此前全部探索，也不把旧检验变成新预注册检验。',
  '实核九份划分：四个供者折、三个严格权限任务、一个全局靶点任务、一个整研究任务均非空且行分离。严格P0/P1/P2共同监督20112行，另有8764行只作允许时的证据池；三者测试均2245行。',
  '168份原套件权重与150份补充权重逐文件SHA256一致。原五份划分锁及98份受保护原文件均未改变。六项权限/聚合软件测试通过；这些人工小测试仅验证软件，不构成生物结果。',
  '复核48份已有供者结果文件，按供者等权重新汇总MAE、相关与方向正确率。MAE与原结果表的最大差为1.11e-16，属于浮点舍入。本轮没有产生新权重或新盲测结论。',
  '已有主任务：分层模型MAE约0.173392，continued FV约0.173386，简单收缩加和约0.172030，越小越好。已有四供者：分层相对固定先验MAE改善约4.15e-7，描述性区间跨零。现有结果不足以声称分层设计有明确增量。',
  '已有外部癌细胞测试中FV的MAE约0.232618，零响应约0.227809；这是一项有价值的负面OOD结果，但不是独立原代T细胞验证。同研究9靶点、13个额外流式供者是额外读出；六个分层模型相关区间均跨零，不能称已证明功能预测。',
  '静态状态提示：prepared/training_preflight.json是下载尚未完成时的历史记录；本轮判断依据已完成训练审计及实际文件核验，未把旧文件的false误当当前仍未训练。历史记录不覆盖改写。'
 ]),
 ('06  用通俗语言看这个领域',[
  '把细胞想成一家工厂。基因像开关和零件清单，RNA像工厂当前发出的工作单，蛋白质更像真正干活的机器。我们想问：关掉某个开关，或加一种药，这家工厂接下来会发生什么？不同人的同类工厂，也可能有不同反应。',
  '第一步进展，是把工厂的“照片”收集得更多、更整齐。scBaseCount解决大量公开数据难找、难统一的问题。这能帮助AI认出细胞及其状态；但照片再多，也不自动告诉它关开关以后会怎样。[D01]',
  '第二步进展，是主动做更多“改变开关”的实验。Tahoe等记录药物处理后的细胞变化。关键不仅是拍了一亿个细胞，还要看测过多少种药、多少背景、多少独立重复。[D02]',
  '第三步进展，是让AI借鉴别处的反应。State学习细胞群如何响应；Stack把参考细胞当提示；PIE把已知生物知识与允许使用的实验记录放在一起。它们解决的是不同条件下的预测问题，并非都已能模拟任意细胞。[M01-M03]',
  '最难的一关是“认得你，不等于知道你会怎样反应”。AI可能很会区分张三和李四的细胞，却只会给两个人同一个平均药物反应。近期研究开始专门检查：模型是否真的学到了人与人、状态与状态之间的差别。[B02-B03]',
  '另一项进展是改进考试。若一万个基因里只有几个改变，始终猜“不变”也可能拿到很低的平均误差。反过来，一张粗糙的考卷也可能看不出模型在那几个关键基因上的进步。所以要同时看总体误差、关键变化和有无真实信号的对照。[B05-B06]',
  '值得关注的方向是补充真正影响反应的信息，例如配对染色质、蛋白质和时间分辨实验，并用独立功能实验检验。生成一个看起来合理的细胞，还不等于预测对一次真实干预；能够预测RNA，也不等于能够预测疗效。[M07-M09]',
  '我们目前相当于做了一台可重复考试的原代T细胞预测器，配了多种对照。它能运行，也有部分信号，但还没有证明新增复杂部件比简单方案更有用。这是当前准确的位置，不能把它讲成已经做出完整数字细胞。'
 ]),
 ('07  投稿还差什么：按必要性排序',[
  '第一，锁定一个站得住的新贡献。官方征稿强调实质计算或方法推进；仅把现成方法用于新数据、整合资源或搭管线不足以自动匹配会议。现稿最紧要的是明确“相对B03/B02/PIE，新增什么可检验的能力或统计方法”，并有证据支撑，而不是增加论文包装。',
  '第二，提供同信息、同划分、同预算下的关键比较。简单收缩加和必须保留；当前层级增量不足，不能只选比它弱的模型。近期强近邻可在任务匹配时做一个有界比较；完整系统重训成本过大则如实说明，不编造基线。',
  '第三，判断真值与独立验证是否够支撑主张。生物正对照、供者级汇总、独立研究和功能读出分开。独立原代T细胞CRISPRi研究尚未落实；现有癌细胞跨研究测试及同研究流式不能自动补齐它。它未必是所有方法稿的形式硬条件，却直接限制跨研究/功能主张。',
  '第四，将主文写完整并可复核。问题、输入权限、数学定义、算法/统计依据、真实主结果、失败案例、局限和成本要闭合；不能只留下大纲、实验目录或图。摘要必须围绕已完成的内容，不许预写SOTA或显著提升。',
  '第五，完成提交所需的可访问代码、说明文档及格式核验。官方要求新软件/分析方法有公开源码和足够README；是否已公开、哪些许可可再分发，本轮未核验到发布完成。准备可公开复现材料不等于已经公开上传。',
  '不建议本轮扩展：训练全新百亿基础模型、下载整个scBaseCount/Tahoe、补齐多器官动力学、临时启动尚无样本的平台实验。除非真实资源已经就绪，否则这些方向无法可靠补上当前最关键证据。',
  '无需把E1-E5所有设想都塞入主文。E3完全匹配的广域预训练比较暂未完成；没有对应因果问题就缩小该主张。器官、组成、时序内容没有匹配观测时从本轮主线移出，旧投稿包继续作为基线参考。'
 ]),
 ('08  有时间约束的执行与停止条件',[
  '当前建议：把10月27-28日用于一次有边界的贡献核验和必要评测，不开展漫无目的的模型搜索。下面是建议排程和内部截止，尚不是已发生的进度或自动执行承诺。',
  '10月27日：根据本轮台账固定一个候选主张；完成正对照元数据可行性检查、同权限基线与近期近邻的适配判断。新外部数据先查完整标签、配对对照和许可，不先大规模下载。预留当天晚间给结果与老师讨论。',
  '10月28日18:00前：做投稿方向去留判断。继续RECOMB需要一个清楚的计算/统计增量、可解释的验证，以及能写成实质摘要的真实结果；不要求只靠单一P值，但也不能只剩已有分解和流水线。若仍无独立贡献，建议本轮不硬投RECOMB方法论文，转为继续研究或选择更合适的稿件类型。',
  '10月29日12:00前：若决定继续，由作者团队核对真实题目、作者名单及实质摘要并完成首次登记，给19:59官方截止留缓冲。本文不假定已经登记。可以微调摘要，不能指望登记后改成完全不同主题。',
  '10月30-31日：仅完成事先固定、直接回答主张的缺口，例如同条件强近邻、正对照或已确认可用的独立数据。已看结果的补充分析标探索性。新增数据若不能匹配任务，不冒充确认集。',
  '11月1日18:00前：冻结主结果、方法和范围。若证据仍不支持新增方法，保留阴性结果并停止扩张；不以更换指标或去掉简单基线修饰结论。',
  '11月2-3日：完成自洽主文、图表、补充材料和最小复现说明，由老师重点审查新颖性与结论边界。11月4日18:00前完成内部全稿检查。11月5日12:00前完成最终上传核对，19:59是官方最后时限。',
  '以当前约十天全文窗口，计算本身未必最慢；真正的风险是贡献尚未证明、独立数据不匹配和写作时间被反复试模型耗尽。不能给出“再跑几天就能达到顶会水平”的承诺。'
 ]),
 ('09  可直接复核的文件与依据',[
  'evaluation_protocol_frozen.json与FREEZE_LOCK.json：本轮补充协议及原始输入锁。evaluation_protocol_amendment_01.json与AMENDMENT_LOCK.json：依据已有评测代码保留DE AP定义的显式修订。有效协议是二者合读。',
  'novelty_matrix.csv、claim_ledger.json、literature_sources.json：逐项比较、可保留/删除主张及论文版本。audit_status.json：实际划分、全部318权重、原文件及截止日期核验。metrics_recheck_status.json和rechecked_donor_summary.csv：已有供者结果复核。',
  '本轮保留的外部来源：B03、B02、PIE、指标校准论文，以及RECOMB官方征稿与日期页面。论文正文使用上轮已保存公开全文，文献站本轮部分请求受限，不据此猜测更新版本。',
  '范围说明：方法或评测方案的固定、新结果执行、独立生物确认、稿件完成和门户提交是五个不同状态。本轮完成前两项中的文件固定与既有结果复核，没有把后续状态改写为已完成。'
 ])
]

def main():
    audit=load('audit_status.json');metrics=load('metrics_recheck_status.json')
    md=['# FVcell：优先级一二落实与RECOMB投稿决策','', '核查日期：2026-10-26，北京时间。原大纲、原工作计划和旧投稿包保持原位。','']
    for h,ps in SECTIONS:
        md += ['## '+h,'']
        for p in ps:md += [p,'']
        if h.startswith('02'):
            md += ['| 维度 | FVcell当前 | B03 | B02 | PIE |','|---|---|---|---|---|']
            md += ['| '+' | '.join(r)+' |' for r in MATRIX];md+=['']
            for r in CLAIMS:md += ['- '+r['claim']+'：'+r['decision']+'。'+r['reason']]
            md+=['']
    sources=[('B03','https://doi.org/10.64898/2026.09.30.755756'),('B02','https://doi.org/10.64898/2026.05.08.723625'),('PIE','https://doi.org/10.64898/2026.10.02.756297'),('指标校准','https://www.nature.com/articles/s41587-026-03307-w'),('RECOMB官方征稿','https://recomb.org/recomb2027/call_for_papers'),('官方截止日期','https://recomb.org/recomb2027/keydates_and_deadlines.html')]
    md+=['## 原始来源','']+[f'- [{a}]({b})' for a,b in sources]
    mdpath=ROOT/'优先级一二落实与投稿决策_20261026.md';mdpath.write_text('\n'.join(md),encoding='utf-8')
    pdfmetrics.registerFont(TTFont('YH','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('YHB','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
    ink=colors.HexColor('#18344B');teal=colors.HexColor('#087F8C');grey=colors.HexColor('#EAF2F4')
    body=ParagraphStyle('b',fontName='YH',fontSize=9.4,leading=15.4,wordWrap='CJK',textColor=ink,spaceAfter=9)
    small=ParagraphStyle('s',parent=body,fontSize=8,leading=12.2,spaceAfter=5)
    heading=ParagraphStyle('h',parent=body,fontName='YHB',fontSize=16,leading=23,textColor=teal,spaceAfter=15)
    sub=ParagraphStyle('sub',parent=body,fontName='YHB',fontSize=10.5,leading=16,spaceAfter=8,keepWithNext=True)
    title=ParagraphStyle('t',parent=heading,fontSize=23,leading=32,spaceAfter=18)
    def p(t,sty=body):return Paragraph(html.escape(t),sty)
    def link(a,b):return Paragraph('<link href="'+html.escape(b,quote=True)+'" color="#087F8C">'+html.escape(a+'  '+b)+'</link>',small)
    def table(rows,widths,head=True):
        t=Table([[p(str(x),small) for x in r] for r in rows],colWidths=widths,repeatRows=1 if head else 0,hAlign='LEFT')
        style=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#C4D8DC'))]
        if head:style += [('BACKGROUND',(0,0),(-1,0),grey)]
        t.setStyle(TableStyle(style));return t
    story=[]
    for i,(h,ps) in enumerate(SECTIONS):
        if i:story.append(PageBreak())
        if i==0:story += [p('优先级一二落实\n与投稿决策'.replace('\n',' / '),title),p('FVcell | 2026年10月26日 | RECOMB 2027',sub)]
        else:story.append(p(h,heading))
        for t in ps:story.append(p(t))
        if h.startswith('02'):
            story.append(PageBreak());story.append(p('逐项比较：问题与信息',heading))
            story.append(table([['维度','FVcell当前','B03','B02','PIE']]+MATRIX[:5],[60,111,111,111,118]))
            story.append(Spacer(1,14));story.append(p('逐项比较：方法与证据',sub))
            story.append(table([['维度','FVcell当前','B03','B02','PIE']]+MATRIX[5:],[60,111,111,111,118]))
            story.append(PageBreak());story.append(p('创新主张处理台账',heading))
            for r in CLAIMS:story += [p(r['claim'],sub),p(r['decision']+'。'+r['reason'])]
        if h.startswith('05'):
            story.append(Spacer(1,8));story.append(p('已有四供者结果的等权复核',sub))
            rr=[['模型','MAE','响应相关','方向正确率']]
            for key in ['fv_continued','hierarchical','fixed_gaussian','unshrunk_offsets']:
                r=metrics['families_equal_donor_mean'][key];rr.append([key,f"{r['mae']:.6f}",f"{r['signed_Pearson']:.4f}",f"{r['direction_accuracy']:.2%}"])
            story.append(table(rr,[190,107,107,107]))
            story.append(p('不同单位任务不能横向比较；此表是既有预测的重新汇总，不是新增盲测。',small))
    story += [Spacer(1,15),p('原始来源',sub)]
    for a,b in sources:story.append(link(a,b))
    out=ROOT/'output'/'pdf';out.mkdir(parents=True,exist_ok=True)
    path=out/'优先级一二落实与投稿决策_20261026.pdf'
    W,H=A4
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#C4D8DC'));c.line(42,40,W-42,40)
        c.setFont('YH',7);c.setFillColor(ink);c.drawString(42,27,'FVcell | 实际证据与投稿决策 | 2026-10-26');c.drawRightString(W-42,27,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=53,title='优先级一二落实与投稿决策',author='FVcell project').build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    info={'report_pdf':str(path),'report_pages':len(PdfReader(path).pages),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_md':str(mdpath),'priority_1':'completed comparison and claim boundaries','priority_2':'specification, audit and existing-metric recheck completed; new positive-control execution and independent primary-T confirmation pending','tests_passed':6,'original_files_unchanged':98,'weights_sha256_verified':318,'visual_review':'pending'}
    (ROOT/'delivery_status.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
