# FVcell v2 扩展实验复现说明

日期：2026-10-26。仅限科研评测；没有对外发布、上传或提交模型。

## 结果入口

- `completion_audit.json`：登记任务与实际检查点是否齐全，原文件 SHA256 是否保持。
- `analysis/all_model_summary.csv`：所有方法×任务的真实测试 MAE、种子变异、参数量和选中拟合时间。
- `analysis/model_checkpoint_manifest.csv`：三随机种子检查点及 SHA256。
- `analysis/paired_model_comparisons.json`：固定队列条件下的靶点簇配对区间。
- `analysis/donor_crossvalidation.csv`、`donor_level_comparisons.json`：四供者轮流留出与供者层统计。
- `analysis/functional_concordance.csv`：两个预设蛋白终点、三个 RNA 背景全部结果。
- `analysis/ensemble_calibration.json`、`selective_prediction.csv`：校准及保留率—风险。
- `output/pdf/FVcell_大纲v2模型实验报告_20261026.pdf`：有局限说明的完整阅读版。

本目录的数据与统计不能把不同单位合成单一排行榜。DESeq2 log2FC 与独立计数重建的 delta log2(CPM+1) 分开比较。随机种子不是独立供者。

## 环境与数据

环境精确版本、GPU/CUDA、代码 SHA256 见 `software_and_code_manifest.json`。运行时 Python 位于用户本机 Codex runtime，脚本通过相邻 `vendor`、`vendor_ml`、`vendor_state` 使用本次已安装依赖。无需联网下载依赖即可在当前机器运行。

两份主数据保存在上级 `large_data`，都有完整 SHA256 收据。外部 scPerturb 文件保存在 `external_data`，已与 Zenodo 公布的 MD5 核对；Reactome 原始快照和人类通路构建证明保留。不要用后来变化的 `current` 下载文件覆盖本次快照。不要把预分配文件长度当作验证字节数。

原主任务划分 `../prepared/FREEZE_LOCK.json` 及所有补充协议是复现约束。原研究上游靶点/特征整理的选择偏差无法由当前划分完全消除。没有读取旧 NPZ、显著性筛选表、旧投稿模型作为教师或响应真值。

## 执行顺序

以下均为上级 `scripts` 内的现有入口。需使用本机记录的 Python，设置 `PYTHONUTF8=1`，工作目录为 `day25_26`。已完成结果会跳过；不要并发启动同一任务或手工移除仍有活动进程的锁文件。

1. `full_suite_run.py`：组合/全局靶点/最小供者划分、输入消融、规模曲线、基础消融和原功能验证划分。
2. `full_suite_strict_permissions_run.py`：共同监督目标下严格去除同靶点效果的 P0/P1/P2。这里 P0 仍只是没有目标未扰动表达输入，并非整个背景从监督数据中完全消失。
3. `full_suite_external_run.py`：相同 519 靶点/6,795 基因的辅助来源对比，以及整外部研究留出。
4. `full_suite_donor_cv_run.py --start-fold 0 --end-fold 4`：四供者完整轮流留出。现有产物齐全时会跳过各家族检查点；并发锁防止重复训练。
5. `full_suite_functional_strict_run.py`：九功能靶点全背景剔除，使用全局未见靶点验证选择规则。
6. `full_suite_additive_baseline.py`：已有收缩均值/背景加性模型，按本轮选择/校准两半重新拟合。沿用此前 4×4 权重网格；该家族在早期已查看，因此是探索性对照。
7. `full_suite_analyse.py`：汇总全部结果、靶点簇配对、三种子集成校准、独立流式读出和四供者统计。
8. `full_suite_final_audit.py`：重新核对测试清单、结果聚合、按 1e-6 最小改进规则回放早停、模型 SHA256、来源真实效应的随机行、原始保护文件与冻结划分。
9. `full_suite_report.py`：只有计算登记表全部通过审计才生成 MD/PDF。PDF 的布局和科学图须重新渲染后检查。

本轮统一管线测试命令为 `python -B -m unittest test_split_rules test_first_training test_atomic_json test_full_suite`，工作目录 `scripts`。`final_pipeline_tests.log` 保存 20 项通过的记录。测试中的合成数据只检查代码，不进入生物学报告。

## 实现边界

STATE 为官方聚合终点 forward 的任务适配，并非其预训练、单细胞 OT 或完整论文重现；许可证 CC-BY-NC-SA-4.0 与来源文件保留。固定状态映射是随机线性映射，并非预训练 SE。当前残差收缩是 L2 潜变量惩罚，并非完整层次贝叶斯模型。

扩展组合测试在首轮最小模型中已被查看，不称为确认性测试。E3 没有完成细胞数/独立供者/平台的全匹配。E5 流式为同研究的独立额外供者与测量，只有九个预选靶点，不能替代独立研究的功能预测验证。没有真实器官组成干预、运输或纵向时间序列；合成恒等式检查不是组织生物学结果。

所有新增文件均在 `day25_26`。用户原文件、Digital_Organ_Twin、旧投稿基线和保留文档不移动、不改写。
