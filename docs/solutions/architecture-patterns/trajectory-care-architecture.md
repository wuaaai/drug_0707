---
title: "TrajectoryCare 架构设计：轨迹驱动的混合专家预测框架"
problem_type: knowledge
category: architecture-patterns
module: models/TrajectoryCare
tags: [moe, trajectory, ehr, clinical-prediction, expert-routing, spectral-clustering]
date: 2026-07-08
---

## Context

ExpertCare 证明了 ICD 章节驱动的 MoE 优于数据驱动的 MoE。但它将同一 ICD 编码的所有患者交给同一个专家——忽视了"同病不同径"的临床现实（心衰可来自高血压代偿耗竭，也可来自心梗心肌坏死）。

TrajectoryCare 提出范式转变：按"病是怎么来的"组织专家，而非按"得了什么病"。

## Guidance

### 三步架构

```
ICD编码 → [空间降维] → 章节块序列 → [轨迹抽象] → K个轨迹原型 → [MoE] → 预测
```

**Step 1 — 空间降维：** 数百种 ICD 编码 → 19/23 个章节块。复用 `expert_selectv2.py` 的 CCS→章节映射。

**Step 2 — 轨迹抽象：** 构建非对称章节转移矩阵 A[i,j]=P(j|i)，在转移拓扑空间中聚类 → K 个轨迹原型。K 通过 4 指标投票（Eigengap/Silhouette/DB/Gap）+ 临床先验约束 (≥7 个已知共病轴) 确定。

**Step 3 — 轨迹驱动 MoE：** K 个原型 → K 个 GRU 专家（独立参数）+ 软路由器（章节分布 + 温度退火）。输出 = Σ softmax(匹配分数) × expert(hidden)。

### 关键设计决策

1. **软路由而非硬路由**：患者可同时走多条轨迹（慢性代谢病 + 急性感染），软路由允许加权混合。
2. **图拓扑聚类而非深度编码**：聚类目标是可解释性，图节点和边（"C4→C9→C14"）天然可读。
3. **GRU 专家而非 FFN**：不同轨迹的疾病进展节奏不同（慢性代偿 vs. 急性打击），独立 GRU 允许学习轨迹特定的时序动态。
4. **冷启动退火**：τ(n) = τ₀·exp(-α·max(0,n-1)) + τ_min，低就诊数患者路由接近均匀。

## Why This Matters

这是首次将疾病演化轨迹从"描述性知识"升级为"模型参数组织原则"。如果轨迹驱动专家划分优于 ICD 静态划分，意味着 MoE 架构的设计原则从静态本体论转变为动态过程论。

## When to Apply

- 纵向 EHR 预测任务（≥2 次就诊），特别是药物推荐和诊断预测
- 需要模型可解释性的临床场景（路由热图可展示患者的轨迹归属）
- MIMIC-III (19 章节) 或 MIMIC-IV (22 章节)

## Examples

```python
# 使用方式
python main.py --model TrajectoryCare --dataset mimic3 --task drug_rec_ts --epochs 200

# 核心模型结构
model = TrajectoryCare(
    Tokenizers_visit_event=...,
    output_size=label_size,         # 195 drugs
    chapter_labels=chapter_labels,  # (19,) 每个章节→原型
    num_prototypes=7,               # K=7
    num_chapters=19,
)
# forward: logits = Σ_k softmax(route_score_k) × expert_k(batch_data)
```

**文件结构：**
```
models/TrajectoryCare.py                           # 主模型
models/trajectory_mining/prototype_discovery.py     # 轨迹发现 + K选择
preprocess/icd_chapter_mapping.py                  # CCS→章节映射
preprocess/trajectory_data_builder.py               # 章节序列 + 转移矩阵
```
