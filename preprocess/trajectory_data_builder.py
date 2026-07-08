"""
轨迹数据构建器 —— 从预处理后的 task_dataset 中提取章节块序列并构建转移图。
与现有 data_load.py 的 preprocess_data() 集成。

注意：数据集中的 conditions 是 CCS/CCSCM 编码，通过 expert_selectv2.map_ccs_to_expert
映射到 Expert ID（1:1 对应 ICD 章节）。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from models.expert_selectv2 import map_ccs_to_expert
from preprocess.icd_chapter_mapping import CHAPTER_NAMES_ICD9, CHAPTER_NAMES_ICD10


def build_transition_matrix(
    all_seq_expert_ids: List[List[List[int]]],
    num_experts: int
) -> np.ndarray:
    """构建非对称章节转移矩阵。

    Args:
        all_seq_expert_ids: 所有患者的 Expert ID 序列
        num_experts: Expert 数量

    Returns:
        trans_matrix: (num_experts, num_experts)，trans_matrix[i, j] = P(j | i)
    """
    trans_counts = np.zeros((num_experts, num_experts))
    from_counts = np.zeros(num_experts)

    for patient_seq in all_seq_expert_ids:
        for t in range(len(patient_seq) - 1):
            current = patient_seq[t]
            next_v = patient_seq[t + 1]
            for eid_from in current:
                if 0 <= eid_from < num_experts:
                    from_counts[eid_from] += 1
                    for eid_to in next_v:
                        if 0 <= eid_to < num_experts:
                            trans_counts[eid_from, eid_to] += 1

    trans_matrix = np.zeros_like(trans_counts, dtype=float)
    for i in range(num_experts):
        if from_counts[i] > 0:
            trans_matrix[i] = trans_counts[i] / from_counts[i]

    return trans_matrix


def extract_chapter_sequences_from_dataset(task_dataset, args) -> Tuple[List[List[List[int]]], np.ndarray, int]:
    """从 task_dataset 中提取所有患者的章节块序列（Expert ID 序列）并构建转移矩阵。

    Args:
        task_dataset: pyhealth 的 SampleDataset（已 set_task）
        args: 命令行参数（包含 dataset 字段）

    Returns:
        all_seqs: 每位患者的 Expert ID 序列 [[[7], [7, 10], ...], ...]
        trans_matrix: 非对称章节转移矩阵
        num_experts: Expert 数量（= ICD 章节数）
    """
    num_experts = 19 if args.dataset == 'mimic3' else 22
    all_seqs = []

    for sample in task_dataset.samples:
        # task_dataset.samples 中每个 sample 是一个 dict，
        # 每个 key (conditions, procedures等) 的值是嵌套列表 [[v1], [v2], ...]
        # 我们需要从 conditions 提取就诊数
        conditions = sample.get('conditions', [])
        if not isinstance(conditions, list) or len(conditions) == 0:
            continue

        # conditions 可能是 [['code1', 'code2'], ['code3']]（多就诊）
        # 或者 ['code1', 'code2']（单就诊）
        if isinstance(conditions[0], list):
            visit_conds = conditions  # 已是嵌套列表
        else:
            visit_conds = [conditions]  # 单就诊包装

        visit_expert_list = []
        for visit_codes in visit_conds:
            if not visit_codes:
                continue
            expert_ids = set()
            for code in visit_codes:
                eid = map_ccs_to_expert(str(code))
                expert_ids.add(eid)
            visit_expert_list.append(sorted(list(expert_ids)))

        if len(visit_expert_list) >= 1:
            all_seqs.append(visit_expert_list)

    trans_matrix = build_transition_matrix(all_seqs, num_experts)
    return all_seqs, trans_matrix, num_experts


def compute_chapter_patient_matrix(all_seqs: List[List[List[int]]], num_experts: int) -> np.ndarray:
    """计算每个患者的 Expert 分布向量（用于冷启动路由）。"""
    patient_dist = np.zeros((len(all_seqs), num_experts))

    for i, patient_seq in enumerate(all_seqs):
        for visit_experts in patient_seq:
            for eid in visit_experts:
                if 0 <= eid < num_experts:
                    patient_dist[i, eid] += 1

        row_sum = patient_dist[i].sum()
        if row_sum > 0:
            patient_dist[i] /= row_sum

    return patient_dist


def get_trajectory_features(all_seqs: List[List[List[int]]], num_experts: int) -> np.ndarray:
    """为每个患者提取轨迹拓扑特征。"""
    features = np.zeros((len(all_seqs), num_experts * 3))

    for i, patient_seq in enumerate(all_seqs):
        freq = np.zeros(num_experts)
        in_edges = np.zeros(num_experts)
        out_edges = np.zeros(num_experts)

        for t, visit_experts in enumerate(patient_seq):
            for eid in visit_experts:
                if 0 <= eid < num_experts:
                    freq[eid] += 1

            if t > 0:
                prev_experts = patient_seq[t - 1]
                for eid_from in prev_experts:
                    if 0 <= eid_from < num_experts:
                        out_edges[eid_from] += 1
                        for eid_to in visit_experts:
                            if 0 <= eid_to < num_experts:
                                in_edges[eid_to] += 1

        in_sum = in_edges.sum()
        out_sum = out_edges.sum()
        freq_sum = freq.sum()
        if in_sum > 0:
            in_edges /= in_sum
        if out_sum > 0:
            out_edges /= out_sum
        if freq_sum > 0:
            freq /= freq_sum

        features[i, :num_experts] = in_edges
        features[i, num_experts:2 * num_experts] = out_edges
        features[i, 2 * num_experts:3 * num_experts] = freq

    return features


def print_chapter_statistics(all_seqs, trans_matrix, num_experts, dataset):
    """打印章节统计信息。"""
    chapter_names = CHAPTER_NAMES_ICD9 if dataset == 'mimic3' else CHAPTER_NAMES_ICD10
    total_visits = sum(len(seq) for seq in all_seqs)
    total_patients = len(all_seqs)

    output = []
    output.append(f"=== 章节块统计 ({dataset}) ===")
    output.append(f"患者数: {total_patients}")
    output.append(f"总就诊数: {total_visits}")
    output.append(f"平均就诊数/患者: {total_visits / max(total_patients, 1):.2f}")

    expert_freq = Counter()
    for seq in all_seqs:
        for visit_experts in seq:
            for eid in visit_experts:
                expert_freq[eid] += 1

    output.append("\nExpert 频率分布:")
    for eid in range(num_experts):
        name = chapter_names.get(eid, f'E{eid}')
        count = expert_freq.get(eid, 0)
        output.append(f"  E{eid} ({name}): {count} ({100 * count / max(total_visits, 1):.1f}%)")

    output.append("\nTop-10 转移边:")
    edges = []
    for i in range(num_experts):
        for j in range(num_experts):
            if trans_matrix[i, j] > 0.01:
                edges.append((i, j, trans_matrix[i, j]))
    edges.sort(key=lambda x: -x[2])
    for i, j, prob in edges[:10]:
        output.append(f"  E{i} → E{j}: {prob:.4f}")

    return '\n'.join(output)
