"""
ICD/CCS 编码 → 章节块映射。

注意：此项目的诊断编码已经被预处理为 CCS/CCSCM 编码（不是原始 ICD-9/10 编码）。
因此直接复用 expert_selectv2.py 中已有的 CCS→章节→Expert ID 映射。
"""

import re
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

# 从 expert_selectv2 导入已有的映射函数
from models.expert_selectv2 import map_ccs_to_expert

# ICD-9 章节名称（与 expert_selectv2 的 CHAPTER_TO_EXPERT_ID_MAP_ICD9 对应）
CHAPTER_NAMES_ICD9 = {
    0: 'C1 传染病和寄生虫疾病',
    1: 'C2 赘生物',
    2: 'C3 内分泌/营养/代谢/免疫',
    3: 'C4 血液及造血器官疾病',
    4: 'C5 精神失常',
    5: 'C6 神经系统疾病',
    6: 'C7 感觉器官疾病',
    7: 'C8 循环系统疾病',
    8: 'C9 呼吸系统疾病',
    9: 'C10 消化系统疾病',
    10: 'C11 泌尿生殖系统疾病',
    11: 'C12 妊娠/分娩/产后合并症',
    12: 'C13 皮肤及皮下组织疾病',
    13: 'C14 肌肉骨骼系统/结缔组织疾病',
    14: 'C15 先天性异常',
    15: 'C16 围产期引起的某些情况',
    16: 'C17 症状/体征/诊断不明',
    17: 'C18 损伤和中毒',
    18: 'C19 外部原因/健康影响因素',
}

# ICD-10 章节名称（与 expert_selectv2 的 CHAPTER_TO_EXPERT_ID_MAP_ICD10 对应）
CHAPTER_NAMES_ICD10 = {
    0: 'C1 传染病和寄生虫病',
    1: 'C2 肿瘤',
    2: 'C3 血液和造血器官/免疫',
    3: 'C4 内分泌/营养/代谢',
    4: 'C5 精神和行为障碍',
    5: 'C6 神经系统疾病',
    6: 'C7 眼和附器疾病',
    7: 'C8 耳和乳突疾病',
    8: 'C9 循环系统疾病',
    9: 'C10 呼吸系统疾病',
    10: 'C11 消化系统疾病',
    11: 'C12 皮肤和皮下组织疾病',
    12: 'C13 肌肉骨骼/结缔组织疾病',
    13: 'C14 泌尿生殖系统疾病',
    14: 'C15 妊娠/分娩/产褥期',
    15: 'C16 围生期疾病',
    16: 'C17 先天性畸形/变形/染色体异常',
    17: 'C18 症状/体征/异常化验结果',
    18: 'C19 损伤/中毒/外因结果',
    19: 'C20 发病和死亡的外因',
    20: 'C21 影响健康状况/接触健康服务',
    21: 'C22 特殊用途编码',
}


def map_code_to_chapter(code: str, dataset: str = 'mimic3') -> Optional[str]:
    """将单个 CCS 编码映射到章节块名称。

    直接复用 expert_selectv2.map_ccs_to_expert，返回 Expert ID 对应的章节名称。
    """
    expert_id = map_ccs_to_expert(str(code).strip())
    chapter_names = CHAPTER_NAMES_ICD9 if dataset == 'mimic3' else CHAPTER_NAMES_ICD10
    return chapter_names.get(expert_id, None)


def build_chapter_sequence(
    condition_sequences: List[List[str]],
    dataset: str = 'mimic3'
) -> List[List[str]]:
    """将多位患者的诊断序列转换为章节块序列。

    Args:
        condition_sequences: 每位患者的 visit 级诊断列表
            [[visit1_conditions], [visit2_conditions], ...]
        dataset: 'mimic3' 或 'mimic4'

    Returns:
        章节块序列 [['C7', 'C4'], ['C7', 'C14'], ...]
    """
    chapter_seqs = []
    for visit_conditions in condition_sequences:
        chapters = set()
        for code in visit_conditions:
            ch = map_code_to_chapter(str(code), dataset)
            if ch:
                chapters.add(ch)
        chapter_seqs.append(sorted(list(chapters)))
    return chapter_seqs


def build_transition_matrix(
    all_patient_chapter_seqs: List[List[List[str]]],
    num_chapters: int = 19
) -> np.ndarray:
    """构建非对称章节转移矩阵。

    Args:
        all_patient_chapter_seqs: 所有患者的章节块序列
        num_chapters: 章节块数量 (MIMIC-III: 19, MIMIC-IV: 23)

    Returns:
        转移矩阵 A, shape (num_chapters, num_chapters)
        A[i, j] = P(章节_j | 章节_i)，非对称
    """
    # 转移计数矩阵
    trans_counts = np.zeros((num_chapters, num_chapters))
    # 每个章节出现的总次数（作为"从"节点）
    from_counts = np.zeros(num_chapters)

    def chapter_idx(ch: str) -> int:
        return int(ch[1:]) - 1

    for patient_seq in all_patient_chapter_seqs:
        for t in range(len(patient_seq) - 1):
            current_chapters = patient_seq[t]
            next_chapters = patient_seq[t + 1]
            for ch_from in current_chapters:
                from_idx = chapter_idx(ch_from)
                from_counts[from_idx] += 1
                for ch_to in next_chapters:
                    to_idx = chapter_idx(ch_to)
                    trans_counts[from_idx, to_idx] += 1

    # 归一化为转移概率
    trans_matrix = np.zeros_like(trans_counts, dtype=float)
    for i in range(num_chapters):
        if from_counts[i] > 0:
            trans_matrix[i] = trans_counts[i] / from_counts[i]

    return trans_matrix


# 章节块名称（用于可解释性输出）
CHAPTER_NAMES_ICD9 = {
    'C1': '传染病和寄生虫病',
    'C2': '肿瘤',
    'C3': '内分泌/营养/代谢',
    'C4': '血液及造血器官',
    'C5': '精神障碍',
    'C6': '神经系统和感觉器官',
    'C7': '循环系统',
    'C8': '呼吸系统',
    'C9': '消化系统',
    'C10': '泌尿生殖系统',
    'C11': '妊娠/分娩/产褥期',
    'C12': '皮肤和皮下组织',
    'C13': '肌肉骨骼/结缔组织',
    'C14': '先天异常',
    'C15': '围生期疾病',
    'C16': '症状/体征/实验室异常',
    'C17': '损伤和中毒',
    'C18': '外部原因损伤',
    'C19': '健康影响因素',
}

CHAPTER_NAMES_ICD10 = {
    **CHAPTER_NAMES_ICD9,
    'C20': '疾病和死亡外因',
    'C21': '健康影响因素',
    'C22': '特殊目的代码',
}
