"""疾病原型验证脚本：分析路线C治疗演化社区原型的合理性与可解释性。

验证维度：
1. 原型数量 (K) 及选择依据
2. 聚类质量 (轮廓系数)
3. 医学同质性: 每个原型的诊断集中在哪些 ICD 章节/系统
4. 患者覆盖: 每个原型的患者分布（是否空/极小）
5. 代表性诊断
"""
import sys, os
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_current_dataset
set_current_dataset('mimic3')


class Args:
    dataset = 'mimic3'
    task = 'drug_rec_ts'
    developer = False


def main():
    from preprocess.data_load import preprocess_data
    from models.trajectory_mining.hetero_kg import (
        sample_to_visits, build_hetero_edges,
        discover_prototypes_metapath2vec,
    )
    from models.expert_selectv2 import map_ccs_to_expert
    from preprocess.icd_chapter_mapping import CHAPTER_NAMES_ICD9

    print("=" * 60)
    print("疾病原型验证分析")
    print("=" * 60)

    # 1. 加载数据 + 构建异构图
    ds = preprocess_data(Args())
    all_visits = []
    disease_nodes = set()
    patient_diseases = []  # 每个多就诊患者的历史诊断（用于患者分布）
    for s in ds.samples:
        visits = sample_to_visits(s)
        if len(visits) >= 2:
            all_visits.append(visits)
            # 患者全部历史诊断
            pd = []
            for v in visits:
                pd.extend(v['D'])
            patient_diseases.append(set(pd))
            for v in visits:
                for d in v['D']:
                    disease_nodes.add('D_' + d)

    disease_list = sorted(disease_nodes)
    print(f"\n1. 数据规模")
    print(f"   多就诊患者: {len(all_visits)}")
    print(f"   CCS诊断节点: {len(disease_list)}")

    edges = build_hetero_edges(all_visits)

    # 2. 原型发现（自动选 K）
    print(f"\n2. 原型发现 (Metapath2Vec, 自动选K)")
    diag_labels, info = discover_prototypes_metapath2vec(
        edges, disease_list, num_prototypes=None,  # 自动选K
        embed_dim=64, num_walks=10, walk_length=8, epochs=3,
    )
    K = info['num_prototypes']
    k_sel = info['k_selection']
    print(f"   自动选K = {K}")
    print(f"   轮廓系数: {k_sel.get('best_silhouette', 'N/A'):.4f}")
    print(f"   K候选分数: {k_sel.get('k_scores', {})}")

    # 3. 原型构成分析
    print(f"\n3. 原型构成（医学同质性）")
    node_label = {n: l for n, l in zip(disease_list, diag_labels)}

    # 原型→患者计数
    proto_patients = Counter()
    for pd in patient_diseases:
        protos = set()
        for d in pd:
            l = node_label.get('D_' + d)
            if l is not None:
                protos.add(l)
        for l in protos:
            proto_patients[l] += 1

    for k in range(K):
        mask = diag_labels == k
        nodes = [disease_list[i] for i in range(len(disease_list)) if mask[i]]
        n_diag = len(nodes)

        # 诊断的 ICD 章节分布
        chapter_counts = Counter()
        for n in nodes:
            ccs = n[2:]
            eid = map_ccs_to_expert(ccs)
            chapter_counts[eid] += 1

        # 患者数
        n_pat = proto_patients.get(k, 0)

        print(f"\n   [原型 {k}] {n_diag} 个诊断, 覆盖 {n_pat} 患者")
        # 主要章节
        total = max(sum(chapter_counts.values()), 1)
        top_ch = chapter_counts.most_common(3)
        ch_str = ', '.join(
            [f"{CHAPTER_NAMES_ICD9.get(eid, f'E{eid}')}:{cnt}({100*cnt/total:.0f}%)"
             for eid, cnt in top_ch])
        print(f"     主要章节: {ch_str}")
        # 代表性诊断（前10个编码）
        print(f"     代表诊断: {nodes[:10]}")

    # 4. 合理性评估
    print(f"\n4. 合理性评估")
    sizes = [int((diag_labels == k).sum()) for k in range(K)]
    pat_counts = [proto_patients.get(k, 0) for k in range(K)]
    print(f"   原型大小: {sizes}")
    print(f"   患者覆盖: {pat_counts}")
    print(f"   大小变异系数(CV): {np.std(sizes)/max(np.mean(sizes),1):.2f}")
    empty_proto = sum(1 for n in pat_counts if n < 10)
    print(f"   覆盖<10患者的原型数: {empty_proto}/{K}")
    if empty_proto == 0:
        print("   => 每个原型都有足够患者支撑, 合理")
    else:
        print("   => 存在患者过少的原型, 需要关注")

    # 5. 保存原型信息
    save_path = 'logs/20260804/prototypes_validation.json'
    import json
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'K': K,
            'k_selection': {kk: (vv if not isinstance(vv, dict) else str(vv))
                            for kk, vv in k_sel.items()},
            'sizes': sizes,
            'patient_coverage': pat_counts,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n   详情已保存: {save_path}")


if __name__ == '__main__':
    main()
