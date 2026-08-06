"""治疗同质性验证：按"治疗模式"而非"ICD章节"评判原型合理性。

路线C的核心主张：原型按治疗演化路径划分，而非疾病章节。
因此验证应看：同一原型内的诊断，是否被相似地治疗（用药/手术）。

指标：
1. 每个原型的用药(ATC)分布 —— 是否形成清晰治疗主题
2. 每个原型的手术分布
3. 原型内 vs 原型间的元路径连接密度
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

    print("=" * 60)
    print("治疗同质性验证（而非章节同质性）")
    print("=" * 60)

    ds = preprocess_data(Args())
    all_visits = []
    disease_nodes = set()
    for s in ds.samples:
        visits = sample_to_visits(s)
        if len(visits) >= 2:
            all_visits.append(visits)
            for v in visits:
                for d in v['D']:
                    disease_nodes.add('D_' + d)
    disease_list = sorted(disease_nodes)
    edges = build_hetero_edges(all_visits)
    print(f"多就诊患者: {len(all_visits)}, 诊断: {len(disease_list)}")

    # 原型发现（自动选K）
    diag_labels, info = discover_prototypes_metapath2vec(
        edges, disease_list, num_prototypes=None,
        embed_dim=32, num_walks=5, walk_length=6, epochs=2)
    K = info['num_prototypes']
    print(f"K = {K}")

    # 从边构建 诊断→用药 映射
    diag_med = defaultdict(Counter)  # 诊断 -> {ATC: count}
    diag_surg = defaultdict(Counter)  # 诊断 -> {手术: count}
    for (f, t, typ), c in edges.items():
        if typ == 'co_dm' and f.startswith('D_'):
            diag_med[f][t] += c
        elif typ == 'co_dp' and f.startswith('D_'):
            diag_surg[f][t] += c

    print(f"\n每个原型的治疗主题分析:")
    for k in range(K):
        nodes = [disease_list[i] for i in range(len(disease_list)) if diag_labels[i] == k]
        # 聚合该原型所有诊断的用药
        med_counter = Counter()
        surg_counter = Counter()
        for n in nodes:
            med_counter.update(diag_med.get(n, {}))
            surg_counter.update(diag_surg.get(n, {}))
        total_med = sum(med_counter.values())
        total_surg = sum(surg_counter.values())

        # 用药集中度：top-1 药占比（衡量治疗主题是否清晰）
        top1_med = med_counter.most_common(1)[0] if med_counter else ('None', 0)
        top1_ratio = top1_med[1] / max(total_med, 1)
        top5_med = [m for m, _ in med_counter.most_common(5)]

        print(f"\n  [原型 {k}] {len(nodes)} 诊断")
        print(f"    用药总数: {total_med}, 唯一用药: {len(med_counter)}")
        print(f"    top-1用药 {top1_med[0]}: {100*top1_ratio:.0f}%")
        print(f"    top-5用药: {top5_med}")
        print(f"    top-3手术: {[s for s,_ in surg_counter.most_common(3)]}")

    # 同质性指标：每个原型用药分布的熵（越低越集中）
    print(f"\n用药集中度汇总:")
    for k in range(K):
        nodes = [disease_list[i] for i in range(len(disease_list)) if diag_labels[i] == k]
        med_counter = Counter()
        for n in nodes:
            med_counter.update(diag_med.get(n, {}))
        total = sum(med_counter.values())
        if total > 0:
            probs = np.array([c / total for c in med_counter.values()])
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            # 归一化熵 (0=完全集中, 1=完全分散)
            norm_entropy = entropy / np.log(len(med_counter) + 1e-9)
            top1 = med_counter.most_common(1)[0][1] / total
            print(f"  原型{k}: 归一化熵={norm_entropy:.2f}, top-1用药占比={top1:.2f}, 唯一用药={len(med_counter)}")


if __name__ == '__main__':
    main()
