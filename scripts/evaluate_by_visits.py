"""分层评估：轨迹原型路由在 单就诊 vs 多就诊 患者上的性能对比。

目的：证明"轨迹原型对多就诊患者更有效"（承认边界）。
用路线C完整实验的 checkpoint，在测试集按就诊数分层计算 Jaccard。
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_current_dataset, seq_dataloader, prepare_labels
set_current_dataset('mimic3')


class Args:
    dataset = 'mimic3'
    task = 'drug_rec_ts'
    developer = False
    dim = 128
    dropout = 0.3


def main():
    from preprocess.data_load import preprocess_data
    from Task import initialize_task
    from models.TrajectoryCare import TrajectoryCare, assign_expert_types
    from models.trajectory_mining.hetero_kg import (
        sample_to_visits, build_hetero_edges,
        discover_prototypes_metapath2vec,
        compute_patient_prototype_activation,
    )
    from trainer import _get_chapter_info_from_batch

    device = torch.device('cpu')

    # 1. 加载数据
    task_dataset = preprocess_data(Args())
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = \
        initialize_task(task_dataset, Args())
    _, _, test_loader = seq_dataloader(task_dataset, batch_size=64)
    label_name = 'drugs'

    # 2. 重新跑 metapath2vec 原型（固定 K=8，与训练一致）
    all_visits = []
    disease_nodes = set()
    for s in task_dataset.samples:
        visits = sample_to_visits(s)
        if len(visits) >= 2:
            all_visits.append(visits)
            for v in visits:
                for d in v['D']:
                    disease_nodes.add('D_' + d)
    edges = build_hetero_edges(all_visits)
    disease_list = sorted(disease_nodes)
    diag_labels, hkg_info = discover_prototypes_metapath2vec(
        edges, disease_list, num_prototypes=8,  # 与训练一致 K=8
        embed_dim=64, num_walks=10, walk_length=8, epochs=3, device=device)
    K = hkg_info['num_prototypes']
    diag_to_proto = {}
    for node, label in zip(disease_list, diag_labels):
        if node.startswith('D_') and label >= 0:
            diag_to_proto[node[2:]] = int(label)

    # 3. 构造模型（与训练一致: 路线C + 共现 + lab）
    model = TrajectoryCare(
        Tokenizers_visit_event=Tokenizers_visit_event,
        Tokenizers_monitor_event=Tokenizers_monitor_event,
        output_size=label_size, device=device,
        chapter_labels=np.zeros(K, dtype=int),
        num_prototypes=K, num_chapters=K,
        embedding_dim=Args.dim, dropout=Args.dropout,
        use_traj_router=False,
        expert_types=assign_expert_types(None, K),
        use_drug_cooccurrence=True,
        use_lab_encoder=True,
        lab_tokenizer=Tokenizers_monitor_event.get('lab_inj_merged_list'),
    )
    model.diag_to_proto = diag_to_proto

    # 4. 加载 checkpoint（路线C完整实验）
    ckpt_path = 'logs/20260804/TrajectoryCare_mimic3_batchsize_32_epochs_50_路线C_完整/best_model_jaccard.ckpt'
    if not os.path.exists(ckpt_path):
        print(f"未找到 checkpoint: {ckpt_path}")
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"模型加载成功, K={K}")

    # 5. 分层评估
    groups = {'single': [], 'multi': []}  # 每个组存 (y_true, y_prob)
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, dict):
                label = prepare_labels(data[label_name], label_tokenizer).to(device)
                conditions_list = data.get('conditions', [])
            else:
                label = prepare_labels(data[0][label_name], label_tokenizer).to(device)
                conditions_list = data[0].get('conditions', [])

            # 判断每个样本单/多就诊
            for b in range(len(conditions_list)):
                conds = conditions_list[b]
                n_visits = len(conds) if (isinstance(conds, list) and len(conds) > 0
                                          and isinstance(conds[0], list)) else 1
                grp = 'multi' if n_visits >= 2 else 'single'

            chapter_dist, traj_features, num_visits, visit_seqs = \
                _get_chapter_info_from_batch(data, model)
            logits, _, _ = model(data, chapter_dist, num_visits, traj_features,
                                 visit_chapter_seqs=visit_seqs)
            prob = torch.sigmoid(logits)

            yt = label.cpu().numpy()
            yp = prob.cpu().numpy()
            for b in range(len(conditions_list)):
                conds = conditions_list[b]
                n_visits = len(conds) if (isinstance(conds, list) and len(conds) > 0
                                          and isinstance(conds[0], list)) else 1
                grp = 'multi' if n_visits >= 2 else 'single'
                groups[grp].append((yt[b], yp[b]))

    # 6. 计算指标
    print("\n" + "=" * 55)
    print("分层评估: 轨迹原型路由性能 (测试集)")
    print("=" * 55)
    for grp in ['single', 'multi']:
        pairs = groups[grp]
        if not pairs:
            print(f"  {grp}: 无样本")
            continue
        yt = np.stack([p[0] for p in pairs])
        yp = np.stack([p[1] for p in pairs])
        ypred = (yp > 0.5).astype(int)
        inter = np.logical_and(yt, ypred).sum(axis=1)
        union = np.logical_or(yt, ypred).sum(axis=1)
        jac = inter / np.clip(union, 1, None)
        # F1
        tp = inter; fp = ypred.sum(1) - inter; fn = yt.sum(1) - inter
        prec = tp / np.clip(tp + fp, 1, None)
        rec = tp / np.clip(tp + fn, 1, None)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1, None)
        print(f"\n  [{grp}就诊] 样本数: {len(yt)}")
        print(f"    Jaccard: {jac.mean():.4f}")
        print(f"    F1:      {f1.mean():.4f}")
        print(f"    平均正标签数: {yt.sum(1).mean():.1f}")

    # 统计分布
    n_s = len(groups['single'])
    n_m = len(groups['multi'])
    print(f"\n  测试集分布: 单就诊 {n_s} ({100*n_s/(n_s+n_m):.0f}%), 多就诊 {n_m} ({100*n_m/(n_s+n_m):.0f}%)")


if __name__ == '__main__':
    main()
