"""统一 sample-wise 指标评估：对 v13/v14/v20/路线C 四个版本重测。

指标：
- sample-wise Jaccard = mean(每个患者 TP/(TP+FP+FN))  [药物推荐主流]
- micro Jaccard       = sklearn jaccard_score(ravel)  [原main.py口径]
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import set_current_dataset, seq_dataloader, prepare_labels
set_current_dataset('mimic3')


class Args:
    dataset = 'mimic3'
    task = 'drug_rec_ts'
    developer = False


VERSIONS = {
    'v13': dict(ckpt='logs/20260731/TrajectoryCare_mimic3_batchsize_32_epochs_50_v13_lab+全创新/best_model_jaccard.ckpt',
                dim=256, use_traj_router=True, num_prototypes=8, num_chapters=19, hetero=False),
    'v14': dict(ckpt='logs/20260801/TrajectoryCare_mimic3_batchsize_32_epochs_50_v14_lab_dim384/best_model_jaccard.ckpt',
                dim=384, use_traj_router=True, num_prototypes=8, num_chapters=19, hetero=False),
    'v20': dict(ckpt='logs/20260804/TrajectoryCare_mimic3_batchsize_32_epochs_50_v20_dim384_full/best_model_jaccard.ckpt',
                dim=384, use_traj_router=True, num_prototypes=8, num_chapters=19, hetero=False),
    'routeC': dict(ckpt='logs/20260804/TrajectoryCare_mimic3_batchsize_32_epochs_50_路线C_完整/best_model_jaccard.ckpt',
                   dim=128, use_traj_router=False, num_prototypes=8, num_chapters=8, hetero=True),
}


def build_model(cfg, device):
    from models.TrajectoryCare import TrajectoryCare, assign_expert_types

    K = cfg['num_prototypes']
    model = TrajectoryCare(
        Tokenizers_visit_event=Tokenizers_visit_event,
        Tokenizers_monitor_event=Tokenizers_monitor_event,
        output_size=label_size, device=device,
        chapter_labels=np.zeros(cfg['num_chapters'], dtype=int),
        num_prototypes=K, num_chapters=cfg['num_chapters'],
        embedding_dim=cfg['dim'], dropout=0.3,
        use_traj_router=cfg['use_traj_router'],
        expert_types=assign_expert_types(None, K),
        use_drug_cooccurrence=True,
        use_lab_encoder=True,
        lab_tokenizer=Tokenizers_monitor_event.get('lab_inj_merged_list'),
    )

    # 路线C: 重跑 metapath2vec 原型，挂载 diag_to_proto
    if cfg['hetero']:
        from models.trajectory_mining.hetero_kg import (
            sample_to_visits, build_hetero_edges, discover_prototypes_metapath2vec)
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
        diag_labels, _ = discover_prototypes_metapath2vec(
            edges, disease_list, num_prototypes=K,
            embed_dim=64, num_walks=10, walk_length=8, epochs=3, device=device)
        model.diag_to_proto = {}
        for node, label in zip(disease_list, diag_labels):
            if node.startswith('D_') and label >= 0:
                model.diag_to_proto[node[2:]] = int(label)
    else:
        model.diag_to_proto = None

    return model


def evaluate(model, loader, device):
    from trainer import _get_chapter_info_from_batch
    from pyhealth.metrics import binary_metrics_fn

    y_t_all, y_p_all = [], []
    with torch.no_grad():
        for data in loader:
            if isinstance(data, dict):
                label = prepare_labels(data['drugs'], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['drugs'], label_tokenizer).to(device)
            cd, tf, nv, vs = _get_chapter_info_from_batch(data, model)
            logits, _, _ = model(data, cd, nv, tf, visit_chapter_seqs=vs)
            y_t_all.append(label.cpu().numpy())
            y_p_all.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(y_t_all)
    y_prob = np.concatenate(y_p_all)
    y_pred = (y_prob > 0.5).astype(int)

    # sample-wise
    inter = np.logical_and(y_true, y_pred).sum(1)
    union = np.logical_or(y_true, y_pred).sum(1)
    sw_jac = (inter / np.clip(union, 1, None)).mean()
    # F1 sample-wise
    tp = inter; fp = y_pred.sum(1) - inter; fn = y_true.sum(1) - inter
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    sw_f1 = (2 * prec * rec / np.clip(prec + rec, 1, None)).mean()
    # micro (原口径)
    micro = binary_metrics_fn(y_true.ravel(), y_prob.ravel(), metrics=['jaccard'])['jaccard']
    return sw_jac, sw_f1, micro, len(y_true)


def main():
    global Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size, task_dataset

    from preprocess.data_load import preprocess_data
    from Task import initialize_task

    device = torch.device('cpu')
    task_dataset = preprocess_data(Args())
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = \
        initialize_task(task_dataset, Args())
    _, _, test_loader = seq_dataloader(task_dataset, batch_size=64)

    print("=" * 60)
    print("统一 sample-wise 指标评估 (测试集全量)")
    print("=" * 60)
    print(f"{'版本':<8}{'sample-wise Jaccard':<20}{'sample-wise F1':<16}{'micro Jaccard':<14}{'样本数'}")
    print("-" * 60)

    results = {}
    for name, cfg in VERSIONS.items():
        try:
            model = build_model(cfg, device)
            model.load_state_dict(
                torch.load(cfg['ckpt'], map_location=device), strict=False)
            model.eval()
            sw_jac, sw_f1, micro, n = evaluate(model, test_loader, device)
            results[name] = (sw_jac, sw_f1, micro, n)
            print(f"{name:<10}{sw_jac:.4f}          {sw_f1:.4f}       {micro:.4f}       {n}")
        except Exception as e:
            print(f"{name:<10}失败: {e}")

    print("\n" + "=" * 60)
    print("sample-wise Jaccard 排名:")
    for name in sorted(results, key=lambda x: -results[x][0]):
        print(f"  {name}: {results[name][0]:.4f}")


if __name__ == '__main__':
    main()
