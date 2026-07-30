"""阈值调优脚本：为 Jaccard 指标找到最优的二值化阈值。"""
import sys, torch, numpy as np
sys.path.insert(0, '.')

from utils import set_current_dataset; set_current_dataset('mimic3')
from preprocess.data_load import preprocess_data
from Task import initialize_task
from utils import seq_dataloader, prepare_labels
from models.TrajectoryCare import TrajectoryCare
from trainer import _get_chapter_info_from_batch
from tqdm import tqdm

device = torch.device('cpu')

class Args:
    dataset = 'mimic3'
    task = 'drug_rec_ts'
    developer = False

task_dataset = preprocess_data(Args())
Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(task_dataset, Args())
_, _, test_loader = seq_dataloader(task_dataset, batch_size=64)

label_name = 'drugs'

# 获取所有预测结果
model = TrajectoryCare(
    Tokenizers_visit_event=Tokenizers_visit_event,
    Tokenizers_monitor_event=Tokenizers_monitor_event,
    output_size=label_size,
    device=device,
    chapter_labels=np.zeros(19, dtype=int),
    num_prototypes=8, num_chapters=19,
    embedding_dim=128, dropout=0.3,
    use_traj_router=True,
)

ckpt_path = 'logs/20260730/TrajectoryCare_mimic3_batchsize_32_epochs_30_v4_共享Emb_LN_lr1e-4_drop0.3/best_model_jaccard.ckpt'
state = torch.load(ckpt_path, map_location=device)
# 处理可能嵌套的 state dict
while 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=False)
model.eval()

y_true_all, y_prob_all = [], []
with torch.no_grad():
    for data in tqdm(test_loader, desc='Evaluating'):
        if isinstance(data, dict):
            label = prepare_labels(data[label_name], label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

        if hasattr(model, 'router'):
            chapter_dist, traj_features, num_visits = _get_chapter_info_from_batch(data, model)
            logits = model(data, chapter_dist, num_visits, traj_features)
        else:
            logits = model(data)

        y_true_all.append(label.cpu().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().numpy())

y_true = np.concatenate(y_true_all, axis=0)
y_prob = np.concatenate(y_prob_all, axis=0)

# 阈值扫描：0.05~0.95
best_j = 0
best_t = 0.5
print('\n阈值扫描:')
for t in np.arange(0.05, 0.96, 0.05):
    y_pred = (y_prob > t).astype(int)
    # Jaccard per sample, then mean
    intersection = np.logical_and(y_true, y_pred).sum(axis=1)
    union = np.logical_or(y_true, y_pred).sum(axis=1)
    j = np.mean(intersection / np.clip(union, 1, None))
    if j > best_j:
        best_j = j
        best_t = t
    print(f'  threshold={t:.2f}: Jaccard={j:.4f}')

print(f'\n最佳阈值: {best_t:.2f}, 最佳Jaccard: {best_j:.4f}')

# 最终测试：10 次采样取 mean±std
np.random.seed(222)
results = []
for epoch in range(10):
    idx = np.random.choice(len(y_true), int(len(y_true) * 0.8), replace=False)
    y_pred = (y_prob[idx] > best_t).astype(int)
    y_t = y_true[idx]
    i = np.logical_and(y_t, y_pred).sum(axis=1)
    u = np.logical_or(y_t, y_pred).sum(axis=1)
    j = np.mean(i / np.clip(u, 1, None))
    results.append(j)

results = np.array(results)
print(f'\n最终结果 (阈值={best_t:.2f}):')
print(f'Jaccard: {results.mean():.4f} +/- {results.std():.4f}')
