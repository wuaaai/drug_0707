import random
import numpy as np
import matplotlib
import argparse
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
import torch
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
from pyhealth.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F

# 此文件修改的地方为67行加入了LABEVENTS以及将cond_hist对应变为drugs_hist

def batch_to_multihot(label, num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def update_best(loss, epoch, state, best_losses, k=5):
    if len(best_losses) < k:
        heapq.heappush(best_losses, (-loss, -epoch, state))
    else:
        if (-loss, -epoch) > best_losses[0][:2]:
            heapq.heappop(best_losses)
            heapq.heappush(best_losses, (-loss, -epoch, state))

def log_params(dataset, args, log_path):
    argsDict = args.__dict__
    with open(log_path, 'w') as log_file:
        log_file.write(f'parameter of dataset:\n')
        log_file.write(f'{dataset} \n')
        log_file.write('--------------------------------------\n')
        log_file.write(f'args params:\n')
        for eachArg, value in argsDict.items():
            log_file.writelines(eachArg + ' : ' + str(value) + '\n')
        log_file.write('--------------------------------------\n')


def log_results(epoch, run_time, train_loss, val_loss, metrics, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch {epoch + 1}\n')
        log_file.write(f'Train Loss: {train_loss:.4f}\n')
        log_file.write(f'Validation Loss: {val_loss:.4f}\n')
        log_file.write(f'F1: {metrics["f1"]:.4f}, '
                       f'Jaccard: {metrics["jaccard"]:.4f}, '
                       f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                       f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                       f'Run_Time: {run_time:.3f}\n')
        if (epoch + 1) % 20 == 0:
            log_file.write(f'{epoch + 1} epoch Model saved !!!\n')
        log_file.write('--------------------------------------\n')
def log_results_multi_task(epoch, run_time, train_loss, val_loss_dict, metrics_dict, log_path):
    """
    多任务版本日志记录函数
    val_loss_dict: dict，每个任务对应一个验证loss，比如 {'drug_rec': 0.25, 'diag_pred': 0.13}
    metrics_dict: dict，每个任务对应一个指标字典，比如
        {
            'drug_rec': {'f1': ..., 'jaccard': ..., 'roc_auc': ..., 'pr_auc': ...},
            'diag_pred': {...}
        }
    """
    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch {epoch + 1}\n')
        log_file.write(f'Train Loss: {train_loss:.4f}\n')

        # 判断val_loss_dict是不是字典，否则直接打印float
        if isinstance(val_loss_dict, dict):
            val_loss_str = ', '.join([f'{task}: {loss:.4f}' for task, loss in val_loss_dict.items()])
        else:
            val_loss_str = f'{val_loss_dict:.4f}'

        log_file.write(f'Validation Loss: {val_loss_str}\n')

        # 循环打印每个任务的指标
        for task_name, metrics in metrics_dict.items():
            log_file.write(f'=== Task: {task_name} ===\n')
            log_file.write(f'F1: {metrics["f1"]:.4f}, '
                           f'Jaccard: {metrics["jaccard"]:.4f}, '
                           f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                           f'PR-AUC: {metrics["pr_auc"]:.4f}\n')

        log_file.write(f'Run_Time: {run_time:.3f}\n')

        if (epoch + 1) % 20 == 0:
            log_file.write(f'{epoch + 1} epoch Model saved !!!\n')
        log_file.write('--------------------------------------\n')





def log_outmemory(data, error_log, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'{error_log}\n')
        log_file.write('--------------------------------------\n')


def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels


# def get_init_tokenizers(task_dataset, keys=['drugs_hist', 'procedures', 'drugs']):
#     Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
#     return Tokenizers
def get_init_tokenizers(task_dataset, keys=None):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers


def get_parent_tokenizers(task_dataset, keys=['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist': InnerMap.load("ICD9CM"), 'procedures': InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers


def seq_dataloader(dataset, split_ratio=[0.75, 0.1, 0.15], batch_size=64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 这三句我都不知道干啥的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


# 纯gpt写的
def get_sample_loader(data_loader, sample_size):
    sample_size = round(len(data_loader.dataset) * sample_size)
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # 随机打乱索引
    sample_indices = indices[:sample_size]  # 取前 sample_size 个索引
    subset = Subset(dataset, sample_indices)
    sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True,
                               collate_fn=data_loader.collate_fn)
    return sample_loader


def plot_losses(epoch_list, train_losses, val_losses, png_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_losses, label='Train Loss')
    plt.plot(epoch_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # 使用 numpy 找到 val_losses 的最小值及其索引
    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_epoch = epoch_list[min_val_loss_idx]
    
    # 标注 val_losses 的最低点
    plt.scatter(min_epoch, min_val_loss, color='red', zorder=5)  # 使用红色点标注最低点
    plt.text(min_epoch, min_val_loss, f'Min Val Loss\n({min_epoch:.0f}, {min_val_loss:.4f})', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=9, color='red')

    # 设置纵坐标范围
    plt.autoscale(True)
    # 保存绘图
    plt.savefig(png_path)
    plt.close()


def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds


def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks


# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters(model):
    print(f"{'Module':<30} {'Parameters':<15}")
    print('-' * 45)
    total_params = 0
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_params
        print(f"{name:<30} {module_params:<15,}")

    # Print total parameters
    print('-' * 45)
    print(f"{'Total Parameters':<30} {total_params:<15,}")


def print_dataset_parameters(task_datasets, Tokenizers_visit_events, Tokenizers_monitor_events, label_sizes, args):
    """
    统一处理单任务和多任务数据集参数统计。
    task_datasets: 单任务是单个Dataset，多任务是长度为2的tuple或list，分别对应两个任务的数据集
    Tokenizers_visit_events: 单任务是单个tokenizer dict，多任务是长度为2的tuple/list
    Tokenizers_monitor_events: 同上
    label_sizes: 单任务是整数，多任务是长度为2的tuple/list
    args: 包含task字段，区分单任务/多任务
    """

    # 判断是否多任务
    multi_task = (args.task == "multi_task")

    if multi_task:
        # 多任务模式，分别统计两个任务
        outputs = []
        for i, task_name in enumerate(['drug_rec', 'diag_pred']):
            task_dataset = task_datasets[i]
            Tokenizers_visit_event = Tokenizers_visit_events[i]
            Tokenizers_monitor_event = Tokenizers_monitor_events[i]
            label_size = label_sizes[i]

            patient_num = len(task_dataset.patient_to_index)
            visit_num = len(task_dataset.visit_to_index)

            if task_name == 'drug_rec':
                cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
                drug_num = label_size
            elif task_name == 'diag_pred':
                cond_num = label_size
                drug_num = len(Tokenizers_visit_event['drugs'].vocabulary)
            else:
                cond_num = drug_num = 0

            proc_num = len(Tokenizers_visit_event['procedures'].vocabulary)
            lab_num = len(Tokenizers_monitor_event['lab_item'].vocabulary)
            inj_num = len(Tokenizers_monitor_event['inj_item'].vocabulary)

            cond = proc = drug = 0
            for visit in task_dataset.samples:
                if task_name == 'drug_rec':
                    cond += len(visit['conditions'][-1])
                    proc += len(visit['procedures'][-1])
                    drug += len(visit['drugs'])
                elif task_name == 'diag_pred':
                    cond += len(visit['conditions'])
                    proc += len(visit['procedures'][-1])
                    drug += len(visit['drugs'][-1])

            avg_visit = visit_num / patient_num if patient_num > 0 else 0
            avg_cond = cond / visit_num if visit_num > 0 else 0
            avg_proc = proc / visit_num if visit_num > 0 else 0
            avg_drug = drug / visit_num if visit_num > 0 else 0

            output = (
                f"--- Task {i} ({task_name}) ---\n"
                f"patient_num: {patient_num}\n"
                f"visit_num: {visit_num}\n"
                f"cond_num: {cond_num}\n"
                f"drug_num: {drug_num}\n"
                f"proc_num: {proc_num}\n"
                f"lab_num: {lab_num}\n"
                f"inj_num: {inj_num}\n"
                f"avg_visit: {avg_visit}\n"
                f"avg_cond: {avg_cond}\n"
                f"avg_proc: {avg_proc}\n"
                f"avg_drug: {avg_drug}"
            )
            outputs.append(output)
        return "\n\n".join(outputs)

    else:
        # 单任务处理
        task_dataset = task_datasets
        Tokenizers_visit_event = Tokenizers_visit_events
        Tokenizers_monitor_event = Tokenizers_monitor_events
        label_size = label_sizes

        patient_num = len(task_dataset.patient_to_index)
        visit_num = len(task_dataset.visit_to_index)

        if args.task == 'drug_rec':
            cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
            drug_num = label_size
        elif args.task == 'diag_pred' or args.task == 'diag_pred_1010':
            cond_num = label_size
            drug_num = len(Tokenizers_visit_event['drugs'].vocabulary)
        elif args.task == 'drug_rec_ts':
            cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
            drug_num = label_size
        else:
            cond_num = drug_num = 0

        proc_num = len(Tokenizers_visit_event['procedures'].vocabulary)
        lab_num = len(Tokenizers_monitor_event['lab_item'].vocabulary)
        inj_num = len(Tokenizers_monitor_event['inj_item'].vocabulary)

        cond = proc = drug = 0
        for visit in task_dataset.samples:
            if args.task == 'drug_rec':
                cond += len(visit['conditions'][-1])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'])
            elif args.task == 'diag_pred':
                cond += len(visit['conditions'])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'][-1])
            elif args.task == 'drug_rec_ts':
                cond += len(visit['conditions'][-1])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'])

        avg_visit = visit_num / patient_num if patient_num > 0 else 0
        avg_cond = cond / visit_num if visit_num > 0 else 0
        avg_proc = proc / visit_num if visit_num > 0 else 0
        avg_drug = drug / visit_num if visit_num > 0 else 0

        output = (
            f"patient_num: {patient_num}\n"
            f"visit_num: {visit_num}\n"
            f"cond_num: {cond_num}\n"
            f"drug_num: {drug_num}\n"
            f"proc_num: {proc_num}\n"
            f"lab_num: {lab_num}\n"
            f"inj_num: {inj_num}\n"
            f"avg_visit: {avg_visit}\n"
            f"avg_cond: {avg_cond}\n"
            f"avg_proc: {avg_proc}\n"
            f"avg_drug: {avg_drug}"
        )
        return output

# 计算敏感性
def calculate_sensitivity(label, predict):
    if predict.shape != label.shape:
        raise ValueError("predict 和 label 的形状必须一致")

    # 计算真阳性和真实阳性
    true_positives = np.sum((predict == 1) & (label == 1), axis=1)
    total_positives = np.sum(label == 1, axis=1)

    # 计算敏感性，避免除零错误
    sensitivity = np.divide(true_positives, total_positives, out=np.zeros_like(true_positives, dtype=float),
                            where=total_positives != 0)

    return sensitivity

# 计算特异性
def calculate_specificity(label, predict):
    if predict.shape != label.shape:
        raise ValueError("predict 和 label 的形状必须一致")

    # 计算真阴性和真实阴性
    true_negatives = np.sum((predict == 0) & (label == 0), axis=1)
    total_negatives = np.sum(label == 0, axis=1)

    # 计算特异性，避免除零错误
    specificity = np.divide(true_negatives, total_negatives, out=np.zeros_like(true_negatives, dtype=float),
                            where=total_negatives != 0)

    return specificity
#多标签损失函数
class AsymmetricLossOptimized(nn.Module):
    '''
    Notice - This is an optimized version of AsymmetricLoss, created by Naveen Kumar.
    It reduces memory consumption by half and is slightly faster.
    '''
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos = one_sided_w * los_pos
            los_neg = one_sided_w * los_neg

        loss = - (los_pos + los_neg)
        return loss.mean()


class EarlyStopper:
    """早停：patience 个 epoch 内验证指标未改善 min_delta 以上则停止。"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, value):
        improved = (value < self.best_value - self.min_delta) if self.mode == 'min' else (value > self.best_value + self.min_delta)
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


_current_dataset_name = None
_is_set_by_main = False
def set_current_dataset(name: str):
    """
    【由 main.py 调用】设置全局数据集名称。
    """
    global _current_dataset_name, _is_set_by_main
    if name not in ['mimic3', 'mimic4']:
        print(f"警告 (config.py): 设置了未知的数据集名称 '{name}'，请检查 main.py 的 choices。")
    _current_dataset_name = name
    _is_set_by_main = True # 标记已被设置
    print(f"配置信息 (config.py): 数据集已由 main.py 设置为 -> {_current_dataset_name}")

def read_dataset_argument() -> str:
    """
    【供其他模块调用】读取当前设置的数据集名称。

    如果 main.py 尚未设置，会尝试从命令行独立解析一次作为备用，
    并使用与 main.py 一致的默认值 'mimic4'。
    """
    global _current_dataset_name 
    
    if _is_set_by_main:
        # 如果 main.py 已经设置过，直接返回存储的值
        return _current_dataset_name
    else:
        # --- 备用逻辑：如果 main.py 未设置 (例如单独运行此模块或导入顺序问题) ---
        print("警告 (config.py - read_dataset_argument): main.py 尚未调用 set_current_dataset()。将尝试独立解析命令行参数 '--dataset'。")
        
        # 创建一个临时的、独立的参数解析器 (与 main.py 的默认值保持一致)
        parser_temp = argparse.ArgumentParser(add_help=False)
        parser_temp.add_argument('--dataset', 
                                 type=str, 
                                 default="mimic4", # <-- 与 main.py 的默认值一致
                                 choices=['mimic3', 'mimic4'],
                                 help='临时解析器使用的帮助信息')

        # 解析已知的参数
        try:
             # 检查 sys.argv 是否包含参数（避免在某些环境中出错）
             cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
             args_known, _ = parser_temp.parse_known_args(cmd_args)
             # 将独立解析的结果也存起来，避免重复解析
             _current_dataset_name = args_known.dataset
             print(f"配置信息 (config.py - read_dataset_argument): 独立解析结果 -> {_current_dataset_name}")
             return _current_dataset_name
        except Exception as e:
             print(f"错误 (config.py - read_dataset_argument): 独立解析命令行失败: {e}。返回默认值 'mimic4'。")
             _current_dataset_name = "mimic4" # 解析失败时的最终回退
             return _current_dataset_name
# def read_dataset_argument() -> str:
#     """
#     专门解析并返回命令行中 '--dataset' 参数的值。
    
#     它会创建一个独立的解析器，只关注 '--dataset' 参数，
#     并使用 parse_known_args() 来忽略命令行中可能存在的其他参数，
#     从而避免与主解析器冲突。

#     Returns:
#         str: 命令行中指定的 dataset 名称 (例如 'mimic3' 或 'mimic4')，
#              如果未指定，则返回默认值 'mimic3'。
#     """
#     # 创建一个临时的、独立的参数解析器
#     parser_temp = argparse.ArgumentParser(add_help=False) # add_help=False 避免打印帮助信息

#     # 只定义我们关心的 '--dataset' 参数
#     parser_temp.add_argument('--dataset', 
#                              type=str, 
#                              default="mimic3", 
#                              choices=['mimic3', 'mimic4'],
#                              help='指定使用的数据集 (例如: mimic3 或 mimic4)')

#     # 解析已知的参数，忽略任何其他未在此处定义的参数
#     # sys.argv[1:] 用于获取命令行参数列表（排除脚本名称本身）
#     args_known, _ = parser_temp.parse_known_args(sys.argv[1:]) 

#     # 返回解析到的 dataset 值
#     return args_known.dataset