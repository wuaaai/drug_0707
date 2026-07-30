import numpy as np
import torch
import torch.nn.functional as F
import traceback
from pyhealth.metrics import binary_metrics_fn
from tqdm import tqdm
from torch import autograd
from itertools import zip_longest
# from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
from utils import prepare_labels, get_sample_loader, visit_level, code_level, calculate_sensitivity, \
    calculate_specificity, log_outmemory,AsymmetricLossOptimized


# def train(data_loader, model, label_tokenizer, optimizer, device):
#     train_loss = 0
#     for data in data_loader:
#         model.train()
#         optimizer.zero_grad()
#         if type(data) == dict:
#             label = prepare_labels(data['conditions'], label_tokenizer).to(device)
#         else:
#             label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
#         out = model(data)
#         loss = F.binary_cross_entropy_with_logits(out, label)
#         # y_prob = torch.sigmoid(out)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.detach().cpu().numpy()
#     return train_loss
# 在训练循环中修改以下部分（training函数中）：
from itertools import zip_longest
import torch
import torch.nn.functional as F
from tqdm import tqdm
import traceback


def _get_chapter_info_from_batch(data, model):
    """从 batch 数据提取路由所需信息。

    返回：
    - 章节分布（用于 TrajectoryRouter）
    - 轨迹特征（用于 TrajectoryAwareRouter，当 use_traj_router=True）
    - 就诊数（用于冷启动退火）

    数据集中的 conditions 是 CCS/CCSCM 编码，通过 map_ccs_to_expert 映射到 Expert ID。
    """
    if not hasattr(model, 'router'):
        return None, None

    from models.expert_selectv2 import map_ccs_to_expert
    from preprocess.trajectory_data_builder import batch_compute_trajectory_features

    if type(data) == dict:
        conditions_list = data.get('conditions', [])
        B = len(data.get('visit_id', []))
    else:
        conditions_list = data[0].get('conditions', []) if len(data) > 0 else []
        B = len(data)

    num_experts = model.num_chapters
    device = model.device

    # === 1. 构建历史就诊序列（排除最后一次就诊 = 当前预测目标） ===
    # conditions_list 格式: [[[v1_codes], [v2_codes], ..., [vn_codes]], ...]
    # 排除最后一次就诊，只使用历史就诊计算路由
    historical_conds = []
    num_visits_list = []
    for b in range(B):
        if type(data) == dict:
            patient_conds = conditions_list[b] if b < len(conditions_list) else []
        else:
            patient_conds = conditions_list[b] if b < len(conditions_list) else []

        if (isinstance(patient_conds, list) and len(patient_conds) > 0
                and isinstance(patient_conds[0], list)):
            hist = patient_conds[:-1]  # 排除当前就诊
        else:
            # 单就诊或非嵌套格式，全部使用
            hist = patient_conds if isinstance(patient_conds, list) else []
        historical_conds.append(hist if isinstance(hist, list) else [hist])
        num_visits_list.append(len(hist))

    # === 2. 计算章节分布（用于 TrajectoryRouter 和先验相似度） ===
    chapter_dists = torch.zeros(B, num_experts, device=device)
    for b in range(B):
        for visit_codes in historical_conds[b]:
            if not isinstance(visit_codes, list):
                continue
            for code in visit_codes:
                eid = map_ccs_to_expert(str(code))
                if 0 <= eid < num_experts:
                    chapter_dists[b, eid] += 1
        row_sum = chapter_dists[b].sum()
        if row_sum > 0:
            chapter_dists[b] /= row_sum

    # === 3. 计算轨迹特征（用于 TrajectoryAwareRouter） ===
    traj_features = None
    if hasattr(model, 'use_traj_router') and model.use_traj_router:
        # 将 historical_conds 转换为 batch_compute_trajectory_features 需要的格式
        # 函数要求: List of patients, each = list of visits, each = list of codes
        feat_input = [
            [visit_codes for visit_codes in patient_hist
             if isinstance(visit_codes, list)]
            for patient_hist in historical_conds
        ]
        feat_np = batch_compute_trajectory_features(feat_input, num_experts)
        traj_features = torch.tensor(feat_np, dtype=torch.float32, device=device)

    num_visits = torch.tensor(num_visits_list, dtype=torch.float32, device=device)
    return chapter_dists, traj_features, num_visits


def training(data_loader, model, label_tokenizer, optimizer, label_name, log_outmemory_txt_path, device):
    model.train()
    train_loss = 0
    asl_drug = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0).to(device)
    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            if type(data) == dict:
                label = prepare_labels(data[label_name], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

            # 开始检测是否有nan，会显著的增加运行时间
            # with autograd.detect_anomaly():
            try:
                # TrajectoryCare 需要章节分布和轨迹特征信息
                if hasattr(model, 'router'):
                    chapter_dist, traj_features, num_visits = _get_chapter_info_from_batch(data, model)
                    model_output = model(data, chapter_dist, num_visits, traj_features)
                else:
                    model_output = model(data)
                if isinstance(model_output, (tuple, list)):
                    out = model_output[0]   # 第一个是 logits
                    # expert_weights = model_output[1] # 训练时用不到
                    if isinstance(out, list):
                        out = out[0]
                else:
                    out = model_output      # 来自 GRU, Transformer 等
            except torch.cuda.OutOfMemoryError:
                error_log = traceback.format_exc()
                log_outmemory(data, error_log, log_outmemory_txt_path)
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                del data
                torch.cuda.empty_cache()
                continue
            # if torch.isnan(out).any() or torch.isinf(out).any():
            #     exit("前向输出中有NaN或Inf值！")
            # loss = F.binary_cross_entropy_with_logits(out, label)
            loss = asl_drug(out, label)
            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     exit("损失中有NaN或Inf值！")
            
            # # ✅ 加入 MoE 解耦损失（正交性 + 协方差）
            # if hasattr(model, "fc_patient") and hasattr(model.fc_patient, "get_orthogonality_loss"):
            #     orth_loss = model.fc_patient.get_orthogonality_loss()

            #     if hasattr(model.fc_patient, "_last_input"):  # 如果 forward 缓存了 x
            #         x_input = model.fc_patient._last_input
            #         disentangle_loss = model.fc_patient.get_disentangle_loss(x_input)
            #         loss = loss + 0.01 * orth_loss + 0.05 * disentangle_loss
            #     else:
            #         loss = loss + 0.01 * orth_loss
            # ✅ 加入 MoE 解耦损失（正交性约束）
            if hasattr(model, "fc_patient") and hasattr(model.fc_patient, "get_orthogonality_loss"):
                orth_loss = model.fc_patient.get_orthogonality_loss()
                loss = loss + 0.01 * orth_loss  # 调整这个权重超参数

            
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            avg_loss = train_loss / (batch_idx + 1)

            # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
            del data, out, loss
            torch.cuda.empty_cache()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return avg_loss

from itertools import zip_longest
import torch
import torch.nn.functional as F
from tqdm import tqdm
import traceback


def training_multi_task(data_loader_drug, data_loader_diag, model, label_tokenizer, optimizer, 
                        label_name_dict, log_outmemory_txt_path, device, loss_weights=(0.5, 0.5)):
    model.train()
    train_loss = 0
    batch_idx = 0
    asl_drug = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0).to(device)
    asl_diag = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1).to(device)
    with tqdm(total=max(len(data_loader_drug), len(data_loader_diag)), desc="Training multi-task", unit="batch") as pbar:
        for data_drug, data_diag in zip_longest(data_loader_drug, data_loader_diag):
            optimizer.zero_grad()
            try:
                current_loss = 0.0
                has_loss = False

                # 药物任务
                if data_drug is not None:
                    if isinstance(data_drug, dict):
                        label_drug = prepare_labels(data_drug[label_name_dict['drug_rec']], label_tokenizer[0]).to(device)
                    else:
                        label_drug = prepare_labels(data_drug[0][label_name_dict['drug_rec']], label_tokenizer[0]).to(device)

                    out_drug, _ = model(data_drug, task="drug")  
                    loss_drug = asl_drug(out_drug, label_drug)
                    current_loss += loss_weights[0] * loss_drug
                    has_loss = True

                    # 冻结任务1专家
                    # for param in model.fc_patient.task_experts[1].parameters():
                    #     param.requires_grad = False
                    # for param in model.fc_patient.task_experts[0].parameters():
                    #     param.requires_grad = True


                # 疾病任务
                if data_diag is not None:
                    if isinstance(data_diag, dict):
                        label_diag = prepare_labels(data_diag[label_name_dict['diag_pred']], label_tokenizer[1]).to(device)
                    else:
                        label_diag = prepare_labels(data_diag[0][label_name_dict['diag_pred']], label_tokenizer[1]).to(device)

                    out_diag, _ = model(data_diag, task="diag")  
                    loss_diag = asl_diag(out_diag, label_diag)
                    current_loss += loss_weights[1] * loss_diag
                    has_loss = True

                    # 冻结任务0专家
                    # for param in model.fc_patient.task_experts[0].parameters():
                    #     param.requires_grad = False
                    # for param in model.fc_patient.task_experts[1].parameters():
                    #     param.requires_grad = True


                # 正交约束
                if has_loss and hasattr(model, "get_orthogonality_loss"):
                    orth_loss = model.get_orthogonality_loss()
                    current_loss += 0.01 * orth_loss

                if has_loss:
                    current_loss.backward()
                    optimizer.step()
                    train_loss += current_loss.item()
                    batch_idx += 1
                    avg_loss = train_loss / batch_idx
                else:
                    avg_loss = train_loss / (batch_idx + 1e-8)

            except torch.cuda.OutOfMemoryError:
                error_log = traceback.format_exc()
                log_outmemory({"drug_batch": data_drug, "diag_batch": data_diag}, error_log, log_outmemory_txt_path)
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                del data_drug, data_diag
                torch.cuda.empty_cache()
                continue

            # 清理显存
            if data_drug is not None:
                del data_drug, out_drug, label_drug, loss_drug
            if data_diag is not None:
                del data_diag, out_diag, label_diag, loss_diag
            if has_loss:
                del current_loss
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return avg_loss




def evaluating(data_loader, model, label_tokenizer, label_name, device):
    model.eval()
    val_loss = 0
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, data in enumerate(data_loader):
                if type(data) == dict:
                    label = prepare_labels(data[label_name], label_tokenizer).to(device)
                else:
                    label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

                model_output = model(data)
                # 检查输出是否为元组 (来自 Multi_DT)
                if isinstance(model_output, (tuple, list)):
                    out = model_output[0]   # 第一个是 logits
                    if isinstance(out, list):
                        out = out[0]
                else:
                    out = model_output      # 来自 GRU, Transformer 等
                loss = F.binary_cross_entropy_with_logits(out, label)

                val_loss += loss.detach().cpu().numpy()
                avg_loss = val_loss / (batch_idx + 1)

                y_t = label.cpu().numpy()
                y_p = torch.sigmoid(out).detach().cpu().numpy()
                y_t_all.append(y_t)
                y_p_all.append(y_p)

                # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
                del data, out, loss
                torch.cuda.empty_cache()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            y_true = np.concatenate(y_t_all, axis=0)
            y_prob = np.concatenate(y_p_all, axis=0)
            y_pred = np.where(y_prob > 0.5, 1, 0)

            code_level_results = code_level(y_true, y_prob)
            visit_level_results = visit_level(y_true, y_prob)

            sensitivity = calculate_sensitivity(y_true, y_pred)
            specificity = calculate_specificity(y_true, y_pred)
            sensitivity = np.mean(sensitivity)
            specificity = np.mean(specificity)

            y_true = y_true.ravel()
            y_prob = y_prob.ravel()

            metrics = binary_metrics_fn(y_true, y_prob,
                                        metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    return avg_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluating_multi_task(data_loader_drug, data_loader_diag, model, label_tokenizer, label_name_dict, device):
    """
    多任务评估函数（双 DataLoader 版本）
    label_name_dict: dict，例如 {'drug_rec': 'drugs', 'diag_pred': 'conditions'}
    """
    model.eval()
    val_loss_drug, val_loss_diag = 0, 0
    y_t_all_drug, y_p_all_drug = [], []
    y_t_all_diag, y_p_all_diag = [], []

    with torch.no_grad():
        # 双 DataLoader 同步迭代
        with tqdm(total=min(len(data_loader_drug), len(data_loader_diag)),
                  desc="Evaluating multi-task", unit="batch") as pbar:
            for batch_idx, (data_drug, data_diag) in enumerate(zip(data_loader_drug, data_loader_diag)):

                # ========= 药物任务标签 =========
                if isinstance(data_drug, dict):
                    label_drug = prepare_labels(data_drug[label_name_dict['drug_rec']], label_tokenizer[0]).to(device)
                else:
                    label_drug = prepare_labels(data_drug[0][label_name_dict['drug_rec']], label_tokenizer[0]).to(device)

                # ========= 疾病任务标签 =========
                if isinstance(data_diag, dict):
                    label_diag = prepare_labels(data_diag[label_name_dict['diag_pred']], label_tokenizer[1]).to(device)
                else:
                    label_diag = prepare_labels(data_diag[0][label_name_dict['diag_pred']], label_tokenizer[1]).to(device)

                # ========= 模型前向（分别执行两个任务） =========
                out_drug, _ = model(data_drug, task="drug")
                out_diag, _ = model(data_diag, task="diag")

                # ========= 计算两个任务的二进制交叉熵损失 =========
                loss_drug = F.binary_cross_entropy_with_logits(out_drug, label_drug)
                loss_diag = F.binary_cross_entropy_with_logits(out_diag, label_diag)

                val_loss_drug += loss_drug.item()
                val_loss_diag += loss_diag.item()

                avg_loss_drug = val_loss_drug / (batch_idx + 1)
                avg_loss_diag = val_loss_diag / (batch_idx + 1)

                # 收集预测和真实标签（转到CPU，方便后续指标计算）
                y_t_all_drug.append(label_drug.cpu().numpy())
                y_p_all_drug.append(torch.sigmoid(out_drug).cpu().numpy())

                y_t_all_diag.append(label_diag.cpu().numpy())
                y_p_all_diag.append(torch.sigmoid(out_diag).cpu().numpy())

                # 清理显存
                del data_drug, data_diag, out_drug, out_diag, loss_drug, loss_diag
                torch.cuda.empty_cache()

                pbar.update(1)
                pbar.set_postfix(avg_loss_drug=f"{avg_loss_drug:.4f}", avg_loss_diag=f"{avg_loss_diag:.4f}")

    # 合并所有batch的标签和预测
    y_true_drug = np.concatenate(y_t_all_drug, axis=0)
    y_prob_drug = np.concatenate(y_p_all_drug, axis=0)
    y_pred_drug = (y_prob_drug > 0.5).astype(int)

    y_true_diag = np.concatenate(y_t_all_diag, axis=0)
    y_prob_diag = np.concatenate(y_p_all_diag, axis=0)
    y_pred_diag = (y_prob_diag > 0.5).astype(int)

    # 分别计算两个任务的指标
    code_level_results_drug = code_level(y_true_drug, y_prob_drug)
    visit_level_results_drug = visit_level(y_true_drug, y_prob_drug)
    sensitivity_drug = np.mean(calculate_sensitivity(y_true_drug, y_pred_drug))
    specificity_drug = np.mean(calculate_specificity(y_true_drug, y_pred_drug))
    metrics_drug = binary_metrics_fn(y_true_drug.ravel(), y_prob_drug.ravel(),
                                     metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    code_level_results_diag = code_level(y_true_diag, y_prob_diag)
    visit_level_results_diag = visit_level(y_true_diag, y_prob_diag)
    sensitivity_diag = np.mean(calculate_sensitivity(y_true_diag, y_pred_diag))
    specificity_diag = np.mean(calculate_specificity(y_true_diag, y_pred_diag))
    metrics_diag = binary_metrics_fn(y_true_diag.ravel(), y_prob_diag.ravel(),
                                     metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    val_loss_dict = {
        "drug_rec": avg_loss_drug,
        "diag_pred": avg_loss_diag
    }
    metrics_dict = {
        "drug_rec": metrics_drug,
        "diag_pred": metrics_diag
    }
    code_level_results_dict = {
        "drug_rec": code_level_results_drug,
        "diag_pred": code_level_results_diag
    }
    visit_level_results_dict = {
        "drug_rec": visit_level_results_drug,
        "diag_pred": visit_level_results_diag
    }
    sensitivity_dict = {
        "drug_rec": sensitivity_drug,
        "diag_pred": sensitivity_diag
    }
    specificity_dict = {
        "drug_rec": specificity_drug,
        "diag_pred": specificity_diag
    }

    return val_loss_dict, metrics_dict, code_level_results_dict, visit_level_results_dict, sensitivity_dict, specificity_dict


def testing(data_loader, test_epochs, model, label_tokenizer, sample_size, label_name, device):
    results = []
    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        sample_loader = get_sample_loader(data_loader, sample_size)

        _, metrics, code_level_results, visit_level_results, sensitivity, specificity = evaluating(sample_loader, model,
                                                                                                   label_tokenizer,
                                                                                                   label_name, device)

        # 打印结果
        print(f'F1: {metrics["f1"]:.4f}, '
              f'Jaccard: {metrics["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics["pr_auc"]:.4f}, '
              f'sensitivity: {sensitivity:.4f}, '
              f'specificity: {specificity:.4f}'
              )

        results.append([metrics["f1"], metrics["jaccard"], metrics["roc_auc"],
                        metrics["pr_auc"]] + list(code_level_results) + list(visit_level_results) +
                       [sensitivity, specificity])

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)
    metric_list = ['F1', 'Jaccard', 'ROC-AUC', 'PR-AUC',
                   'code-10', 'code-20', 'code-30', 'code-40', 'code-50', 'code-60', 'code-70', 'code-80',
                   'visit-10', 'visit-20', 'visit-30', 'visit-40', 'visit-50', 'visit-60', 'visit-70', 'visit-80',
                   'sensitivity', 'specificity']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])

    return outstring
import numpy as np


def testing_multi_task(drug_loader, diag_loader, test_epochs, model, label_tokenizer, sample_size, label_name_dict, device):
    results_drug = []
    results_diag = []

    metric_list = ['F1', 'Jaccard', 'ROC-AUC', 'PR-AUC',
                   'code-10', 'code-20', 'code-30', 'code-40', 'code-50', 'code-60', 'code-70', 'code-80',
                   'visit-10', 'visit-20', 'visit-30', 'visit-40', 'visit-50', 'visit-60', 'visit-70', 'visit-80',
                   'sensitivity', 'specificity']

    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        sample_drug_loader = get_sample_loader(drug_loader, sample_size)
        sample_diag_loader = get_sample_loader(diag_loader, sample_size)

        val_loss_dict, metrics_dict, code_level_results_dict, visit_level_results_dict, sensitivity_dict, specificity_dict = \
            evaluating_multi_task(sample_drug_loader, sample_diag_loader, model, label_tokenizer, label_name_dict, device)

        # 打印药物推荐任务指标
        print(f'[Drug Prediction] F1: {metrics_dict["drug_rec"]["f1"]:.4f}, '
              f'Jaccard: {metrics_dict["drug_rec"]["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics_dict["drug_rec"]["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics_dict["drug_rec"]["pr_auc"]:.4f}, '
              f'sensitivity: {sensitivity_dict["drug_rec"]:.4f}, '
              f'specificity: {specificity_dict["drug_rec"]:.4f}')

        # 打印疾病预测任务指标
        print(f'[Diagnosis Prediction] F1: {metrics_dict["diag_pred"]["f1"]:.4f}, '
              f'Jaccard: {metrics_dict["diag_pred"]["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics_dict["diag_pred"]["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics_dict["diag_pred"]["pr_auc"]:.4f}, '
              f'sensitivity: {sensitivity_dict["diag_pred"]:.4f}, '
              f'specificity: {specificity_dict["diag_pred"]:.4f}')

        # 记录药物推荐任务结果
        results_drug.append(
            [metrics_dict["drug_rec"]["f1"], metrics_dict["drug_rec"]["jaccard"], metrics_dict["drug_rec"]["roc_auc"],
             metrics_dict["drug_rec"]["pr_auc"]] + list(code_level_results_dict["drug_rec"]) + list(visit_level_results_dict["drug_rec"]) +
            [sensitivity_dict["drug_rec"], specificity_dict["drug_rec"]]
        )

        # 记录疾病预测任务结果
        results_diag.append(
            [metrics_dict["diag_pred"]["f1"], metrics_dict["diag_pred"]["jaccard"], metrics_dict["diag_pred"]["roc_auc"],
             metrics_dict["diag_pred"]["pr_auc"]] + list(code_level_results_dict["diag_pred"]) + list(visit_level_results_dict["diag_pred"]) +
            [sensitivity_dict["diag_pred"], specificity_dict["diag_pred"]]
        )

    # 转numpy计算均值和标准差
    results_drug = np.array(results_drug)
    results_diag = np.array(results_diag)

    mean_drug, std_drug = results_drug.mean(axis=0), results_drug.std(axis=0)
    mean_diag, std_diag = results_diag.mean(axis=0), results_diag.std(axis=0)

    # 生成输出字符串
    outstring_drug = '【Drug Prediction】\n' + ''.join([
        "{}:\t{:.4f} ± {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean_drug, std_drug))
    ])

    outstring_diag = '【Diagnosis Prediction】\n' + ''.join([
        "{}:\t{:.4f} ± {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean_diag, std_diag))
    ])

    return outstring_drug, outstring_diag


# 定义一个函数，用来加载指定任务的部分权重，更新模型权重
def load_partial_state_dict(model, partial_state_dict):
    model_state = model.state_dict()
    for k, v in partial_state_dict.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state)