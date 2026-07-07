import argparse
from Task import initialize_task
from utils import *
from baselines.baselines import *
from trainer import training, evaluating, testing
from preprocess.data_load import preprocess_data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime
from models.Base2_1 import Base2_1
import os
import time


def main(args):
    if args.developer:
        args.epochs = 3
        args.test_epochs = 2
        args.batch_size = 2
        
    set_random_seed(args.seed)
    print('{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # 数据读取
    task_dataset = preprocess_data(args)

    # 任务定义
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(task_dataset, args)

    # 切分数据
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)

    # 模型定义
    if args.model == 'Base2_1':
        model = Base2_1(Tokenizers_visit_event, Tokenizers_monitor_event, label_size, device, dropout=args.dropout)
    else:
        print("没有这个模型")
        return

    if args.task == "drug_rec":
        label_name = 'drugs'
    elif args.task == "drug_rec_ts":
        label_name = 'drugs'
    elif args.task == "diag_pred_ts":
        label_name = 'conditions'
    else:
        label_name = 'conditions'

    # 打印数据集的统计信息
    dataset_output = print_dataset_parameters(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, label_size, args)
    print('parameter of dataset:', dataset_output)

    # 打印模型参数的，需要可以开
    # print('number of parameters', count_parameters(model))
    # print_model_parameters(model)

    # 保存checkpoint的路径（按日期分类）
    log_date = datetime.now().strftime("%Y%m%d")
    folder_path = 'logs/{}/{}_{}_batchsize_{}_epochs_{}_{}'.format(log_date, args.model, args.dataset, args.batch_size, args.epochs, args.notes)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'
    log_outmemory_txt_path = f'{folder_path}/log_outmemory.txt'

    jaccard_ckpt_path = f'{folder_path}/best_model_jaccard.ckpt'
    final_jaccard_model_log = f'{folder_path}/final_result_jaccard.txt'

    if not args.test:
        # 记录 loss 的列表
        epoch_list = []
        train_losses = []
        val_losses = []

        log_params(dataset_output, args, log_txt_path)

        print('--------------------Begin Training--------------------')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if args.scheduler:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # 早停
        early_stopper = EarlyStopper(patience=args.patience, min_delta=0.0001, mode='min')
        best = float('inf')  # 无限大
        best_jaccard = float('-inf')
        best_model = None
        best_model_jaccard = None
        for epoch in range(args.epochs):
            start_time = time.time()
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            model = model.to(device)

            train_loss = training(args, train_loader, model, label_tokenizer, optimizer, label_name, log_outmemory_txt_path, device)
            val_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity \
                = evaluating(args, val_loader, model, label_tokenizer, label_name, device)
            
            if early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # 保存最佳模型
            if val_loss == early_stopper.best_value:
                best_model = model.state_dict()
            
            # 跟踪Jaccard指标
            if metrics["jaccard"] > best_jaccard:
                best_jaccard = metrics["jaccard"]
                best_model_jaccard = model.state_dict()

            end_time = time.time()
            run_time = end_time - start_time

            # 对两个ndarray进行格式化
            code_level_results = ', '.join(map(lambda x: f"{x:.4f}", code_level_results))
            visit_level_results = ', '.join(map(lambda x: f"{x:.4f}", visit_level_results))

            # 打印结果
            print(f'F1: {metrics["f1"]:.4f}, '
                  f'Jaccard: {metrics["jaccard"]:.4f}, '
                  f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                  f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                  f'code_level: {code_level_results}, '
                  f'visit_level: {visit_level_results},'
                  f'sensitivity: {sensitivity}, '
                  f'specificity: {specificity}'
                  )

            # 记录结果到 log.txt
            log_results(epoch, run_time, train_loss, val_loss, metrics, log_txt_path)

            # if val_loss < best:
            #     best = val_loss
            #     best_model = model.state_dict()
            
            # if metrics["jaccard"] > best_jaccard:
            #     best_jaccard = metrics["jaccard"]
            #     best_model_jaccard = model.state_dict()

            # if (epoch + 1) % 20 == 0:
            #     torch.save(best_model, ckpt_path)
            #     torch.save(best_model_jaccard, jaccard_ckpt_path)

            # 每个epoch都保存最佳模型
            torch.save(best_model, ckpt_path)
            torch.save(best_model_jaccard, jaccard_ckpt_path)

            # 记录损失
            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 每个epoch都绘制一次，绘制损失曲线
            plot_losses(epoch_list, train_losses, val_losses, png_path)

            # 学习率递减
            if args.scheduler:
                scheduler.step()

        # 这里本来可以每个epoch都保存一次，但是太大了，所以只保存一次
        torch.save(best_model, ckpt_path)
        torch.save(best_model_jaccard, jaccard_ckpt_path)

    print('--------------------Begin Testing--------------------')
    # 读取最新的model
    best_model = torch.load(ckpt_path)
    model.load_state_dict(best_model)
    model = model.to(device)

    # 开始测试
    sample_size = 0.8  # 国际惯例选取0.8
    outstring = testing(args, test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device)

    # 输出结果
    print("\nFinal test result:")
    print(outstring)
    with open(txt_path, 'w+') as file:
        file.write("model_path:")
        file.write(ckpt_path)
        file.write('\n')
        file.write(outstring)
    

    # 读取最新的model_jaccard
    best_model_jaccard = torch.load(jaccard_ckpt_path)
    model.load_state_dict(best_model_jaccard)
    model = model.to(device)

    outstring_jaccard = testing(args, test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device)

    # 输出结果
    print("\nFinal test result(jaccard):")
    print(outstring_jaccard)
    with open(final_jaccard_model_log, 'w+') as file:
        file.write("model_path:")
        file.write(jaccard_ckpt_path)
        file.write('\n')
        file.write(outstring_jaccard)


if __name__ == '__main__':
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
    parser.add_argument('--model', type=str, default="Base2_1",
                        help='Base2_1')
    parser.add_argument('--device_id', type=int, default=0, help="选gpu编号的")
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--task', type=str, default="drug_rec_ts", choices=['drug_rec', 'diag_pred_ts', 'drug_rec_ts'])
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dim', type=int, default=128, help='embedding dim')
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
    parser.add_argument('--developer', type=bool, default=False, help='developer mode')
    parser.add_argument('--test', type=bool, default=False, help='test mode')
    parser.add_argument('--notes', type=str, default=f"onlyvisit_{current_date}", help='notes')
    parser.add_argument("--scheduler", type=bool, default=True, help="scheduler mode")
    parser.add_argument("--gamma", type=float, default=0.2, help="scheduler parameter")
    parser.add_argument("--step_size", type=int, default=50, help="step_size")
    parser.add_argument('--patience', type=int, default=100, help='patience of earlystopper')
    parser.add_argument("--trans_num_heads", type=int, default=2, help="trans_num_heads")
    parser.add_argument("--trans_num_layers", type=int, default=2, help="trans_num_layers")
    parser.add_argument("--coef_loss_cvc", type=float, default=0.3, help="coefficient of cvc loss")
    args = parser.parse_args()

    main(args)
