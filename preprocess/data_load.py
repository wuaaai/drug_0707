import os
import sys
import dill

# pickle 文件中的对象引用了顶层模块名（无 preprocess 前缀），需确保 preprocess 目录在 path 中
_preprocess_dir = os.path.dirname(os.path.abspath(__file__))
if _preprocess_dir not in sys.path:
    sys.path.insert(0, _preprocess_dir)

from preprocess.drug_recommendation_mimic34_fn import *
from preprocess.diag_prediction_mimic34_fn import *

from joblib import load

from preprocess.OverWrite_mimic3 import MIMIC3Dataset
from preprocess.OverWrite_mimic4 import MIMIC4Dataset


# 定义保存数据的函数
def save_preprocessed_data(data, filepath):
    print("Saving data...")
    dill.dump(data, open(filepath, 'wb'))
    print(f"Data saved to {filepath}")


def load_preprocessed_data(filepath):
    data = dill.load(open(filepath, 'rb'))
    print(f"Data loaded from {filepath}")
    return data


def load_dataset(dataset, root, tables=None, task_fn=None, dev=False):
    if dataset == 'mimic3':
        dataset = MIMIC3Dataset(
            root=root,
            dev=dev,
            tables=tables,
            # NDC->ATC3的编码映射
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                          "ICD9CM": "CCSCM",
                          "ICD9PROC": "CCSPROC"
                          },
            refresh_cache=True
        )

    elif dataset == 'mimic4':
        dataset = MIMIC4Dataset(
            root=root,
            dev=dev,
            tables=tables,
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
                "ICD10CM": "CCSCM",
                "ICD10PROC": "CCSPROC",
            },
            refresh_cache=False,
        )
    else:
        return load(root)

    return dataset.set_task(task_fn=task_fn)


def preprocess_data(args, task=None):
    """数据预处理，支持指定task，默认使用args.task"""
    if task is None:
        task = args.task  # 默认任务

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base, f"data/{args.dataset}/raw_data")
    if args.developer:
        processed_data_path = os.path.join(base, f'data/{args.dataset}/processed_data/{task}/processed_developer_data.pkl')
    else:
        processed_data_path = os.path.join(base, f'data/{args.dataset}/processed_data/{task}/processed_data.pkl')

    if os.path.exists(processed_data_path):
        task_dataset = load_preprocessed_data(processed_data_path)
    else:
        print(f"数据不存在，开始做{args.dataset}数据集在{task}任务的数据预处理")
        if args.dataset == 'mimic3':
            if task == 'drug_rec':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic3_fn,
                                            dev=args.developer)
            elif task == 'diag_pred':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic3_fn,
                                            dev=args.developer)
            elif task == 'diag_pred_1010':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic3_fn,
                                            dev=args.developer)
            elif task == 'drug_rec_ts':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic3_fn,
                                            dev=args.developer)
            elif task == 'diag_pred_ts':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic3_fn,
                                            dev=args.developer)
            else:
                raise ValueError("检查一下这个task")
        elif args.dataset == 'mimic4':
            if task == 'drug_rec':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic4_fn,
                                            dev=args.developer)
            elif task == 'diag_pred':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic4_fn,
                                            dev=args.developer)
            elif task == 'drug_rec_ts':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic4_fn,
                                            dev=args.developer)
            elif task == 'diag_pred_ts':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic4_fn,
                                            dev=args.developer)
            else:
                raise ValueError("检查一下这个task")
        else:
            raise ValueError("检查一下dataset，没有这个数据集")

        save_preprocessed_data(task_dataset, processed_data_path)

    return task_dataset
