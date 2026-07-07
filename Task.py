from utils import get_init_tokenizers
from pyhealth.tokenizer import Tokenizer


def initialize_task(task_dataset, args):
    """任务定义"""
    if args.task == 'drug_rec':
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['conditions', 'procedures', 'drugs_hist'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('drugs'))
        label_size = len(task_dataset.get_all_tokens('drugs'))
    elif args.task == 'diag_pred':
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['cond_hist', 'procedures', 'drugs'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))
        label_size = len(task_dataset.get_all_tokens('conditions'))
    elif args.task == 'diag_pred_1010':
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['cond_hist', 'procedures', 'drugs'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt', 'lab_inj_groups'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))
        label_size = len(task_dataset.get_all_tokens('conditions'))
    elif args.task == 'drug_rec_ts':
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['conditions', 'procedures', 'drugs_hist'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt', 'lab_inj_groups', 'lab_inj_merged_list'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('drugs'))
        label_size = len(task_dataset.get_all_tokens('drugs'))
    elif args.task == 'multi_task':
        # 分别初始化两个任务的 tokenizer 和 label 信息
        # 加载drug_rec数据集
        drug_dataset = get_dataset(args, task="drug_rec")
        Tokenizers_visit_event_drug = get_init_tokenizers(drug_dataset, keys=['conditions', 'procedures', 'drugs_hist'])
        Tokenizers_monitor_event_drug = get_init_tokenizers(drug_dataset,
                                                            keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt'])
        label_tokenizer_drug = Tokenizer(tokens=drug_dataset.get_all_tokens('drugs'))
        label_size_drug = len(drug_dataset.get_all_tokens('drugs'))

        # 加载diag_pred数据集
        diag_dataset = get_dataset(args, task="diag_pred")
        Tokenizers_visit_event_diag = get_init_tokenizers(diag_dataset, keys=['cond_hist', 'procedures', 'drugs'])
        Tokenizers_monitor_event_diag = get_init_tokenizers(diag_dataset,
                                                            keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt'])
        label_tokenizer_diag = Tokenizer(tokens=diag_dataset.get_all_tokens('conditions'))
        label_size_diag = len(diag_dataset.get_all_tokens('conditions'))

        return (
            (Tokenizers_visit_event_drug, Tokenizers_monitor_event_drug, label_tokenizer_drug, label_size_drug),
            (Tokenizers_visit_event_diag, Tokenizers_monitor_event_diag, label_tokenizer_diag, label_size_diag)
        )

    else:
        raise ValueError('没有这个任务')

    return Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size
