from typing import List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import dill

def load_preprocessed_data(filepath):
    data = dill.load(open(filepath, 'rb'))
    print(f"Data loaded from {filepath}")
    return data

def jaccard_similarity(list1: List[int], list2: List[int]) -> float:
    """计算两个列表的Jaccard相似度"""
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:  
        # 如果并集为空，则相似度未定义，这里返回0或抛出异常都可以
        # 根据实际情况选择，这里我们选择返回0
        return 0.0
    jaccard_index = len(intersection) / len(union)
    return jaccard_index

def process_disc(list) -> List[float]:

    results = []  
    for i in range(len(list) - 1):
        # list去除第一个元素
        list1 = list[i][1:]
        list2 = list[i + 1][1:]
          
        jaccard = jaccard_similarity(list1, list2)
          
        results.append(jaccard)
    if len(results) == 0:
        results.append(0.0)
    return results

def draw_plt(results, filename):
    plt.hist(results, bins=len(results), edgecolor='black')  
    plt.title('Jaccard Similarity Distribution')  
    plt.xlabel('Jaccard Similarity')  
    plt.ylabel('Frequency')  
    plt.grid(True)  
    
    plt.savefig(filename)

def analysis():
    dataset = 'mimic3'
    task = 'diag_pred_1010'
    processed_data_path = f'../data/{dataset}/processed_data/{task}/processed_data.pkl'

    task_dataset = load_preprocessed_data(processed_data_path)
    samples = task_dataset.samples
    results_lab = []
    results_inj = []
    for s in tqdm(samples, total=len(samples), desc='samples_analysis'):
        lab_time_groups = s['lab_time_groups']
        inj_time_groups = s['inj_time_groups']

        visit_lab = []
        visit_inj = []
        for l in lab_time_groups:
            lab_avg = np.average(process_disc(l))
            visit_lab.append(lab_avg)
        for i in inj_time_groups:
            inj_avg = np.average(process_disc(i))
            visit_inj.append(inj_avg)

        vl = np.average(visit_lab)
        vi = np.average(visit_inj)

        results_lab.append(vl)
        results_inj.append(vi)

    draw_plt(results_lab, 'lab_jaccard.png')
    draw_plt(results_inj, 'inj_jaccard.png')


analysis()
