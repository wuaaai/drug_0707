"""
轨迹原型发现——谱聚类实现 + 多指标 K 选择。

核心思路：
1. 从章节转移矩阵构建章节相似图
2. 多种 K 选择方法（Eigengap, Silhouette, Davies-Bouldin, Gap Statistic）
3. 结合临床先验确定最优 K
4. 谱聚类将具有相似入边/出边分布的章节合并为轨迹原型
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import json
import warnings

# ICD-9 章节的临床先验分组（基于已知共病通路）
CLINICAL_PRIOR_GROUPS = [
    {'name': '心肾代谢轴', 'chapters': [2, 7, 9]},       # C3(内分泌)+C8(循环)+C10(消化)
    {'name': '呼吸-感染轴', 'chapters': [0, 8]},          # C1(感染)+C9(呼吸)
    {'name': '神经-精神轴', 'chapters': [4, 5]},           # C5(精神)+C6(神经)
    {'name': '肿瘤轴', 'chapters': [1]},                   # C2(肿瘤)
    {'name': '损伤-外因轴', 'chapters': [16, 17]},         # C17(症状)+C18(损伤)
    {'name': '肌肉骨骼-皮肤轴', 'chapters': [12, 11]},     # C13(骨骼)+C12(皮肤)
    {'name': '妇产-围产-先天轴', 'chapters': [14, 15, 10]},# C15(先天)+C16(围产)+C11(泌尿)
]
# 临床先验建议的 K 范围
CLINICAL_K_MIN = len(CLINICAL_PRIOR_GROUPS)  # 7
CLINICAL_K_MAX = len(CLINICAL_PRIOR_GROUPS) + 3  # 10


def compute_gap_statistic(embeddings: np.ndarray, k: int, n_ref: int = 10, random_state: int = 42) -> float:
    """计算 Gap Statistic——比较聚类紧密度与随机均匀分布的期望。"""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    if len(set(labels)) < 2:
        return -np.inf

    # 实际数据的类内离散度
    disp_real = 0
    for label in set(labels):
        cluster_points = embeddings[labels == label]
        center = cluster_points.mean(axis=0)
        disp_real += np.sum((cluster_points - center) ** 2)
    disp_real = np.log(disp_real)

    # 参考分布（均匀分布）的期望离散度
    ref_disps = []
    rng = np.random.RandomState(random_state)
    for _ in range(n_ref):
        ref_data = rng.uniform(
            embeddings.min(axis=0), embeddings.max(axis=0), embeddings.shape
        )
        ref_labels = kmeans.fit_predict(ref_data)
        disp_ref = 0
        for label in set(ref_labels):
            cluster_points = ref_data[ref_labels == label]
            center = cluster_points.mean(axis=0)
            disp_ref += np.sum((cluster_points - center) ** 2)
        ref_disps.append(np.log(disp_ref))

    gap = np.mean(ref_disps) - disp_real
    return gap


def compute_eigengap(affinity_matrix: np.ndarray) -> List[float]:
    """计算拉普拉斯矩阵的特征值间隙。"""
    from sklearn.preprocessing import normalize
    # 归一化拉普拉斯: L = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(affinity_matrix.sum(axis=1) + 1e-8))
    L_norm = np.eye(len(affinity_matrix)) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(L_norm)
    eigenvalues.sort()
    gaps = np.diff(eigenvalues)
    return gaps.tolist()


def select_optimal_k(
    embeddings: np.ndarray,
    min_k: int = 4,
    max_k: int = 12,
    random_state: int = 42
) -> Tuple[int, Dict]:
    """综合多指标选择最优 K。

    使用 4 种指标：
    1. Eigengap（谱间隙）
    2. Silhouette Score（轮廓系数）
    3. Davies-Bouldin Index（戴维斯-博尔丁指数，越小越好）
    4. Gap Statistic（间隙统计量）

    每种指标给出一个最优 K 候选，然后投票 + 临床先验约束确定最终 K。

    Returns:
        best_k: 最优 K
        info: 各指标详情
    """
    C = embeddings.shape[0]
    valid_mask = embeddings.sum(axis=1) > 0
    valid_embeddings = embeddings[valid_mask]
    n_valid = valid_embeddings.shape[0]

    effective_max_k = min(max_k, n_valid - 1)
    effective_min_k = max(min_k, 2)

    if effective_max_k < effective_min_k:
        return 2, {'error': f'Not enough valid chapters: {n_valid}'}

    candidates = {}
    metric_details = {}

    # 1. Eigengap
    from sklearn.metrics.pairwise import rbf_kernel
    affinity = rbf_kernel(valid_embeddings)
    gaps = compute_eigengap(affinity)
    # Eigengap 建议：最大间隙位置（排除第一个间隙，通常噪声）
    gap_indices = list(range(effective_min_k, min(effective_max_k, len(gaps))))
    if gap_indices:
        best_gap_k = max(gap_indices, key=lambda i: gaps[i - 1] if i - 1 < len(gaps) else 0)
        candidates['eigengap'] = best_gap_k
        metric_details['eigengap'] = {'k': best_gap_k, 'gap_value': gaps[best_gap_k - 1] if best_gap_k - 1 < len(gaps) else 0}

    # 2-4. Silhouette, DB, Gap
    best_silhouette = -1
    best_db = float('inf')
    best_gap = -float('inf')
    best_k_sil = effective_min_k
    best_k_db = effective_min_k
    best_k_gap = effective_min_k

    for k in range(effective_min_k, effective_max_k + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(valid_embeddings)
                if len(set(labels)) < 2:
                    continue

                sil = silhouette_score(valid_embeddings, labels)
                db = davies_bouldin_score(valid_embeddings, labels)
                gap = compute_gap_statistic(valid_embeddings, k, random_state=random_state)

                metric_details[f'K={k}'] = {'silhouette': float(sil), 'db': float(db), 'gap': float(gap)}

                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k_sil = k
                if db < best_db:
                    best_db = db
                    best_k_db = k
                if gap > best_gap:
                    best_gap = gap
                    best_k_gap = k
        except Exception:
            continue

    candidates['silhouette'] = best_k_sil
    candidates['db'] = best_k_db
    candidates['gap'] = best_k_gap

    # 投票机制：取各指标建议的中位数，约束到临床先验范围
    votes = list(candidates.values())
    median_k = int(np.median(votes))

    # 临床先验约束
    if CLINICAL_K_MIN <= median_k <= CLINICAL_K_MAX:
        best_k = median_k
    elif median_k < CLINICAL_K_MIN:
        best_k = CLINICAL_K_MIN  # 不少于已知临床轴数
    else:
        best_k = min(CLINICAL_K_MAX, median_k)

    info = {
        'candidates': candidates,
        'metric_details': metric_details,
        'median_k': median_k,
        'best_k': best_k,
        'clinical_k_range': [CLINICAL_K_MIN, CLINICAL_K_MAX],
        'clinical_groups': [g['name'] for g in CLINICAL_PRIOR_GROUPS],
        'selection_method': '投票(median of 4 metrics) + 临床先验约束',
    }
    return best_k, info


def build_chapter_embedding(trans_matrix: np.ndarray) -> np.ndarray:
    """从转移矩阵构建章节嵌入：拼接入边和出边分布。

    Args:
        trans_matrix: shape (C, C)，trans_matrix[i, j] = P(j | i)

    Returns:
        embeddings: shape (C, 2*C)，每行 = [入边分布, 出边分布]
    """
    C = trans_matrix.shape[0]
    # 入边分布：每列归一化 = P(从各章节来 | 到达此章节)
    in_edges = trans_matrix.sum(axis=0)
    in_dist = trans_matrix / (in_edges + 1e-8)  # (C, C)，列归一化
    in_embed = in_dist.T  # (C, C)，每行 = 该章节的入边分布

    # 出边分布：每行已归一化 = P(去往各章节 | 从此章节出发)
    out_embed = trans_matrix.copy()

    # 拼接
    embeddings = np.concatenate([in_embed, out_embed], axis=1)
    return embeddings


def discover_prototypes_spectral(
    trans_matrix: np.ndarray,
    min_k: int = 4,
    max_k: int = 12,
    random_state: int = 42
) -> Tuple[np.ndarray, int, Dict]:
    """使用谱聚类发现轨迹原型，K 通过多指标综合+临床先验确定。

    Args:
        trans_matrix: 非对称章节转移矩阵 (C, C)
        min_k, max_k: K 搜索范围
        random_state: 随机种子

    Returns:
        chapter_labels: (C,) 每个章节的原型标签
        best_k: 最优原型数
        info: K 选择详情 + 原型描述
    """
    embeddings = build_chapter_embedding(trans_matrix)
    C = embeddings.shape[0]
    valid_mask = embeddings.sum(axis=1) > 0
    valid_indices = np.where(valid_mask)[0]

    # Step 1: 多指标 K 选择
    best_k, k_info = select_optimal_k(embeddings, min_k, max_k, random_state)
    print(f"K selection: candidates={k_info.get('candidates', {})}, best_k={best_k}")
    print(f"  Method: {k_info.get('selection_method', 'unknown')}")
    print(f"  Clinical prior groups: {k_info.get('clinical_groups', [])}")

    # Step 2: 用最优 K 做谱聚类
    all_labels = np.full(C, -1, dtype=int)
    try:
        clustering = SpectralClustering(
            n_clusters=best_k,
            affinity='nearest_neighbors',
            n_neighbors=min(10, len(valid_indices) - 1),
            random_state=random_state,
            assign_labels='kmeans',
        )
        labels_valid = clustering.fit_predict(embeddings[valid_indices])
        all_labels[valid_indices] = labels_valid
    except Exception:
        # 退化为 KMeans
        kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        labels_valid = kmeans.fit_predict(embeddings[valid_indices])
        all_labels[valid_indices] = labels_valid

    # 构建原型信息
    prototypes = {}
    for label in range(best_k):
        chapter_indices = np.where(all_labels == label)[0]
        prototypes[int(label)] = {
            'chapters': [f'C{i + 1}' for i in chapter_indices],
            'num_chapters': len(chapter_indices),
        }

    info = {
        'best_k': best_k,
        'k_selection': k_info,
        'prototypes': prototypes,
        'chapter_labels': all_labels.tolist(),
    }
    return all_labels, best_k, info


def assign_patients_to_prototypes(
    all_seqs: List[List[List[str]]],
    chapter_labels: np.ndarray,
    trans_matrix: np.ndarray
) -> np.ndarray:
    """将患者分配到轨迹原型（基于硬投票：患者最常见的章节属于哪个原型）。

    Args:
        all_seqs: 每位患者的章节块序列
        chapter_labels: (C,) 每个章节的原型标签
        trans_matrix: 转移矩阵（用于加权）

    Returns:
        patient_prototypes: (P,) 每位患者的主要原型
        patient_prototype_dists: (P, K) 每位患者的原型分布（软分配）
    """
    K = len(set(chapter_labels)) - (1 if -1 in chapter_labels else 0)
    P = len(all_seqs)
    patient_prototype_dists = np.zeros((P, K))

    def chapter_idx(ch: str) -> int:
        return int(ch[1:]) - 1

    for i, patient_seq in enumerate(all_seqs):
        for visit_chapters in patient_seq:
            for ch in visit_chapters:
                c_idx = chapter_idx(ch)
                if c_idx < len(chapter_labels):
                    label = chapter_labels[c_idx]
                    if label >= 0:
                        patient_prototype_dists[i, label] += 1

        # 归一化
        row_sum = patient_prototype_dists[i].sum()
        if row_sum > 0:
            patient_prototype_dists[i] /= row_sum

    patient_prototypes = patient_prototype_dists.argmax(axis=1)
    return patient_prototypes, patient_prototype_dists


def save_prototypes(info: Dict, filepath: str):
    """保存轨迹原型到 JSON 文件。"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def load_prototypes(filepath: str) -> Dict:
    """从 JSON 文件加载轨迹原型。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
