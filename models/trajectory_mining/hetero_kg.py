"""
异构时序知识图谱 (Heterogeneous Temporal Knowledge Graph) —— 路线C

医学创新：现有 Jensen 法只建模"诊断→诊断"的纯自然演化，忽略了治疗干预。
但现实中，第二次就诊的诊断往往是被第一次的治疗（用药/手术）**改变过**的。

本模块构建 诊断(D) / 手术(P) / 用药(M) 三类节点的异构时序图：
- 同次就诊共现边: D↔M, D↔P  (本次就诊开的药/做的手术对应哪些诊断)
- 跨就诊演化边: 前一次就诊的所有事件 → 后一次就诊的诊断  (前因后果)

然后用医学元路径挖掘治疗演化模式：
- D→D   : 纯自然演化（Jensen 基线）
- D→M→D : 诊断→用药→下一诊断   （药物治疗如何改变病程）
- D→P→D : 诊断→手术→下一诊断   （手术干预后的病程走向）

核心价值：把"怎么治疗的"纳入"病是怎么来的"，两个诊断被聚到同一原型，
不仅因为它们自然相连，还因为它们共享相似的治疗演化路径。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import networkx as nx


# ============================
# 1) 从样本构建异构图
# ============================
def sample_to_visits(sample: Dict) -> List[Dict]:
    """把单个样本转成 visit 级异构事件序列。

    Args:
        sample: pyhealth 样本，含 conditions/procedures/drugs_hist/lab

    Returns:
        visits: [{'D': [...], 'P': [...], 'M': [...], 'L': [...]}, ...]
                 L = lab/检验项目（从 lab_inj_merged_list 展平去重）
    """
    conditions = sample.get('conditions', [])
    procedures = sample.get('procedures', [])
    drugs_hist = sample.get('drugs_hist', [])
    lab = sample.get('lab_inj_merged_list', [])

    def _to_codes(x):
        if not isinstance(x, list):
            return []
        if len(x) > 0 and isinstance(x[0], list):
            return [[str(c) for c in v] for v in x]
        return [str(c) for c in x]

    conds = _to_codes(conditions)
    procs = _to_codes(procedures)
    drugs = _to_codes(drugs_hist)

    # lab 结构: (visits, monitors, items)，展平 monitors 去重
    lab_visits = []
    if isinstance(lab, list):
        for visit in lab:
            items = set()
            if isinstance(visit, list):
                for monitor in visit:
                    if isinstance(monitor, list):
                        for code in monitor:
                            s = str(code)
                            if s != 'nan' and s.strip():
                                items.add(s)
            lab_visits.append(sorted(items))

    n_visits = max(len(conds), len(procs), len(drugs), len(lab_visits))
    visits = []
    for t in range(n_visits):
        visits.append({
            'D': conds[t] if t < len(conds) else [],
            'P': procs[t] if t < len(procs) else [],
            'M': drugs[t] if t < len(drugs) else [],
            'L': lab_visits[t] if t < len(lab_visits) else [],
        })
    return visits


# ICU 通用支持性用药（ATC 第三级）——与具体疾病无关，需过滤
# 这些是所有 ICU 病人的常规护理药，会掩盖疾病特异性治疗信号
ICU_COMMON_MED = {
    'B05X',  # 静脉补液/血液代用品
    'B01A',  # 抗血栓
    'N02B',  # 镇痛
    'A02B',  # 抗酸/抗溃疡
    'A06A',  # 泻药/通便
    'B05A',  # 血液和相关制品
    'B05B',  # 静脉溶液
    'A03F',  # 促胃肠动力
    'A04A',  # 止吐
    'N05C',  # 催眠镇静
    'N05B',  # 抗焦虑
}


def build_hetero_edges(all_visits: List[List[Dict]]) -> Dict[Tuple[str, str, str], int]:
    """从所有多就诊患者构建异构边计数。

    边类型:
      - co_dm: 同次就诊 D↔M (诊断-用药共现)
      - co_dp: 同次就诊 D↔P (诊断-手术共现)
      - evo:   跨就诊 前次所有节点 → 后次诊断 (时序演化)

    Args:
        all_visits: 每个患者的 visit 序列 [{'D','P','M'}, ...]

    Returns:
        edges: {(from_node, to_node, edge_type): count}
               node 形如 'D_<code>', 'M_<code>', 'P_<code>'
    """
    edges = defaultdict(int)
    for patient in all_visits:
        n = len(patient)
        for t in range(n):
            D_t = patient[t]['D']
            P_t = patient[t]['P']
            # 过滤 ICU 通用支持性用药（保留疾病特异性用药）
            M_t = [m for m in patient[t]['M'] if m not in ICU_COMMON_MED]
            L_t = patient[t].get('L', [])

            # 同次共现边
            for d in D_t:
                for m in M_t:
                    edges[('D_' + d, 'M_' + m, 'co_dm')] += 1
                    edges[('M_' + m, 'D_' + d, 'co_dm')] += 1
                for p in P_t:
                    edges[('D_' + d, 'P_' + p, 'co_dp')] += 1
                    edges[('P_' + p, 'D_' + d, 'co_dp')] += 1
                for l in L_t:
                    edges[('D_' + d, 'L_' + l, 'co_dl')] += 1
                    edges[('L_' + l, 'D_' + d, 'co_dl')] += 1

            # 跨就诊演化边: 前次所有事件 → 后次诊断
            if t + 1 < n:
                D_next = patient[t + 1]['D']
                for dn in D_next:
                    for d in D_t:
                        edges[('D_' + d, 'D_' + dn, 'evo')] += 1
                    for m in M_t:
                        edges[('M_' + m, 'D_' + dn, 'evo')] += 1
                    for p in P_t:
                        edges[('P_' + p, 'D_' + dn, 'evo')] += 1
                    for l in L_t:
                        edges[('L_' + l, 'D_' + dn, 'evo')] += 1

    return dict(edges)


# ============================
# 2) 元路径共现矩阵
# ============================
def _path_counts(edges: Dict, path_type: str) -> Dict[Tuple[str, str], int]:
    """计算指定元路径下的 诊断→诊断 连接次数。

    元路径类型:
      - 'natural' : D --evo--> D'           (纯自然演化)
      - 'med'     : D --co_dm--> M --evo--> D'  (用药干预演化)
      - 'surg'    : D --co_dp--> P --evo--> D'  (手术干预演化)
      - 'med_med' : D --co_dm--> M --co_dm--> D 无意义; 使用:
        'med_evo_med': D --co_dm--> M --evo--> M'(?) 不常用，跳过

    Returns:
        {(D_i, D_j): count}  通过该元路径，D_i 通向 D_j 的次数
    """
    if path_type == 'natural':
        # D --evo--> D'
        counts = defaultdict(int)
        for (f, t, typ), c in edges.items():
            if typ == 'evo' and f.startswith('D_') and t.startswith('D_'):
                counts[(f, t)] += c
        return dict(counts)

    elif path_type in ('med', 'surg'):
        # D --co_dX--> X --evo--> D'
        # 用加权计数避免 list 展开导致的性能/内存膨胀
        if path_type == 'med':
            co_typ, co_src = 'co_dm', 'M_'
        else:
            co_typ, co_src = 'co_dp', 'P_'

        dx = defaultdict(list)  # D -> [(X, count)]
        xd = defaultdict(list)  # X -> [(D', count)]  (evo)
        for (f, t, typ), c in edges.items():
            if typ == co_typ:
                dx[f].append((t, c))
            elif typ == 'evo' and t.startswith('D_'):
                xd[f].append((t, c))

        counts = defaultdict(int)
        for d, xs in dx.items():
            for x, c_dx in xs:
                for dn, c_evo in xd.get(x, []):
                    counts[(d, dn)] += c_dx * c_evo
        return dict(counts)

    else:
        raise ValueError(f'Unknown path type: {path_type}')


def compute_metapath_matrices(
    edges: Dict,
    disease_nodes: List[str],
    path_weights: Dict[str, float] = None,
) -> Tuple[np.ndarray, Dict]:
    """融合多条元路径，构建 诊断-诊断 加权共现矩阵。

    Args:
        edges: build_hetero_edges 的输出
        disease_nodes: 所有诊断节点列表 ['D_96', 'D_108', ...]
        path_weights: 各元路径的权重，如 {'natural':1.0, 'med':1.0, 'surg':1.0}

    Returns:
        A: (num_diseases, num_diseases) 加权共现矩阵
        info: 各元路径的统计信息
    """
    if path_weights is None:
        path_weights = {'natural': 1.0, 'med': 1.0, 'surg': 1.0}

    idx = {n: i for i, n in enumerate(disease_nodes)}
    num_d = len(disease_nodes)
    A = np.zeros((num_d, num_d))
    info = {}

    for ptype, w in path_weights.items():
        counts = _path_counts(edges, ptype)
        total_w = sum(counts.values())
        info[ptype] = {'total_weight': total_w, 'num_pairs': len(counts)}

        # 构建该元路径的转移矩阵
        A_p = np.zeros((num_d, num_d))
        for (d_i, d_j), c in counts.items():
            if d_i in idx and d_j in idx:
                A_p[idx[d_i], idx[d_j]] += c

        # 每条路径单独行归一化（转移概率形式）
        # 关键：均衡各元路径贡献，避免 med(亿级) 淹没 natural(万级)
        row_sums = A_p.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        A_p = A_p / row_sums

        A += w * A_p

    # 融合矩阵再行归一化
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums

    return A_norm, info


# ============================
# 3) 在元路径矩阵上发现原型
# ============================
def _build_similarity_matrix(edges, disease_nodes, path_weights=None, top_k=10):
    """构建诊断相似度矩阵（元路径融合 + 对称化 + top-k 稀疏化）。

    top-k 稀疏化: 医学演化网络中，每个诊断只与少数强连接相关。
    若不稀疏化，med 元路径会把几乎所有诊断通过"用药→下一诊断"连起来，
    导致相似度矩阵饱和、谱聚类退化为一个巨型社区。
    """
    A, path_info = compute_metapath_matrices(edges, disease_nodes, path_weights)
    S = A + A.T
    np.fill_diagonal(S, 1.0)
    S = np.maximum(S, 0)

    # top-k 邻居稀疏化（保留每行最强连接）
    if top_k is not None and top_k > 0 and S.shape[0] > top_k:
        S_sparse = np.zeros_like(S)
        for i in range(S.shape[0]):
            top_idx = np.argsort(S[i])[::-1][:top_k]
            S_sparse[i, top_idx] = S[i, top_idx]
        # 对称化: 保留双向强连接（i→j 或 j→i 任一强则保留）
        S = np.maximum(S_sparse, S_sparse.T)

    return S, path_info


def select_k_metapath(
    edges: Dict,
    disease_nodes: List[str],
    min_k: int = 6,
    max_k: int = 20,
    path_weights: Dict[str, float] = None,
    random_state: int = 42,
) -> Tuple[int, Dict]:
    """用轮廓系数在 [min_k, max_k] 搜索最优原型数 K。

    原型数要"合理"：太大则每个专家数据稀疏，太小则无法区分演化模式。
    用轮廓系数衡量聚类紧密度，结合临床先验约束 K 范围。

    Args:
        edges, disease_nodes: 同 discover_prototypes_metapath
        min_k, max_k: K 搜索范围
        path_weights: 元路径权重

    Returns:
        best_k: 轮廓系数最优的 K
        info: K 选择详情
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    S, path_info = _build_similarity_matrix(edges, disease_nodes, path_weights)
    valid_mask = S.sum(axis=1) > 0
    S_valid = S[valid_mask][:, valid_mask]
    n_valid = S_valid.shape[0]

    effective_max = min(max_k, n_valid - 1)
    effective_min = max(min_k, 2)

    best_k = effective_min
    best_score = -1
    k_details = {}

    for k in range(effective_min, effective_max + 1):
        try:
            labels = SpectralClustering(
                n_clusters=k, affinity='precomputed',
                random_state=random_state, assign_labels='kmeans',
            ).fit_predict(S_valid)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(S_valid, labels)
            k_details[k] = round(float(score), 4)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    info = {
        'best_k': best_k,
        'best_silhouette': float(best_score),
        'k_scores': k_details,
        'method': f'silhouette over [{effective_min}, {effective_max}]',
        'metapath_stats': path_info,
    }
    return best_k, info


def discover_prototypes_metapath(
    edges: Dict,
    disease_nodes: List[str],
    num_prototypes: Optional[int] = None,
    min_k: int = 6,
    max_k: int = 20,
    path_weights: Dict[str, float] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """用元路径加权矩阵做谱聚类，发现疾病轨迹原型（路线C）。

    Args:
        edges: build_hetero_edges 的输出
        disease_nodes: 诊断节点列表
        num_prototypes: 目标原型数 K。None 时用轮廓系数自动选择（推荐）。
        min_k, max_k: 自动选 K 的搜索范围
        path_weights: 元路径权重

    Returns:
        diag_labels: (num_diseases,) 每个诊断的原型标签
        info: 原型详情 + 元路径统计 + K 选择
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import KMeans

    S, path_info = _build_similarity_matrix(edges, disease_nodes, path_weights)

    # 自动选 K（轮廓系数）
    if num_prototypes is None:
        num_prototypes, k_info = select_k_metapath(
            edges, disease_nodes, min_k, max_k, path_weights, random_state)
    else:
        k_info = {'best_k': num_prototypes, 'method': 'specified'}

    valid_mask = S.sum(axis=1) > 0
    n_valid = valid_mask.sum()
    effective_k = min(num_prototypes, max(2, n_valid))

    all_labels = np.full(len(disease_nodes), -1, dtype=int)
    if n_valid >= 2:
        try:
            clustering = SpectralClustering(
                n_clusters=effective_k,
                affinity='precomputed',
                random_state=random_state,
                assign_labels='kmeans',
            )
            labels_valid = clustering.fit_predict(S[valid_mask][:, valid_mask])
            all_labels[valid_mask] = labels_valid
        except Exception:
            kmeans = KMeans(n_clusters=effective_k, random_state=random_state, n_init=10)
            labels_valid = kmeans.fit_predict(S[valid_mask][:, valid_mask])
            all_labels[valid_mask] = labels_valid

    # 孤立节点归入原型0
    all_labels[~valid_mask] = 0

    # 原型描述：每个原型内的高频诊断
    prototypes = {}
    for k in range(effective_k):
        mask = all_labels == k
        nodes = [disease_nodes[i] for i in range(len(disease_nodes)) if mask[i]]
        prototypes[int(k)] = {
            'num_diseases': len(nodes),
            'diseases': nodes[:20],
        }

    info = {
        'num_prototypes': effective_k,
        'metapath_stats': path_info,
        'prototypes': prototypes,
        'chapter_labels': all_labels.tolist(),
        'k_selection': k_info,
    }
    return all_labels, info


# ============================
# 4) Metapath2Vec 图嵌入（更'图'的方案，处理枢纽结构）
# ============================
def build_adjacency(edges: Dict) -> Dict[str, List[Tuple[str, str]]]:
    """把边转成邻接表 {node: [(neighbor, edge_type)]}。"""
    adj = defaultdict(list)
    for (f, t, typ), c in edges.items():
        for _ in range(min(c, 20)):  # 截断权重，避免重复过多
            adj[f].append((t, typ))
    return dict(adj)


def metapath_walk(
    adj: Dict,
    start: str,
    schema: List[str],
    walk_length: int,
    rng: Optional[np.random.RandomState] = None,
) -> List[str]:
    """按元路径引导的随机游走。

    Args:
        adj: 邻接表
        start: 起始节点
        schema: 节点类型序列，如 ['D','M','D','P','D']（医学元路径）
        walk_length: 游走长度

    Returns:
        walk: 节点序列
    """
    if rng is None:
        rng = np.random.RandomState(42)
    walk = [start]
    cur = start
    for step in range(1, walk_length):
        target_type = schema[step % len(schema)]
        candidates = [n for n, typ in adj.get(cur, []) if n.startswith(target_type)]
        if not candidates:
            break
        cur = candidates[rng.randint(len(candidates))]
        walk.append(cur)
    return walk


def generate_metapath_walks(
    adj: Dict,
    disease_nodes: List[str],
    schema: List[str],
    num_walks: int = 10,
    walk_length: int = 8,
    random_state: int = 42,
) -> List[List[str]]:
    """从每个诊断节点出发，生成元路径引导的随机游走。"""
    rng = np.random.RandomState(random_state)
    walks = []
    starts = [n for n in disease_nodes if n in adj]
    for _ in range(num_walks):
        for start in starts:
            w = metapath_walk(adj, start, schema, walk_length, rng)
            if len(w) >= 3:
                walks.append(w)
    return walks


def learn_skipgram_embeddings(
    walks: List[List[str]],
    node_list: List[str],
    dim: int = 64,
    epochs: int = 3,
    context_size: int = 4,
    neg_samples: int = 5,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """用 Skip-gram + 负采样学习节点嵌入（自实现，不依赖 gensim）。

    Args:
        walks: 随机游走序列列表
        node_list: 所有节点
        dim: 嵌入维度
        epochs: 训练轮数

    Returns:
        embeddings: (len(node_list), dim) 节点嵌入
    """
    import random as pyrandom
    pyrandom.seed(42)
    torch.manual_seed(42)  # 关键: 固定 Embedding 初始化, 保证原型可复现

    node2idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    embed = nn.Embedding(N, dim).to(device)
    opt = torch.optim.Adam(embed.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for walk in walks:
            L = len(walk)
            for i, center in enumerate(walk):
                if center not in node2idx:
                    continue
                cidx = node2idx[center]
                # 上下文窗口
                lo, hi = max(0, i - context_size), min(L, i + context_size + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    ctx = walk[j]
                    if ctx not in node2idx:
                        continue
                    pidx = node2idx[ctx]
                    # 负采样
                    negs = [node2idx[n] for n in
                            pyrandom.sample(node_list, min(neg_samples, N))
                            if n != ctx]

                    center_v = embed(torch.tensor(cidx, device=device))  # (D,)
                    pos_v = embed(torch.tensor(pidx, device=device))
                    # 正样本：-log σ(center·pos)
                    loss_pos = -F.logsigmoid((center_v * pos_v).sum())
                    # 负样本：-log σ(-center·neg)
                    if negs:
                        neg_t = torch.tensor(negs, device=device)
                        neg_v = embed(neg_t)  # (K, D)
                        loss_neg = -F.logsigmoid(-(center_v.unsqueeze(0) * neg_v).sum(dim=1)).sum()
                    else:
                        loss_neg = 0.0

                    loss = loss_pos + loss_neg
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    n_samples += 1

        if n_samples > 0:
            total_loss = total_loss / n_samples

    with torch.no_grad():
        return embed.weight.detach().cpu()


def discover_prototypes_metapath2vec(
    edges: Dict,
    disease_nodes: List[str],
    path_schema: List[str] = None,
    num_prototypes: Optional[int] = None,
    embed_dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 8,
    epochs: int = 3,
    min_k: int = 6,
    max_k: int = 20,
    random_state: int = 42,
    device: torch.device = torch.device('cpu'),
) -> Tuple[np.ndarray, Dict]:
    """Metapath2Vec: 用元路径引导随机游走 + Skip-gram 嵌入，聚类诊断原型。

    相比"共现矩阵+谱聚类"，图嵌入能正确捕捉医学网络的枢纽结构，
    避免巨型社区问题。

    Args:
        edges: build_hetero_edges 的输出
        disease_nodes: 诊断节点列表
        path_schema: 医学元路径，默认 ['D','M','D','P','D']
        num_prototypes: 原型数，None 时自动选择

    Returns:
        diag_labels: (num_diseases,) 诊断原型标签
        info: 详情
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if path_schema is None:
        # 医学元路径（精化版）：突出手术(P)干预 + 加入检验(L)监测
        # P 出现更频繁 → 手术路径采样更多（手术比用药更有疾病特异性）
        # L 加入检验信号（如肌钙蛋白↑ 强化疾病特异性）
        path_schema = ['D', 'P', 'D', 'M', 'D', 'P', 'D', 'L', 'D']

    # 只保留与诊断节点相关的边（减小子图规模）
    disease_set = set(disease_nodes)
    sub_edges = {k: v for k, v in edges.items() if k[0] in disease_set or k[1] in disease_set}

    adj = build_adjacency(sub_edges)
    walks = generate_metapath_walks(
        adj, disease_nodes, path_schema, num_walks, walk_length, random_state)

    # 全部节点（诊断+中间节点）
    all_nodes = sorted(set(n for w in walks for n in w))
    emb = learn_skipgram_embeddings(
        walks, all_nodes, embed_dim,
        epochs=epochs, device=device)

    # 取诊断节点嵌入（缺失节点用零向量兜底）
    idx_map = {n: i for i, n in enumerate(all_nodes)}
    zero_vec = np.zeros(embed_dim)
    diag_emb = np.array([
        emb[idx_map[d]].numpy() if d in idx_map else zero_vec
        for d in disease_nodes
    ])

    # 自动选 K（轮廓系数）
    if num_prototypes is None:
        best_k, best_score = min_k, -1
        k_scores = {}
        for k in range(min_k, min(max_k, len(diag_emb)) + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labs = km.fit_predict(diag_emb)
            if len(set(labs)) < 2:
                continue
            sc = silhouette_score(diag_emb, labs)
            k_scores[k] = round(float(sc), 4)
            if sc > best_score:
                best_score, best_k = sc, k
        num_prototypes = best_k
        k_info = {'best_k': best_k, 'best_silhouette': float(best_score), 'k_scores': k_scores}
    else:
        k_info = {'best_k': num_prototypes, 'method': 'specified'}

    # 最终聚类
    kmeans = KMeans(n_clusters=num_prototypes, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(diag_emb)

    # 原型描述
    prototypes = {}
    for k in range(num_prototypes):
        mask = labels == k
        nodes = [disease_nodes[i] for i in range(len(disease_nodes)) if mask[i]]
        prototypes[int(k)] = {'num_diseases': len(nodes), 'diseases': nodes[:20]}

    info = {
        'num_prototypes': num_prototypes,
        'prototypes': prototypes,
        'k_selection': k_info,
        'embed_dim': embed_dim,
        'path_schema': path_schema,
        'method': 'metapath2vec',
    }
    return labels, info


def compute_patient_prototype_activation(
    patient_conds: List[List[str]],
    diag_labels: np.ndarray,
    disease_nodes: List[str],
    num_prototypes: int,
) -> np.ndarray:
    """计算患者的原型激活向量（用于路由，替代章节分布）。

    路线C 不用章节：直接统计患者历史诊断属于哪些治疗演化社区（原型），
    得到 (B, K) 的激活向量作为路由信号。

    Args:
        patient_conds: (B, visits, codes) 每个患者的诊断序列
        diag_labels: (num_diseases,) 每个诊断的原型标签
        disease_nodes: 诊断节点列表 ['D_96', ...]
        num_prototypes: 原型数 K

    Returns:
        activation: (B, num_prototypes) 每个患者的原型激活向量（归一化）
    """
    node_to_label = {}
    for node, label in zip(disease_nodes, diag_labels):
        if node.startswith('D_') and label >= 0:
            node_to_label[node[2:]] = label  # ccs code -> prototype label

    B = len(patient_conds)
    activation = np.zeros((B, num_prototypes))

    for b in range(B):
        for visit_codes in patient_conds[b]:
            if not isinstance(visit_codes, list):
                continue
            for code in visit_codes:
                label = node_to_label.get(str(code), None)
                if label is not None:
                    activation[b, label] += 1
        row_sum = activation[b].sum()
        if row_sum > 0:
            activation[b] /= row_sum

    return activation


def log_metapath_info(info: Dict):
    """打印元路径统计和原型信息。"""
    print(f"\n异构时序知识图谱原型 (K={info['num_prototypes']}):")
    stats = info.get('metapath_stats', {})
    for ptype, s in stats.items():
        print(f"  元路径 {ptype}: {s['total_weight']} 总权重, {s['num_pairs']} 对诊断")
    for k, p in info.get('prototypes', {}).items():
        print(f"  原型{k}: {p['num_diseases']} 个诊断")
