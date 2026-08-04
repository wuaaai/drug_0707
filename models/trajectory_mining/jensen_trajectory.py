"""
Jensen 式 CCS 级轨迹挖掘。

方法（参考 Jensen et al., 2014, Nature Communications）：
1. 从多就诊患者中收集 CCS 编码转移对 (A → B)
2. 统计检验筛选显著方向性转移对（RR + 二项检验 + Bonferroni 校正）
3. 贪心拼接成 3-4 步轨迹
4. Jaccard 相似度 + 层次聚类 → 轨迹原型

与章节级方法的本质区别：直接在 CCS 编码（~270 种）上操作，不归约到 19 个章节。
"""
"""

  第 1 步：找"病与病之间显著的转移"（main.py:77-81）

  从所有多就诊患者的历史里，统计"这次得了病 A，下次得了病 B"的组合，然后做统计检验，只保留真正显著的转移：

  # main.py:77-81
  transitions = collect_transitions(all_ccs_seqs, min_occurrence=20)  # 收集A→B转移
  significant = compute_rr_and_significance(transitions, total, alpha=0.001)  # 统计检验
  strong = [s for s in significant if s['rr'] > 2.0]  # 只留RR>2的

  跑遍所有病人，数"得了 A 的人，下次得 B 的概率是不是明显偏高"。
  - RR > 2：得 A 的人，得 B 的风险是没得 A 的人的 2 倍以上
  - Bonferroni 校正 p < 0.001：排除随机巧合
  - 转移对至少出现 20 次：排除稀有情况

  ---
  第 2 步：把转移"按疾病的来龙去脉"分组（main.py:83-91）

  每个显著转移都是一条"A → B"的边。现在把 A、B 都映射到它们所属的 ICD
  章节（比如心衰、高血压都属于循环系统），然后按"从哪个章节 → 到哪个章节"分组：

  # main.py:83-91
  chapter_groups = defaultdict(list)
  for s in strong:
      ch_pair = (map_ccs_to_expert(s['from']), map_ccs_to_expert(s['to']))
      chapter_groups[ch_pair].append(s)

  通俗理解：把所有"循环→呼吸"的转移归一堆、"代谢→循环"归另一堆……每个堆代表一种典型的疾病演化路径。要求每堆至少有 5
  条，且源和目标不是同一个章节。

  ---
  第 3 步：挑最大的几堆当作"疾病原型"（main.py:94-100）

  # main.py:94-100
  sorted_groups = sorted(valid_groups.items(), key=lambda x: -len(x[1]))
  proto_groups = [(ch, pairs) for ch, pairs in sorted_groups if len(pairs) >= 15][:8]
  num_prototypes = len(proto_groups)

  通俗理解：按堆的大小排序、最多 8 个最大的堆作为原型。每个原型 = 一条"从某类病 →
  到某类病"的统计验证演化路径。

  原型0: E9→E7    # 呼吸→循环
  原型1: E7→E9    # 循环→呼吸
  原型2: E7→E2    # 循环→代谢


"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from scipy.stats import binomtest
from itertools import combinations
import json


def collect_transitions(
    all_seqs: List[List[List[str]]],
    min_occurrence: int = 5
) -> Dict[Tuple[str, str], Dict]:
    """从多就诊患者的 CCS 序列中收集所有诊断转移对。

    Args:
        all_seqs: 每位患者的 visit 级 CCS 编码序列 [[['49','98'], ['108']], ...]
        min_occurrence: 转移对最小出现次数

    Returns:
        transitions: {(A, B): {'count': N_AB, 'A_total': N_A, 'B_total': N_B}}
    """
    # 统计转移对
    pair_counts = Counter()
    code_counts = Counter()

    for patient_seq in all_seqs:
        for t in range(len(patient_seq) - 1):
            current_codes = set(patient_seq[t])
            next_codes = set(patient_seq[t + 1])

            for code in current_codes:
                code_counts[code] += 1

            for a in current_codes:
                for b in next_codes:
                    if a != b:
                        pair_counts[(a, b)] += 1

    # 只保留足够频繁的转移对
    transitions = {}
    for (a, b), count in pair_counts.items():
        if count >= min_occurrence:
            transitions[(a, b)] = {
                'count': count,
                'A_total': code_counts[a],
                'B_total': code_counts[b],
            }

    return transitions


def compute_rr_and_significance(
    transitions: Dict,
    total_transitions: int,
    alpha: float = 0.05
) -> List[Dict]:
    """计算 Relative Risk 和统计显著性。

    RR = P(B|A) / P(B|not A)
    二项检验 H0: P(B|A) <= P(B)  (A 不增加 B 的风险)

    Args:
        transitions: collect_transitions 的输出
        total_transitions: 总转移次数
        alpha: 显著性水平（Bonferroni 校正前）

    Returns:
        significant_pairs: 通过检验的显著方向性转移对
    """
    n_pairs = len(transitions)
    corrected_alpha = alpha / n_pairs  # Bonferroni

    significant = []
    for (a, b), stats in transitions.items():
        n_ab = stats['count']
        n_a = stats['A_total']
        n_b = stats['B_total']

        # P(B|A) = N_AB / N_A
        p_b_given_a = n_ab / n_a if n_a > 0 else 0
        # P(B|not A) = (N_B - N_AB) / (total - N_A)
        n_not_a = total_transitions - n_a
        n_b_not_a = n_b - n_ab
        p_b_given_not_a = n_b_not_a / n_not_a if n_not_a > 0 else 0

        # RR = P(B|A) / P(B|not A)
        rr = p_b_given_a / p_b_given_not_a if p_b_given_not_a > 0 else float('inf')

        # 二项检验: 在 N_A 次试验中观察到 >= N_AB 次 B 的概率
        # H0: P(B) = N_B / total
        p_b = n_b / total_transitions if total_transitions > 0 else 0
        try:
            test = binomtest(n_ab, n_a, p_b, alternative='greater')
            p_value = test.pvalue
        except Exception:
            p_value = 1.0

        # 方向性检验: is A→B significantly more likely than B→A?
        reverse_count = 0
        if (b, a) in transitions:
            reverse_count = transitions[(b, a)]['count']

        # A→B 显著多于 B→A？
        directional = n_ab > reverse_count

        if p_value < corrected_alpha and rr > 1.0 and directional:
            significant.append({
                'from': a,
                'to': b,
                'count': n_ab,
                'rr': round(rr, 2),
                'p_value': p_value,
                'directional': directional,
                'reverse_count': reverse_count,
            })

    # 按 RR 降序排列
    significant.sort(key=lambda x: -x['rr'])
    return significant


def stitch_trajectories(
    pairs: List[Dict],
    max_length: int = 4,
    min_patients: int = 3
) -> List[Dict]:
    """贪心拼接显著转移对为更长轨迹。

    Args:
        pairs: 显著方向性转移对列表
        max_length: 最大轨迹长度（步数）
        min_patients: 轨迹最少患者数

    Returns:
        trajectories: [{'codes': [A,B,C], 'steps': [...], 'total_count': N}, ...]
    """
    # 构建转移图
    edges = defaultdict(list)
    for p in pairs:
        edges[p['from']].append(p)

    trajectories = []
    visited_pairs = set()

    for p in pairs:
        if (p['from'], p['to']) in visited_pairs:
            continue

        traj_codes = [p['from'], p['to']]
        traj_steps = [p]
        visited_pairs.add((p['from'], p['to']))
        total_count = p['count']

        # 向后延伸
        current = p['to']
        for _ in range(max_length - 2):
            candidates = [e for e in edges.get(current, [])
                         if (e['from'], e['to']) not in visited_pairs]
            if not candidates:
                break
            best = max(candidates, key=lambda e: e['rr'])
            traj_codes.append(best['to'])
            traj_steps.append(best)
            visited_pairs.add((best['from'], best['to']))
            total_count = min(total_count, best['count'])
            current = best['to']

        if len(traj_codes) >= 2 and total_count >= min_patients:
            trajectories.append({
                'codes': traj_codes,
                'steps': [{'from': s['from'], 'to': s['to'], 'rr': s['rr']} for s in traj_steps],
                'total_count': total_count,
                'length': len(traj_codes),
            })

    trajectories.sort(key=lambda t: -t['total_count'])
    return trajectories


def cluster_trajectories(
    trajectories: List[Dict],
    min_clusters: int = 5,
    max_clusters: int = 15,
    max_code_freq: float = 0.5
) -> Tuple[np.ndarray, int, Dict]:
    """用 Jaccard 相似度 + 层次聚类将轨迹分组为原型。

    Args:
        trajectories: stitch_trajectories 的输出
        min_clusters, max_clusters: 聚类数范围

    Returns:
        labels: 每条轨迹的簇标签
        best_k: 最优簇数
        info: 聚类详情
    """
    from sklearn.metrics import silhouette_score

    n = len(trajectories)
    if n < min_clusters * 2:
        # 轨迹太少，不分簇
        labels = np.zeros(n, dtype=int)
        return labels, 1, {'error': 'Too few trajectories'}

    # 过滤高频通用编码（出现在 >max_code_freq 轨迹中的编码）
    code_doc_freq = Counter()
    for t in trajectories:
        for code in set(t['codes']):
            code_doc_freq[code] += 1
    n_trajs = len(trajectories)
    filtered_codes = {code for code, freq in code_doc_freq.items()
                      if freq / n_trajs > max_code_freq}
    if filtered_codes:
        print(f"  过滤 {len(filtered_codes)} 个高频通用编码: {sorted(filtered_codes)[:10]}...")

    # 构建 TF-IDF 加权 Jaccard 相似度
    code_sets = []
    for t in trajectories:
        fs = set(t['codes']) - filtered_codes
        if not fs:
            fs = set(t['codes'])
        code_sets.append(fs)

    idf = {}
    for code in set().union(*code_sets):
        df = sum(1 for s in code_sets if code in s)
        idf[code] = np.log((n + 1) / (df + 1)) + 1

    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            common = code_sets[i] & code_sets[j]
            all_codes = code_sets[i] | code_sets[j]
            if not all_codes:
                continue
            w_common = sum(idf.get(c, 1) for c in common)
            w_all = sum(idf.get(c, 1) for c in all_codes)
            sim = w_common / w_all if w_all > 0 else 0
            similarity[i, j] = sim
            similarity[j, i] = sim

    # 层次聚类
    from sklearn.cluster import AgglomerativeClustering
    best_k = min_clusters
    best_score = -1
    best_labels = None
    distance = 1 - similarity

    for k in range(min_clusters, min(max_clusters + 1, n)):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k, metric='precomputed', linkage='average'
            )
            labels = clustering.fit_predict(distance)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(distance, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels.copy()
        except Exception:
            continue

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    # 构建原型描述
    prototypes = {}
    for label in range(best_k):
        mask = best_labels == label
        cluster_trajs = [trajectories[i] for i in range(n) if mask[i]]
        # 收集该原型中的高频 CCS 编码
        code_freq = Counter()
        for t in cluster_trajs:
            for code in t['codes']:
                code_freq[code] += 1

        prototypes[int(label)] = {
            'num_trajectories': len(cluster_trajs),
            'top_codes': code_freq.most_common(10),
            'representative_trajectories': [
                t['codes'] for t in sorted(cluster_trajs, key=lambda x: -x['total_count'])[:3]
            ],
            'avg_length': np.mean([t['length'] for t in cluster_trajs]),
        }

    info = {
        'best_k': best_k,
        'best_score': float(best_score),
        'prototypes': prototypes,
        'total_trajectories': n,
    }
    return best_labels, best_k, info


def map_trajectory_to_experts(
    trajectories: List[Dict],
    labels: np.ndarray,
    num_experts: int = 19,
    num_prototypes: int = None
) -> np.ndarray:
    """将轨迹原型映射到章节级专家。

    每个轨迹原型 → 其在各 ICD 章节上的激活权重。
    用于初始化 TrajectoryCare 的专家原型向量。

    Returns:
        expert_weights: (num_prototypes, num_experts) 每个原型对各章节专家的权重
    """
    from models.expert_selectv2 import map_ccs_to_expert

    if num_prototypes is None:
        num_prototypes = len(set(labels)) - (1 if -1 in labels else 0)

    expert_weights = np.zeros((num_prototypes, num_experts))

    for proto_id in range(num_prototypes):
        mask = labels == proto_id
        proto_trajs = [trajectories[i] for i in range(len(trajectories)) if mask[i]]

        # 统计轨迹中每个 CCS 编码映射到的章节
        chapter_counts = np.zeros(num_experts)
        for t in proto_trajs:
            for code in t['codes']:
                eid = map_ccs_to_expert(str(code))
                if 0 <= eid < num_experts:
                    chapter_counts[eid] += 1

        if chapter_counts.sum() > 0:
            expert_weights[proto_id] = chapter_counts / chapter_counts.sum()

    return expert_weights


def discover_trajectories_jensen(
    all_seqs: List[List[List[str]]],
    min_occurrence: int = 5,
    alpha: float = 0.05,
    max_traj_length: int = 4,
    min_clusters: int = 5,
    max_clusters: int = 15
) -> Dict:
    """Jensen 式轨迹发现的完整流水线。

    Returns:
        result: 包含 transitions, trajectories, clusters, expert_weights
    """
    # Step 1: 收集转移对
    print(f"Step 1: 收集转移对 (min_occurrence={min_occurrence})...")
    transitions = collect_transitions(all_seqs, min_occurrence)
    total_transitions = sum(t['count'] for t in transitions.values())
    print(f"  收集到 {len(transitions)} 对转移, 总转移 {total_transitions} 次")

    # Step 2: 显著性检验
    print(f"Step 2: 显著性检验 (alpha={alpha}, Bonferroni)...")
    significant = compute_rr_and_significance(transitions, total_transitions, alpha)
    print(f"  显著方向性转移: {len(significant)} 对")
    if significant:
        print(f"  Top-5 转移: {[(s['from'], s['to'], s['rr']) for s in significant[:5]]}")

    # Step 3: 拼接轨迹
    print(f"Step 3: 贪心拼接轨迹 (max_length={max_traj_length})...")
    trajectories = stitch_trajectories(significant, max_length=max_traj_length)
    print(f"  生成 {len(trajectories)} 条轨迹")
    if trajectories:
        for t in trajectories[:5]:
            print(f"    {'→'.join(t['codes'])} (n={t['total_count']})")

    # Step 4: 聚类轨迹原型
    print(f"Step 4: 聚类轨迹为原型 (K∈[{min_clusters},{max_clusters}])...")
    labels, best_k, cluster_info = cluster_trajectories(
        trajectories, min_clusters, max_clusters
    )
    print(f"  发现 {best_k} 个轨迹原型")
    for proto_id, info in cluster_info.get('prototypes', {}).items():
        print(f"    原型 {proto_id}: {info['num_trajectories']} 条轨迹, "
              f"平均长度 {info['avg_length']:.1f}, "
              f"Top codes: {info['top_codes'][:5]}")

    return {
        'transitions': transitions,
        'significant_pairs': significant,
        'trajectories': trajectories,
        'cluster_labels': labels.tolist() if len(labels) > 0 else [],
        'best_k': best_k,
        'cluster_info': cluster_info,
    }


def compress_rules_to_prototypes(
    significant_pairs: List[Dict],
    num_chapters: int,
    num_prototypes: int,
    weight_by: str = 'rr',
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """将 Jensen 统计显著的 CCS 转移规则压缩为 K 个路由原型。

    Phase 2 核心函数：将数百条 CCS→CCS 显著转移规则用 KMeans 聚类压缩，
    产生的原型向量用于初始化 TrajectoryAwareRouter 的可学习原型。

    每一条显著转移规则 A→B 被编码为一个 2*C 维向量：
        [0..C-1] = from_chapter 的 one-hot（乘以权重）
        [C..2C-1] = to_chapter 的 one-hot（乘以权重）
    其中 C = num_chapters。

    KMeans 聚类后，每个簇中心代表一种典型的章节→章节转移模式。
    再将其扩展为 3*C 维（加入频率成分），与路由器输入空间对齐。

    Args:
        significant_pairs: Jensen 检验产生的显著转移对列表
            [{'from': CCS_A, 'to': CCS_B, 'rr': float, 'count': int}, ...]
        num_chapters: 章节数量 (ICD-9: 19, ICD-10: 22)
        num_prototypes: 目标原型数 K
        weight_by: 规则向量的权重方式 ('rr'=相对风险, 'count'=频次, 'none'=均等)
        random_state: 随机种子

    Returns:
        rule_prototypes: (K, 3*C) ndarray，扩展后的规则原型（初始化为路由器原型）
        chapter_labels: 每个章节隶属的原型标签
        info: 压缩详情（各原型映射到的章节对、簇大小等）
    """
    from sklearn.cluster import KMeans
    from models.expert_selectv2 import map_ccs_to_expert

    if len(significant_pairs) < num_prototypes:
        print(f"  警告: 显著转移对({len(significant_pairs)}) < 原型数({num_prototypes})，将自动调整 K")
        num_prototypes = max(2, len(significant_pairs))

    # Step 1: 将每条规则编码为 2*C 维向量
    D = num_chapters * 2
    n_rules = len(significant_pairs)
    rule_vectors = np.zeros((n_rules, D))
    rule_meta = [None] * n_rules  # 预分配，保持与 n_rules 长度一致

    for i, pair in enumerate(significant_pairs):
        from_ch = map_ccs_to_expert(pair['from'])
        to_ch = map_ccs_to_expert(pair['to'])

        if not (0 <= from_ch < num_chapters and 0 <= to_ch < num_chapters):
            continue
        if from_ch == to_ch:
            continue  # 跳过同章节转移（噪音较大）

        if weight_by == 'rr':
            w = min(pair.get('rr', 1.0), 100.0)  # 截断极端 RR
        elif weight_by == 'count':
            w = max(pair.get('count', 1), 1)
            w = min(np.log(w + 1), 5.0)  # log 缩放
        else:
            w = 1.0

        rule_vectors[i, from_ch] = w
        rule_vectors[i, num_chapters + to_ch] = w
        rule_meta[i] = {
            'from': pair['from'],
            'to': pair['to'],
            'from_ch': from_ch,
            'to_ch': to_ch,
            'weight': w,
            'rr': pair.get('rr', 0),
            'count': pair.get('count', 0),
        }

    # 过滤全零向量
    nonzero_mask = rule_vectors.sum(axis=1) > 0
    rule_vectors = rule_vectors[nonzero_mask]
    rule_meta = [rule_meta[i] for i in range(n_rules) if nonzero_mask[i] and rule_meta[i] is not None]
    n_valid = len(rule_vectors)

    if n_valid < num_prototypes:
        num_prototypes = max(2, n_valid)
        print(f"  警告: 有效规则数({n_valid})不足，调整 K={num_prototypes}")

    # Step 2: KMeans 聚类
    kmeans = KMeans(n_clusters=num_prototypes, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(rule_vectors)
    centroids_2c = kmeans.cluster_centers_  # (K, 2*C)

    # Step 3: 扩展为 3*C 维（添加频率成分）
    # 对于每个簇，统计该簇规则的 from_ch / to_ch 分布作为 freq 分量
    centroids_3c = np.zeros((num_prototypes, num_chapters * 3))
    centroids_3c[:, :num_chapters] = centroids_2c[:, :num_chapters]  # in_edges ≈ from
    centroids_3c[:, num_chapters:2*num_chapters] = centroids_2c[:, num_chapters:]  # out_edges ≈ to
    # freq = avg of from + to
    centroids_3c[:, 2*num_chapters:] = (
        centroids_2c[:, :num_chapters] + centroids_2c[:, num_chapters:]
    ) / 2

    # 每行归一化到 unit norm（与路由器使用的 F.normalize 一致）
    row_norms = np.linalg.norm(centroids_3c, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    centroids_3c = centroids_3c / row_norms

    # Step 4: 构建原型描述信息
    chapter_labels = np.full(num_chapters, -1, dtype=int)
    prototypes_info = {}
    for k in range(num_prototypes):
        # 找到该簇中最具代表性的章节对
        mask = cluster_labels == k
        cluster_rules = [rule_meta[i] for i in range(n_valid) if mask[i]]
        # 统计源/目标章节频率
        from_counter = Counter(r['from_ch'] for r in cluster_rules)
        to_counter = Counter(r['to_ch'] for r in cluster_rules)
        top_from = from_counter.most_common(3)
        top_to = to_counter.most_common(3)

        prototypes_info[int(k)] = {
            'num_rules': len(cluster_rules),
            'top_from_chapters': [(int(ch), cnt) for ch, cnt in top_from],
            'top_to_chapters': [(int(ch), cnt) for ch, cnt in top_to],
            'example_rules': [
                {'from': r['from'], 'to': r['to'], 'rr': r['rr']}
                for r in cluster_rules[:3]
            ],
        }

        # 章节→原型映射：该簇中最常见的 from_ch 和 to_ch 分配到这个原型
        for ch, _ in top_from + top_to:
            if 0 <= ch < num_chapters:
                if chapter_labels[ch] == -1:
                    chapter_labels[ch] = k

    # 未被分配的章节 -> 最近的原型
    for ch in range(num_chapters):
        if chapter_labels[ch] == -1:
            chapter_labels[ch] = 0  # 默认归入原型 0

    info = {
        'num_rules': n_valid,
        'num_prototypes': num_prototypes,
        'prototypes': prototypes_info,
        'chapter_labels': chapter_labels.tolist(),
        'method': f'KMeans(weight_by={weight_by})',
    }
    return centroids_3c, chapter_labels, info


def log_rule_prototypes(info: Dict):
    """打印规则原型的详细信息。"""
    print(f"\n规则原型压缩结果 (K={info['num_prototypes']}, {info['method']}):")
    print(f"  总规则数: {info['num_rules']}")
    for k, p in info.get('prototypes', {}).items():
        from_str = ', '.join([f'E{ch}({cnt})' for ch, cnt in p['top_from_chapters']])
        to_str = ', '.join([f'E{ch}({cnt})' for ch, cnt in p['top_to_chapters']])
        print(f"  原型{k} ({p['num_rules']}条规则):")
        print(f"    源章节: {from_str}")
        print(f"    目标章节: {to_str}")
        for ex in p['example_rules'][:1]:
            print(f"    示例: {ex['from']}→{ex['to']} (RR={ex['rr']})")
