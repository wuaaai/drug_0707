"""
TrajectoryCare —— 轨迹驱动的混合专家（MoE）预测模型。

核心架构：
1. K 个轨迹专家（GRU-based），每个对应一个轨迹原型
2. 软路由器：基于患者章节块序列与各原型的匹配分数进行软路由
3. 冷启动退火：低就诊数患者使用均匀先验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TrajectoryRouter(nn.Module):
    """软路由器：基于患者章节分布计算与各轨迹原型的匹配分数。

    这是原始路由器，输入为静态的章节词袋分布。
    参见 TrajectoryAwareRouter 获取使用轨迹拓扑特征的新版本。
    """

    def __init__(
        self,
        num_prototypes: int,
        num_chapters: int,
        hidden_dim: int = 128,
        tau_init: float = 2.0,
        tau_min: float = 0.5,
        alpha: float = 0.3,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.hidden_dim = hidden_dim
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.alpha = alpha

        # 每个原型的章节先验分布（可学习）
        self.prototype_prior = nn.Parameter(
            torch.randn(num_prototypes, num_chapters) * 0.1
        )

        # 路由投影：章节分布 → 匹配分数
        self.route_proj = nn.Sequential(
            nn.Linear(num_chapters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prototypes),
        )

    def forward(
        self,
        patient_chapter_dist: torch.Tensor,
        num_visits: torch.Tensor,
    ) -> torch.Tensor:
        """计算路由权重。

        Args:
            patient_chapter_dist: (B, num_chapters) 患者的章节累计分布
            num_visits: (B,) 每位患者的就诊数

        Returns:
            route_weights: (B, num_prototypes) softmax归一化的路由权重
        """
        B = patient_chapter_dist.size(0)
        device = patient_chapter_dist.device

        # 基础匹配分数
        base_scores = self.route_proj(patient_chapter_dist)  # (B, K)

        # 与原型先验的相似度
        prior_sim = F.cosine_similarity(
            patient_chapter_dist.unsqueeze(1),  # (B, 1, C)
            F.softmax(self.prototype_prior, dim=-1).unsqueeze(0),  # (1, K, C)
            dim=-1
        )  # (B, K)

        logits = base_scores + prior_sim

        # 冷启动退火温度
        tau = self._compute_temperature(num_visits)
        tau = tau.view(-1, 1)  # (B, 1)

        # 温度缩放 softmax
        route_weights = F.softmax(logits / tau, dim=-1)  # (B, K)

        return route_weights

    def _compute_temperature(self, num_visits: torch.Tensor) -> torch.Tensor:
        """计算冷启动退火温度。"""
        tau = self.tau_init * torch.exp(
            -self.alpha * torch.clamp(num_visits - 1, min=0)
        ) + self.tau_min
        return tau


class TemporalTrajectoryEncoder(nn.Module):
    """时序轨迹编码器：用轻量 GRU 建模就诊序列的时序依赖。

    原始的 TrajectoryAwareRouter 使用静态的入边/出边/频率统计(3C特征)，
    丢失了就诊之间的时序顺序。此编码器通过 GRU 捕获动态转移模式。

    输入: (B, visits, C) 每就诊的章节出现向量
    输出: (B, 2*H) = [last_hidden || mean_pool] 时序轨迹表征
    """

    def __init__(self, num_chapters: int, hidden_dim: int = 64):
        super().__init__()
        self.visit_encoder = nn.Sequential(
            nn.Linear(num_chapters, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, visit_seqs: torch.Tensor) -> torch.Tensor:
        x = self.visit_encoder(visit_seqs)
        output, hidden = self.gru(x)
        last = hidden.squeeze(0)
        mean_pool = output.mean(dim=1)
        return torch.cat([last, mean_pool], dim=-1)


class TrajectoryAwareRouter(nn.Module):
    """轨迹感知路由器：使用患者的轨迹特征进行路由。

    路由信号 = 静态拓扑特征(3C) + 可选时序编码(2*H) + 章节先验
    """

    def __init__(
        self,
        num_prototypes: int,
        num_chapters: int,
        hidden_dim: int = 128,
        tau_init: float = 2.0,
        tau_min: float = 0.5,
        alpha: float = 0.3,
        use_cosine_routing: bool = True,
        rule_prototypes: Optional[np.ndarray] = None,
        use_temporal_encoder: bool = False,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.hidden_dim = hidden_dim
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.alpha = alpha
        self.use_cosine_routing = use_cosine_routing
        self.use_temporal_encoder = use_temporal_encoder

        self.traj_feat_dim = num_chapters * 3
        if use_temporal_encoder:
            self.temporal_enc = TemporalTrajectoryEncoder(num_chapters, hidden_dim // 2)
            router_input_dim = self.traj_feat_dim + hidden_dim  # 3C + 2*(H/2)
        else:
            self.temporal_enc = None
            router_input_dim = self.traj_feat_dim

        # 轨迹原型向量（维度与路由输入对齐）
        init_tensor = torch.randn(num_prototypes, router_input_dim) * 0.1
        if rule_prototypes is not None and not use_temporal_encoder:
            # 规则原型仅用于静态特征维度的初始化
            assert rule_prototypes.shape == (num_prototypes, self.traj_feat_dim)
            init_tensor[:, :self.traj_feat_dim] = torch.tensor(rule_prototypes, dtype=torch.float32)
        self.prototype_vectors = nn.Parameter(init_tensor)

        self.prototype_prior = nn.Parameter(
            torch.randn(num_prototypes, num_chapters) * 0.1)

        # 路由投影（输入维度随时序编码器扩展）
        self.route_proj = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_prototypes),
        )

    def forward(
        self,
        traj_features: torch.Tensor,
        patient_chapter_dist: torch.Tensor,
        num_visits: torch.Tensor,
        visit_chapter_seqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算路由权重。

        三通道路由信号融合：
        1. 静态拓扑特征（in/out/freq 三维）
        2. 时序编码特征（GRU on visit sequences）[新增]
        3. 章节先验相似度

        Args:
            traj_features: (B, C*3) 静态轨迹拓扑特征
            patient_chapter_dist: (B, C) 章节分布
            num_visits: (B,) 就诊数
            visit_chapter_seqs: (B, max_visits, C) 就诊级章节序列 [新增]

        Returns:
            route_weights: (B, K) 归一化路由权重
        """
        B = traj_features.size(0)
        device = traj_features.device

        # === 融合轨迹特征（静态 + 时序） ===
        if self.use_temporal_encoder and visit_chapter_seqs is not None:
            temporal_embed = self.temporal_enc(visit_chapter_seqs)  # (B, H)
            router_input = torch.cat([traj_features, temporal_embed], dim=-1)
        else:
            router_input = traj_features

        logits = 0

        # === 信号 1：余弦相似度路由（可解释性强） ===
        if self.use_cosine_routing:
            norm_features = F.normalize(router_input, dim=-1)
            norm_protos = F.normalize(self.prototype_vectors, dim=-1)
            cosine_logits = torch.mm(norm_features, norm_protos.t())
            cosine_logits = cosine_logits * (router_input.size(-1) ** 0.5)
            logits = logits + cosine_logits

        # === 信号 2：线性投影路由 ===
        proj_logits = self.route_proj(router_input)
        logits = logits + proj_logits

        # === 信号 3：章节先验相似度 ===
        prior_sim = F.cosine_similarity(
            patient_chapter_dist.unsqueeze(1),
            F.softmax(self.prototype_prior, dim=-1).unsqueeze(0), dim=-1)
        logits = logits + prior_sim

        # === 冷启动退火 ===
        tau = self._compute_temperature(num_visits).view(-1, 1)
        route_weights = F.softmax(logits / tau, dim=-1)
        return route_weights

    def _compute_temperature(self, num_visits: torch.Tensor) -> torch.Tensor:
        """计算冷启动退火温度。"""
        tau = self.tau_init * torch.exp(
            -self.alpha * torch.clamp(num_visits - 1, min=0)
        ) + self.tau_min
        return tau

    def get_prototype_similarity(
        self, traj_features: torch.Tensor
    ) -> torch.Tensor:
        """导出患者与各原型的余弦相似度（用于可解释性分析）。"""
        with torch.no_grad():
            norm_features = F.normalize(traj_features, dim=-1)
            norm_protos = F.normalize(self.prototype_vectors, dim=-1)
            sim = torch.mm(norm_features, norm_protos.t())
        return sim


class TrajectoryExpert(nn.Module):
    """单个轨迹专家 —— 支持多种架构变体的 GRU 编码器 + 输出头。

    Phase 3: 不同专家使用不同架构，适应不同的疾病演化模式。

    Expert Types:
    - 'standard': 1 层 GRU, 96 dim (同 Phase 1) —— 通用模式
    - 'deep': 3 层 GRU + 残差连接 —— 渐进退化型（HTN→HF→CKD）
    - 'attentive': 1 层 GRU + 自注意力池化 —— 急性触发型（MI→Shock）
    - 'wide': 1 层 GRU, 192 dim —— 共病扩散型（DM→CKD→CVD→PAD）

    相比 Base2_1 的完整版，所有变体：
    - 只处理 visit_event keys（conditions, procedures, drugs_hist）
    - 无 monitor_event 处理（检验信号通过时序分析器注入）
    """

    EXPERT_TYPE_CONFIG = {
        'standard': {'gru_layers': 1, 'hidden_dim': None, 'use_attention': False},
        'deep':     {'gru_layers': 3, 'hidden_dim': None, 'use_attention': False},
        'attentive':{'gru_layers': 1, 'hidden_dim': None, 'use_attention': True},
        'wide':     {'gru_layers': 1, 'hidden_dim': None, 'use_attention': False},
    }

    def __init__(
        self,
        Tokenizers_visit_event: Dict,
        output_size: int,
        device: torch.device,
        embedding_dim: int = 96,
        dropout: float = 0.5,
        expert_type: str = 'standard',
        shared_embeddings: Optional[nn.ModuleDict] = None,
        lab_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert expert_type in self.EXPERT_TYPE_CONFIG, \
            f"Unknown expert_type: {expert_type}, choose from {list(self.EXPERT_TYPE_CONFIG.keys())}"

        self.expert_type = expert_type
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.feature_keys = list(Tokenizers_visit_event.keys())
        self.device = device
        self.lab_encoder = lab_encoder  # 可选检验特征编码器

        # 根据专家类型确定架构参数
        config = self.EXPERT_TYPE_CONFIG[expert_type]
        gru_layers = config['gru_layers']
        expert_hidden = config['hidden_dim'] or embedding_dim
        self.use_attention = config['use_attention']

        # Embedding layers（若传入 shared_embeddings 则共享，否则各自独立）
        self.embeddings = shared_embeddings if shared_embeddings is not None else nn.ModuleDict()
        if shared_embeddings is None:
            for key in self.feature_keys:
                tokenizer = self.visit_event_token[key]
                self.embeddings[key] = nn.Embedding(
                    tokenizer.get_vocabulary_size(),
                    embedding_dim,
                    padding_idx=tokenizer.get_padding_index(),
                )

        # LayerNorm: 稳定 visit 级 embedding 分布
        self.visit_ln = nn.LayerNorm(embedding_dim)

        # GRU layers（按 expert type 差异化）
        self.gru_layers = nn.ModuleDict()
        for key in self.feature_keys:
            if expert_type == 'deep' and gru_layers > 1:
                # 深层 GRU：每一层输出维度不同，用 ModuleList
                gru = nn.GRU(embedding_dim, expert_hidden,
                             num_layers=gru_layers, dropout=dropout if gru_layers > 1 else 0,
                             batch_first=True)
            else:
                gru = nn.GRU(embedding_dim, expert_hidden,
                             num_layers=gru_layers, batch_first=True)
            self.gru_layers[key] = gru

        # 'attentive' 类型：自注意力池化层
        if self.use_attention:
            self.self_attn = nn.MultiheadAttention(
                expert_hidden, num_heads=4, batch_first=True, dropout=dropout
            )

        self.dropout = nn.Dropout(p=dropout)

        # 输出头
        item_num = len(self.feature_keys)
        fc_input_dim = item_num * expert_hidden
        # 'wide' 类型的输出头需要降维
        if expert_type == 'wide' and expert_hidden > embedding_dim:
            self.fc_reduce = nn.Linear(expert_hidden, embedding_dim)
            fc_input_dim = item_num * embedding_dim
        else:
            self.fc_reduce = None

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_input_dim, fc_input_dim // 2),  # 隐藏层：384→192
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_input_dim // 2, output_size),
        )

    def forward(self, batch_data: Dict) -> torch.Tensor:
        """前向传播。

        Args:
            batch_data: 同 Base2_1 的输入格式，包含 conditions, procedures, drugs_hist

        Returns:
            logits: (B, output_size)
        """
        patient_emb_list = []

        for key in self.feature_keys:
            x = self.visit_event_token[key].batch_encode_3d(batch_data[key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (B, visits, events)
            x = self.dropout(self.embeddings[key](x))
            # (B, visits, events, embedding_dim)
            x = torch.sum(x, dim=2)
            # (B, visits, embedding_dim)
            x = self.visit_ln(x)  # LayerNorm 稳定分布

            # === GRU 编码（按 expert type 不同） ===
            if self.expert_type == 'deep':
                # 深层 GRU + 残差连接
                output, hidden = self.gru_layers[key](x)
                # 残差：将第一层输出加到最后一层
                if self.gru_layers[key].num_layers > 1:
                    hidden = hidden[-1:, :, :]  # 取最后一层 hidden
                patient_emb = hidden.squeeze(0)
            else:
                output, hidden = self.gru_layers[key](x)
                patient_emb = hidden.squeeze(0)  # (B, embedding_dim)

            # 'attentive'：对 output 序列做自注意力池化
            if self.use_attention:
                # output: (B, visits, hidden_dim)
                attn_out, _ = self.self_attn(output, output, output)
                # 对 visit 维度做平均池化
                patient_emb = attn_out.mean(dim=1)

            # 'wide'：通过降维层
            if self.fc_reduce is not None:
                patient_emb = self.fc_reduce(patient_emb)

            patient_emb_list.append(patient_emb)

        patient_emb = torch.cat(patient_emb_list, dim=-1)
        logits = self.fc(patient_emb)
        return logits, patient_emb


def assign_expert_types(
    rule_proto_info: Optional[Dict] = None,
    num_prototypes: int = 8,
) -> List[str]:
    """根据 Phase 2 规则原型特征为每个专家分配架构类型。

    分配策略（基于原型对应的转移规则特征）：
    - 高 RR（急性）→ 'attentive'
    - 多目标章节（共病扩散）→ 'wide'
    - 高自转移率（渐进退化）→ 'deep'
    - 其他 → 'standard'

    Args:
        rule_proto_info: compress_rules_to_prototypes() 返回的 info
        num_prototypes: 专家/原型数

    Returns:
        expert_types: list of str, 长度为 num_prototypes
    """
    if rule_proto_info is None or 'prototypes' not in rule_proto_info:
        # 无规则原型信息时，轮询分配四种类型以保持多样性
        cycle = ['deep', 'attentive', 'wide', 'standard']
        return [cycle[i % len(cycle)] for i in range(num_prototypes)]

    prototypes_info = rule_proto_info.get('prototypes', {})
    expert_types = []

    for k in range(num_prototypes):
        p = prototypes_info.get(k, {})
        num_rules = p.get('num_rules', 0)
        top_from = p.get('top_from_chapters', [])
        top_to = p.get('top_to_chapters', [])

        # 计算源/目标章节的分布特征
        from_chs = [ch for ch, _ in top_from]
        to_chs = [ch for ch, _ in top_to]
        unique_targets = len(set(to_chs))
        unique_sources = len(set(from_chs))
        num_from = sum(cnt for _, cnt in top_from) if top_from else 1

        # 源章节集中度：最高频源章节占比
        top_from_share = top_from[0][1] / num_from if top_from else 0

        # 分配逻辑基于拓扑多样性（而非 RR 绝对值，因为 MIMIC 中 RR 普遍很高）
        # 'wide': 多目标章节 → 共病扩散型（DM→CKD→CVD→PAD）
        if unique_targets >= 4:
            etype = 'wide'
        # 'deep': 源高度集中 + 规则丰富 → 渐进退化型（HTN→HF→CKD）
        elif top_from_share > 0.7 and num_rules > 100:
            etype = 'deep'
        # 'attentive': 源分散（多原因引起同一结果）→ 急性触发型（MI→Shock）
        elif top_from_share < 0.4 and unique_sources >= 3:
            etype = 'attentive'
        else:
            etype = 'standard'

        expert_types.append(etype)

    return expert_types


class LabFeatureEncoder(nn.Module):
    """实验室检验 + 输液特征编码器。

    临床动机：ICU 药物推荐高度依赖检验结果（e.g. 钾离子高→避免补钾，
    白细胞异常→抗生素）。原始专家只用 conditions/procedures/drugs_hist，
    忽略了 lab_inj_merged_list（检验+输液项目）。

    输入: 预处理后的 (B, visits, max_items) 扁平项目序列
    输出: (B, embedding_dim) 检验特征向量
    """

    def __init__(
        self,
        tokenizer,
        embedding_dim: int = 96,
        dropout: float = 0.3,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.embedding = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )
        self.visit_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, flat_seqs: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """编码检验特征。

        Args:
            flat_seqs: (B, visits, max_items) 字符串数组
            mask: (B, visits) 有效就诊掩码

        Returns:
            lab_embed: (B, embedding_dim)
        """
        B, max_visits, max_items = flat_seqs.shape
        pad_idx = self.tokenizer.get_padding_index()

        # 预构建 token->idx 安全映射（未知 token 用 padding）
        # 避免 test 集出现 OOV 编码时抛 ValueError 崩溃
        if not hasattr(self, '_safe_vocab'):
            self._safe_vocab = {}
        known = self._safe_vocab

        # 收集所有需要查找的 token（去重，减少 lookup 次数）
        all_tokens = set(flat_seqs.flatten().tolist()) - {''}
        for tok in all_tokens:
            if tok not in known:
                try:
                    known[tok] = self.tokenizer.vocabulary(tok)
                except (ValueError, KeyError):
                    known[tok] = pad_idx

        # 向量化映射
        encoded = np.zeros((B, max_visits, max_items), dtype=np.int64)
        for b in range(B):
            for t in range(max_visits):
                for j in range(max_items):
                    tok = flat_seqs[b, t, j]
                    encoded[b, t, j] = known.get(tok, pad_idx) if tok != '' else pad_idx

        x = torch.tensor(encoded, dtype=torch.long, device=self.device)
        # (B, visits, items)
        x = self.dropout(self.embedding(x))
        # (B, visits, items, D) -> sum over items -> (B, visits, D)
        x = x.sum(dim=2)
        x = self.ln(x)

        # GRU 时序建模
        output, hidden = self.visit_gru(x)
        # 用 mask 对 output 做加权平均
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        mask_t = mask_t.unsqueeze(-1)  # (B, visits, 1)
        weighted = (output * mask_t).sum(dim=1)
        denom = mask_t.sum(dim=1).clamp(min=1)
        pooled = weighted / denom  # (B, D)

        # 无有效就诊时用 0 向量
        return pooled


class DrugCooccurrenceModule(nn.Module):
    """药物共现传播模块：利用药物组合的结构化先验。

    药物推荐本质是组合预测（平均每次开 34 种药）。独立 sigmoid 预测
    忽略了药物间的共现结构。此模块在 logits 上做一次共现传播：
    预测到药物 A → 提升与 A 常共同开出的药物 B。

    logits_new = logits + alpha * (W_cooccur @ sigmoid(logits))

    W_cooccur 可学习（初始化为单位阵，不改变初始行为），
    通过训练数据自动学习药物组合模式。
    """

    def __init__(self, num_drugs: int, alpha: float = 0.3,
                 init_identity: bool = True):
        super().__init__()
        self.num_drugs = num_drugs
        self.alpha = alpha

        if init_identity:
            # 初始化为单位阵：初始不影响预测，随训练学习共现模式
            self.cooccur = nn.Parameter(torch.zeros(num_drugs, num_drugs))
            nn.init.eye_(self.cooccur)
            # 零对角（避免自增强）
            with torch.no_grad():
                self.cooccur.fill_diagonal_(0)
        else:
            self.cooccur = nn.Parameter(
                torch.zeros(num_drugs, num_drugs) * 0.01)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """共现传播。

        Args:
            logits: (B, num_drugs) 原始预测 logits

        Returns:
            logits + alpha * (cooccur @ sigmoid(logits)): (B, num_drugs)
        """
        prob = torch.sigmoid(logits)
        # (B, D) @ (D, D) -> (B, D)
        boost = torch.mm(prob, self.cooccur)
        return logits + self.alpha * boost


class TrajectoryCare(nn.Module):
    """TrajectoryCare 完整模型。

    支持两种路由器模式：
    - use_traj_router=False: 使用原始 TrajectoryRouter（静态章节词袋作为输入）
    - use_traj_router=True: 使用 TrajectoryAwareRouter（轨迹拓扑特征作为输入）

    Phase 2 增强：
    - 当提供 rule_prototypes 时，用它初始化 TrajectoryAwareRouter 的原型向量
      （来自 Jensen 显著转移规则的 KMeans 压缩）

    Args:
        Tokenizers_visit_event: visit 事件的 tokenizer 字典
        Tokenizers_monitor_event: monitor 事件的 tokenizer 字典
        output_size: 标签数（药物数或诊断数）
        device: 设备
        chapter_labels: (C,) 每个章节的原型标签
        num_prototypes: 轨迹原型数 K
        num_chapters: 章节数（ICD-9: 19, ICD-10: 22）
        embedding_dim: 嵌入维度
        dropout: dropout 率
        tau_init: 冷启动初始温度
        cold_start_alpha: 退火速率
        use_traj_router: 是否使用轨迹感知路由器
        rule_prototypes: (K, 3*C) ndarray，Jensen 规则压缩的原型（Phase 2）
    """

    def __init__(
        self,
        Tokenizers_visit_event: Dict,
        Tokenizers_monitor_event: Dict,
        output_size: int,
        device: torch.device,
        chapter_labels: np.ndarray,
        num_prototypes: int,
        num_chapters: int = 19,
        embedding_dim: int = 96,
        dropout: float = 0.5,
        tau_init: float = 2.0,
        cold_start_alpha: float = 0.3,
        use_traj_router: bool = False,
        rule_prototypes: Optional[np.ndarray] = None,
        rule_proto_info: Optional[Dict] = None,
        expert_types: Optional[List[str]] = None,
        use_drug_cooccurrence: bool = False,
        cooccur_alpha: float = 0.3,
        use_lab_encoder: bool = False,
        lab_tokenizer=None,
        lab_embedding_dim: int = 96,
    ):
        """初始化 TrajectoryCare 模型。

        Args:
            rule_proto_info: Phase 2 规则压缩的详情 dict（用于分配专家类型）。
            expert_types: (K,) list，每个专家的架构类型。
                Phase 3: 若提供，创建异构专家；否则所有专家使用 'standard' 类型。
            use_lab_encoder: 是否启用检验特征编码器。
            lab_tokenizer: lab_inj_merged_list 的 tokenizer。
        """
        super().__init__()
        self.device = device
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.output_size = output_size
        self.use_traj_router = use_traj_router

        # === 药物共现传播模块 ===
        self.use_drug_cooccurrence = use_drug_cooccurrence
        if use_drug_cooccurrence:
            self.drug_cooccur = DrugCooccurrenceModule(
                output_size, alpha=cooccur_alpha)
        else:
            self.drug_cooccur = None

        # === 检验特征编码器（lab → 药物预测直通通道） ===
        self.use_lab_encoder = use_lab_encoder
        if use_lab_encoder and lab_tokenizer is not None:
            self.lab_encoder = LabFeatureEncoder(
                lab_tokenizer, embedding_dim=lab_embedding_dim,
                dropout=dropout, device=device)
            # lab 嵌入 → 药物 logits
            self.lab_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(lab_embedding_dim, output_size),
            )
        else:
            self.lab_encoder = None
            self.lab_proj = None

        # === 路由器 ===
        if use_traj_router:
            self.router = TrajectoryAwareRouter(
                num_prototypes=num_prototypes,
                num_chapters=num_chapters,
                hidden_dim=embedding_dim,
                tau_init=tau_init,
                tau_min=0.5,
                alpha=cold_start_alpha,
                rule_prototypes=rule_prototypes,
            )
        else:
            self.router = TrajectoryRouter(
                num_prototypes=num_prototypes,
                num_chapters=num_chapters,
                hidden_dim=embedding_dim,
                tau_init=tau_init,
                alpha=cold_start_alpha,
            )

        # === Phase 3: K 个异构轨迹专家（共享 Embedding 层） ===
        # 创建一组共享的 Embedding，所有专家共用（大幅减少参数量，提高训练效率）
        shared_emb = nn.ModuleDict()
        for key in Tokenizers_visit_event.keys():
            tokenizer = Tokenizers_visit_event[key]
            shared_emb[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        if expert_types is None:
            auto_types = assign_expert_types(rule_proto_info, num_prototypes)
        else:
            auto_types = expert_types
        self.expert_types = auto_types

        # 打印专家类型分配
        type_counts = {}
        for t in auto_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        type_str = ', '.join([f'{k}={v}' for k, v in type_counts.items()])
        print(f"  专家类型分配 ({num_prototypes}个): {type_str}  |  共享Embedding=是")

        self.experts = nn.ModuleList([
            TrajectoryExpert(
                Tokenizers_visit_event=Tokenizers_visit_event,
                output_size=output_size,
                device=device,
                embedding_dim=embedding_dim,
                dropout=dropout,
                expert_type=auto_types[k],
                shared_embeddings=shared_emb,  # 共享 embedding
            )
            for k in range(num_prototypes)
        ])

        # 原型嵌入（用于路由可视化和可解释性）
        self.register_buffer(
            'chapter_labels',
            torch.tensor(chapter_labels, dtype=torch.long)
        )

    def forward(
        self,
        batch_data: Dict,
        patient_chapter_dist: Optional[torch.Tensor] = None,
        num_visits: Optional[torch.Tensor] = None,
        traj_features: Optional[torch.Tensor] = None,
        visit_chapter_seqs: Optional[torch.Tensor] = None,
        contrastive_weight: float = 0.0,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            batch_data: 同 Base2_1 的输入
            patient_chapter_dist: (B, num_chapters) 患者的章节分布，可选
            num_visits: (B,) 每位患者的就诊数，可选
            traj_features: (B, num_chapters * 3) 轨迹拓扑特征（仅 use_traj_router 时需要）

        Returns:
            logits: (B, output_size)
        """
        B = len(batch_data['visit_id'])
        device = self.device

        # === 路由 ===
        has_routing_info = (patient_chapter_dist is not None
                            and num_visits is not None)
        if has_routing_info:
            if self.use_traj_router and traj_features is not None:
                route_weights = self.router(traj_features,
                                            patient_chapter_dist,
                                            num_visits,
                                            visit_chapter_seqs)
            else:
                route_weights = self.router(patient_chapter_dist, num_visits)
        else:
            # 无路由信息时使用均匀权重
            route_weights = torch.ones(B, self.num_prototypes, device=device)
            route_weights = route_weights / self.num_prototypes

        # === 各专家独立计算（同时获取患者嵌入用于对比学习） ===
        expert_outputs = []
        patient_embs = []
        for k in range(self.num_prototypes):
            out, p_emb = self.experts[k](batch_data)  # (B, output_size), (B, D)
            expert_outputs.append(out)
            patient_embs.append(p_emb)

        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, output_size)
        route_w = route_weights.unsqueeze(-1)  # (B, K, 1)
        logits = (expert_stack * route_w).sum(dim=1)  # (B, output_size)

        # === 药物共现传播（利用组合结构先验） ===
        if self.use_drug_cooccurrence and self.drug_cooccur is not None:
            logits = self.drug_cooccur(logits)

        # === 检验特征通道（lab → 药物预测） ===
        if self.use_lab_encoder and self.lab_encoder is not None:
            raw_lab = batch_data.get('lab_inj_merged_list')
            if raw_lab is not None:
                from preprocess.trajectory_data_builder import preprocess_lab_batch
                flat_seqs, lab_mask = preprocess_lab_batch(raw_lab)
                lab_embed = self.lab_encoder(flat_seqs, lab_mask)  # (B, D)
                lab_logits = self.lab_proj(lab_embed)  # (B, output_size)
                logits = logits + lab_logits

        # === 轨迹对比损失（Trajectory-Contrastive MoE） ===
        # 创新: 路由权重相似的患者，其表征也应相似
        # 对比损失 = MSE(patient_emb_similarity, routing_similarity)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if contrastive_weight > 0:
            # 用路由权重加权融合各专家的患者嵌入
            # route_weights: (B, K), patient_embs: list of (B, D)
            p_emb_stack = torch.stack(patient_embs, dim=1)  # (B, K, D)
            fused_emb = (p_emb_stack * route_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

            # 归一化
            emb_norm = F.normalize(fused_emb, dim=-1)
            route_norm = F.normalize(route_weights, dim=-1)

            # 成对相似度矩阵
            emb_sim = torch.mm(emb_norm, emb_norm.t())  # (B, B)
            route_sim = torch.mm(route_norm, route_norm.t())  # (B, B)

            # MSE: 让嵌入相似度逼近路由相似度
            contrastive_loss = F.mse_loss(emb_sim, route_sim)

        # === 专家解耦损失（Expert Disentangle Loss） ===
        # 协方差解耦: 强制不同专家的输出去相关
        # 如果两个专家总是预测相似的药物组合，它们的信息是冗余的
        K = self.num_prototypes
        disentangle_loss = torch.tensor(0.0, device=self.device)
        if contrastive_weight > 0 and K > 1:
            # expert_stack: (B, K, output_size)
            # 转置为 (K, B*output_size) 便于计算专家间协方差
            expert_flat = expert_stack.permute(1, 0, 2).reshape(K, -1)
            expert_centered = expert_flat - expert_flat.mean(dim=1, keepdim=True)
            covariance = (expert_centered @ expert_centered.T) / (expert_centered.size(1) - 1)
            diag = torch.diag_embed(torch.diagonal(covariance))
            disentangle_loss = torch.norm(covariance - diag, p='fro') / K

        return logits, contrastive_loss, disentangle_loss

    def get_route_weights(
        self,
        patient_chapter_dist: torch.Tensor,
        num_visits: torch.Tensor,
        traj_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """导出路由权重（用于可解释性分析）。"""
        with torch.no_grad():
            if self.use_traj_router and traj_features is not None:
                return self.router(traj_features,
                                   patient_chapter_dist,
                                   num_visits)
            else:
                return self.router(patient_chapter_dist, num_visits)
