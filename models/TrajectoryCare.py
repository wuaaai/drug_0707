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


class TrajectoryAwareRouter(nn.Module):
    """轨迹感知路由器：使用患者的轨迹拓扑特征（入边/出边/频率）进行路由。

    相比 TrajectoryRouter（使用静态章节词袋），此路由器的优势：
    - 能区分演化路径不同但词袋相同的患者
    - 可学习原型向量直观对应"轨迹模式"
    - 余弦相似度路由具有天然可解释性

    路由权重 = softmax( cos(患者轨迹特征, 可学习原型) + 线性投影 + 章节先验 , 温度)
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
    ):
        """轨迹感知路由器。

        Args:
            rule_prototypes: (K, 3*C) ndarray，来自 compress_rules_to_prototypes() 的规则原型。
                若提供，用于初始化 prototype_vectors（而非随机初始化）。
                Phase 2: 将 Jensen 规则先验引入路由。
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.hidden_dim = hidden_dim
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.alpha = alpha
        self.use_cosine_routing = use_cosine_routing

        # 轨迹特征维度 = num_chapters * 3（入边 + 出边 + 频率分布）
        self.traj_feat_dim = num_chapters * 3

        # === 轨迹原型向量（可学习，直观对应于 K 种轨迹模式） ===
        # Phase 2: 若提供了 rule_prototypes，用它初始化（而非随机）
        if rule_prototypes is not None:
            assert rule_prototypes.shape == (num_prototypes, self.traj_feat_dim), \
                f'rule_prototypes shape mismatch: {rule_prototypes.shape} vs {(num_prototypes, self.traj_feat_dim)}'
            init_tensor = torch.tensor(rule_prototypes, dtype=torch.float32)
        else:
            init_tensor = torch.randn(num_prototypes, self.traj_feat_dim) * 0.1

        self.prototype_vectors = nn.Parameter(init_tensor)

        # === 章节先验（同原版路由器） ===
        self.prototype_prior = nn.Parameter(
            torch.randn(num_prototypes, num_chapters) * 0.1
        )

        # === 线性投影路由器（作为余弦路由的补充） ===
        self.route_proj = nn.Sequential(
            nn.Linear(self.traj_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_prototypes),
        )

    def forward(
        self,
        traj_features: torch.Tensor,
        patient_chapter_dist: torch.Tensor,
        num_visits: torch.Tensor,
    ) -> torch.Tensor:
        """计算路由权重。

        Args:
            traj_features: (B, num_chapters * 3) 轨迹拓扑特征
                           [in_edges(0..C-1), out_edges(C..2C-1), freq(2C..3C-1)]
            patient_chapter_dist: (B, num_chapters) 章节分布（补充信号）
            num_visits: (B,) 每位患者就诊数（控制退火温度）

        Returns:
            route_weights: (B, num_prototypes) softmax 归一化路由权重
        """
        B = traj_features.size(0)
        device = traj_features.device

        logits = 0

        # === 信号 1：余弦相似度路由（可解释性强） ===
        if self.use_cosine_routing:
            norm_features = F.normalize(traj_features, dim=-1)        # (B, D)
            norm_protos = F.normalize(self.prototype_vectors, dim=-1) # (K, D)
            cosine_logits = torch.mm(norm_features, norm_protos.t())  # (B, K)
            # 缩放到 [-sqrt(D), sqrt(D)] 防止 softmax 过度锐化
            cosine_logits = cosine_logits * (self.traj_feat_dim ** 0.5)
            logits = logits + cosine_logits

        # === 信号 2：线性投影路由（表达能力） ===
        proj_logits = self.route_proj(traj_features)  # (B, K)
        logits = logits + proj_logits

        # === 信号 3：章节先验相似度 ===
        prior_sim = F.cosine_similarity(
            patient_chapter_dist.unsqueeze(1),
            F.softmax(self.prototype_prior, dim=-1).unsqueeze(0),
            dim=-1
        )  # (B, K)
        logits = logits + prior_sim

        # === 冷启动退火 ===
        tau = self._compute_temperature(num_visits)
        tau = tau.view(-1, 1)  # (B, 1)

        route_weights = F.softmax(logits / tau, dim=-1)  # (B, K)
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
    """单个轨迹专家 —— 轻量版 GRU 编码器 + 输出头。

    相比 Base2_1 的完整版，这个轻量版：
    - 只处理 visit_event keys（conditions, procedures, drugs_hist）
    - 使用较小的 hidden dim（默认 96）
    - 无 monitor_event 处理（检验信号通过时序分析器注入）
    """

    def __init__(
        self,
        Tokenizers_visit_event: Dict,
        output_size: int,
        device: torch.device,
        embedding_dim: int = 96,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.feature_keys = list(Tokenizers_visit_event.keys())
        self.device = device

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for key in self.feature_keys:
            tokenizer = self.visit_event_token[key]
            self.embeddings[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        # GRU layers
        self.gru_layers = nn.ModuleDict()
        for key in self.feature_keys:
            self.gru_layers[key] = nn.GRU(
                embedding_dim, embedding_dim, batch_first=True
            )

        self.dropout = nn.Dropout(p=dropout)

        # 输出头
        item_num = len(self.feature_keys)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(item_num * embedding_dim, output_size),
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

            _, hidden = self.gru_layers[key](x)
            # hidden: (1, B, embedding_dim)
            patient_emb_list.append(hidden.squeeze(0))  # (B, embedding_dim)

        patient_emb = torch.cat(patient_emb_list, dim=-1)
        logits = self.fc(patient_emb)
        return logits


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
    ):
        super().__init__()
        self.device = device
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.output_size = output_size
        self.use_traj_router = use_traj_router

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

        # K 个轨迹专家
        self.experts = nn.ModuleList([
            TrajectoryExpert(
                Tokenizers_visit_event=Tokenizers_visit_event,
                output_size=output_size,
                device=device,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
            for _ in range(num_prototypes)
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
                                            num_visits)
            else:
                route_weights = self.router(patient_chapter_dist, num_visits)
        else:
            # 无路由信息时使用均匀权重
            route_weights = torch.ones(B, self.num_prototypes, device=device)
            route_weights = route_weights / self.num_prototypes

        # === 各专家独立计算 ===
        expert_outputs = []
        for k in range(self.num_prototypes):
            out = self.experts[k](batch_data)  # (B, output_size)
            expert_outputs.append(out)

        # === 加权组合 ===
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, output_size)
        route_weights = route_weights.unsqueeze(-1)  # (B, K, 1)
        logits = (expert_stack * route_weights).sum(dim=1)  # (B, output_size)

        return logits

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
