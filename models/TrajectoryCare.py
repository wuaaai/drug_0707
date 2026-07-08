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
    """软路由器：计算患者与各轨迹原型的匹配分数。"""

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
        tau = self.tau_init * torch.exp(
            -self.alpha * torch.clamp(num_visits - 1, min=0)
        ) + self.tau_min
        tau = tau.view(-1, 1)  # (B, 1)

        # 温度缩放 softmax
        route_weights = F.softmax(logits / tau, dim=-1)  # (B, K)

        return route_weights


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

    Args:
        Tokenizers_visit_event: visit 事件的 tokenizer 字典
        Tokenizers_monitor_event: monitor 事件的 tokenizer 字典
        output_size: 标签数（药物数或诊断数）
        device: 设备
        chapter_labels: (C,) 每个章节的原型标签
        num_prototypes: 轨迹原型数 K
        embedding_dim: 嵌入维度
        dropout: dropout 率
        tau_init: 冷启动初始温度
        cold_start_alpha: 退火速率
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
    ):
        super().__init__()
        self.device = device
        self.num_prototypes = num_prototypes
        self.num_chapters = num_chapters
        self.output_size = output_size

        # 路由器
        self.router = TrajectoryRouter(
            num_prototypes=num_prototypes,
            num_chapters=num_chapters,
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
    ) -> torch.Tensor:
        """前向传播。

        Args:
            batch_data: 同 Base2_1 的输入
            patient_chapter_dist: (B, num_chapters) 患者的章节分布，可选
            num_visits: (B,) 每位患者的就诊数，可选

        Returns:
            logits: (B, output_size)
        """
        B = len(batch_data['visit_id'])
        device = self.device

        # 路由：如果没有章节信息，使用均匀权重
        if patient_chapter_dist is not None and num_visits is not None:
            route_weights = self.router(patient_chapter_dist, num_visits)
        else:
            route_weights = torch.ones(B, self.num_prototypes, device=device)
            route_weights = route_weights / self.num_prototypes

        # 各专家独立计算
        expert_outputs = []
        for k in range(self.num_prototypes):
            out = self.experts[k](batch_data)  # (B, output_size)
            expert_outputs.append(out)

        # 加权组合
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, K, output_size)
        route_weights = route_weights.unsqueeze(-1)  # (B, K, 1)
        logits = (expert_stack * route_weights).sum(dim=1)  # (B, output_size)

        return logits

    def get_route_weights(
        self,
        patient_chapter_dist: torch.Tensor,
        num_visits: torch.Tensor,
    ) -> torch.Tensor:
        """导出路由权重（用于可解释性分析）。"""
        with torch.no_grad():
            return self.router(patient_chapter_dist, num_visits)
