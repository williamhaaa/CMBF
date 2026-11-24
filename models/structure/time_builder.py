# -*- coding: utf-8 -*-
"""
models/sthyper/structure/time_builder.py  v0.2.4  (2025-12-13)

【版本号与修改记录（changelog）】
- v0.2.4 [Unified-ME][StrictEmptyGraph]
  1) 显式使用 get_mask(seq_len, t_window_size) 计算多尺度节点数 M_i，
     与主模型 ST_Hyper_Model 中 conv_layers / all_size 语义完全对齐；
  2) 在 __init__ 中预先构造每尺度节点数 node_sizes，并据此初始化节点嵌入；
  3) forward 中严格检查每一尺度：若 M_i<=0 或 TopK+阈值后所有超边为空，
     立即抛 RuntimeError，不再返回“空图”；
  4) 输出统一为 {'M','E','mask','prob_val'}，其中 mask 为 [2,P] 且满足
     0 <= node_idx < M, 0 <= edge_idx < E。

- v0.2.3 [Align-LegacyAHL][Strict-EmptyCheck]
  1) 多尺度节点数 M_i 的计算方式与旧版 multi_adaptive_hypergraoh 语义对齐；
  2) 保留逐尺度合法性检查与“空图报错”逻辑（本版在此基础上进一步收紧）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.utils import _as_int, _as_float, get_mask  # 与主模型保持一致


class multi_adaptive_hypergraoh_prob(nn.Module):
    """
    AHL：自适应超图学习
      H = Softmax(ReLU(α * node_emb @ hyper_emb^T)) → TopK(η) → 阈值二值化(β)

    设计要点：
      - 多尺度节点数 M_i 由 get_mask(seq_len, t_window_size) 决定；
      - 每一尺度 i 输出结构：
          {
            'M': M_i,            # 节点数（必须 >0）
            'E': E_i,            # 有效超边数（必须 >0）
            'mask': Long[2,P],   # (node_idx, edge_idx)，0<=node<M_i, 0<=edge<E_i
            'prob_val': None,    # 预留接口，目前不返回概率矩阵
          }
      - 若某尺度在 TopK+阈值 后所有超边都被剪空，则直接 RuntimeError，
        由上游（主模型）决定是否采用兜底 ring 构图。
    """
    def __init__(self, configs):
        super(multi_adaptive_hypergraoh_prob, self).__init__()

        # ====== 1. 基础配置 ======
        self.seq_len = _as_int(getattr(configs, 'seq_len', 96), 96)

        # t_window_size 归一化为 list[int]
        raw_window_size = getattr(configs, 't_window_size', None)
        if raw_window_size is None:
            raw_window_size = getattr(configs, 'window_size', None)
        if raw_window_size is None:
            raise ValueError("[AHL] t_window_size/window_size 未配置。")

        if isinstance(raw_window_size, int):
            self.window_size = [int(raw_window_size)]
        else:
            self.window_size = [int(w) for w in raw_window_size]
        if len(self.window_size) == 0:
            raise ValueError("[AHL] t_window_size/window_size 为空。")

        # 特征维：优先使用 t_d_model，其次回退到 d_model
        dim = getattr(configs, 't_d_model', None)
        if dim is None:
            dim = getattr(configs, 'd_model', None)
        if dim is None:
            raise ValueError("[AHL] configs 中缺少 t_d_model / d_model 配置。")
        self.dim = _as_int(dim, 128)

        # 超边数 hyper_num：与主模型 hyper_num1 对应
        raw_hyper_num = getattr(configs, 't_hyper_num', None)
        if raw_hyper_num is None:
            raw_hyper_num = getattr(configs, 'hyper_num', None)
        if raw_hyper_num is None:
            raise ValueError("[AHL] t_hyper_num/hyper_num 未配置。")

        if isinstance(raw_hyper_num, int):
            self.hyper_num = [int(raw_hyper_num)]
        else:
            self.hyper_num = [int(e) for e in raw_hyper_num]
        if len(self.hyper_num) == 0:
            raise ValueError("[AHL] t_hyper_num/hyper_num 为空。")

        if any(e <= 0 for e in self.hyper_num):
            raise ValueError(
                f"[AHL] 所有尺度的超边数必须为正，当前为 {self.hyper_num}"
            )

        # 多尺度长度约束：len(window_size) ≥ max(1, n_scales-1)
        n_scales = len(self.hyper_num)
        min_ws_len = max(1, n_scales - 1)
        if len(self.window_size) < min_ws_len:
            raise ValueError(
                f"[AHL] window_size 长度={len(self.window_size)} 太短，"
                f"至少需要 {min_ws_len} 以支持 n_scales={n_scales}；"
                f"请检查 t_window_size 与 t_hyper_num。"
            )

        # ====== 2. 多尺度节点数：与主模型 get_mask 完全对齐 ======
        node_sizes = get_mask(self.seq_len, self.window_size)
        if isinstance(node_sizes, (list, tuple)):
            node_sizes = [int(m) for m in node_sizes]
        else:
            node_sizes = [int(node_sizes)]

        if len(node_sizes) < n_scales:
            raise ValueError(
                f"[AHL] get_mask 返回的尺度数={len(node_sizes)} 小于超边尺度数={n_scales}；"
                f"请检查 seq_len={self.seq_len}, window_size={self.window_size}。"
            )

        self.node_sizes = node_sizes[:n_scales]  # 每个尺度的 M_i

        for i, M_i in enumerate(self.node_sizes):
            if M_i <= 0:
                raise ValueError(
                    f"[AHL] 第 {i} 个尺度的节点数 M_i={M_i} 非法，请检查 get_mask 配置。"
                )

        # ====== 3. 超图学习超参数 ======
        self.alpha = float(getattr(configs, 't_alpha', 2.0))       # ReLU 缩放
        self.eta = int(getattr(configs, 't_kprime', 10))             # TopK
        self.beta = float(getattr(configs, 't_beta_thr', 0.3))          # 二值化阈值

        # ====== 4. 嵌入层 ======
        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        self.linhy = nn.ModuleList()
        self.linnod = nn.ModuleList()

        for i in range(n_scales):
            E_conf = int(self.hyper_num[i])
            M_i = int(self.node_sizes[i])

            self.embedhy.append(nn.Embedding(E_conf, self.dim))
            self.linhy.append(nn.Linear(self.dim, self.dim))

            self.embednod.append(nn.Embedding(M_i, self.dim))
            self.linnod.append(nn.Linear(self.dim, self.dim))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]（目前仅用于获取 device，构图本身与 x 无直接关系）

        返回：
            List[dict]，长度 = len(self.hyper_num)，每个元素为：
              {
                'M': M_i,
                'E': E_i,
                'mask': LongTensor[2, P],
                'prob_val': None,
              }
        """
        device = x.device
        hyperedge_all = []
        n_scales = len(self.hyper_num)

        for i in range(n_scales):
            M_i = int(self.node_sizes[i])
            E_conf = int(self.hyper_num[i])

            if M_i <= 0:
                raise RuntimeError(
                    f"[AHL] forward 时发现第 {i} 个尺度节点数 M_i={M_i}<=0，"
                    f"请检查 get_mask 与配置。"
                )
            if E_conf <= 0:
                raise RuntimeError(
                    f"[AHL] forward 时发现第 {i} 个尺度超边数 E_conf={E_conf}<=0，"
                    f"请检查 t_hyper_num/hyper_num 配置。"
                )

            # ---- 1) 节点 / 超边索引 ----
            nodeidx = torch.arange(M_i, device=device, dtype=torch.long)   # [M_i]
            hypidxc = torch.arange(E_conf, device=device, dtype=torch.long)  # [E_conf]

            # ---- 2) 嵌入 + 线性变换 ----
            nodeec = self.embednod[i](nodeidx)  # [M_i, D]
            nodeec = self.linnod[i](nodeec)     # [M_i, D]

            hyperen = self.embedhy[i](hypidxc)  # [E_conf, D]
            hyperen = self.linhy[i](hyperen)    # [E_conf, D]

            # ---- 3) 相似度 → 概率矩阵 ----
            a = torch.mm(nodeec, hyperen.t())   # [M_i, E_conf]
            adj_prob = F.softmax(
                F.relu(self.alpha * a),
                dim=-1
            )                                   # [M_i, E_conf]

            # ---- 4) TopK by η ----
            if self.eta > 0:
                k = min(adj_prob.size(1), self.eta)
                if k <= 0:
                    raise RuntimeError(
                        f"[AHL] 第 {i} 个尺度 TopK k={k} 非法：eta={self.eta}, "
                        f"E_conf={E_conf}。"
                    )
                topv, topi = torch.topk(adj_prob, k=k, dim=1)
                mask = torch.zeros_like(adj_prob).scatter(1, topi, 1.0)
                adj = adj_prob * mask
            else:
                adj = adj_prob

            # ---- 5) 二值化 + 去掉全零超边 ----
            adj_bin = (adj > self.beta).to(torch.float32)  # [M_i, E_conf]
            keep = (adj_bin.sum(dim=0) > 0)                # [E_conf]

            if not bool(keep.any()):
                raise RuntimeError(
                    f"[AHL] 第 {i} 个尺度在 TopK(eta={self.eta}) + 阈值(beta={self.beta}) "
                    f"后所有超边均为空：M_i={M_i}, E_conf={E_conf}。"
                    f"建议调大 t_kprime 或减小 t_gamma。"
                )

            adj_bin = adj_bin[:, keep]   # [M_i, E_i]
            E_i = int(adj_bin.size(1))
            if E_i <= 0:
                raise RuntimeError(
                    f"[AHL] 第 {i} 个尺度过滤后 E_i={E_i}，出现空图。"
                )

            pairs = adj_bin.nonzero(as_tuple=False)  # [P, 2] (node, edge)
            if pairs.numel() == 0:
                raise RuntimeError(
                    f"[AHL] 第 {i} 个尺度过滤后没有任何 (node,edge) 关联："
                    f"M_i={M_i}, E_i={E_i}。"
                )

            mask_idx = pairs.t().contiguous()  # [2, P]

            # 此处不再额外做索引修补，交由主模型统一检查。
            hyperedge_all.append({
                'M': M_i,
                'E': E_i,
                'mask': mask_idx,
                'prob_val': None,
            })

        return hyperedge_all
