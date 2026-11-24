# -*- coding: utf-8 -*-
"""
models/sthyper/ops/hypergraph_conv.py  v0.2.2  (2025-12-13)

【版本号与修改记录（changelog）】
- v0.2.2 [StrictEmptyGraph][No-Fallback]
  1) 移除所有“空图退化为 Identity”的兜底逻辑：
     - 若 hyperedge_index 为 None 或 numel()==0，直接抛 RuntimeError；
     - 若 nodes/edges 在检查后为空，直接抛 RuntimeError；
     - 若计算得到的超边数 Euniq<=0，同样视为构图错误；
     空图策略统一交由上游（例如局部 refine 在 sample 级别退化为 Identity）。
  2) 保留 v0.2.1-minfix 中的布局假设与 CPU 侧索引边界检查，
     仍以 x.shape[1] 作为真实节点数 N_from_x，避免 CUDA 端越界。

- v0.2.1-minfix [SafeIndex][BNCLayout]
  1) 统一假定输入 x 为 [B, N, C]（第 0 维是 batch，第 1 维是节点）；
  2) 删除所有“自动猜测布局”的逻辑；
  3) 在 forward 入口对 node 索引做 0 <= node_idx < N 的边界检查，
     一旦不满足立即在 CPU 抛 RuntimeError，避免 CUDA Scatter/Gather 内核越界。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, degree


class HypergraphConv(MessagePassing):
    """
    intra-scale 超图卷积（N→E→N 两跳传播） + NHC 结构约束

    约定：输入 x 的形状固定为 [B, N, C]
        - B: batch size
        - N: 节点数
        - C: 特征维度

    注意：
      - 本模块不再对“空图”做任何兜底处理，一旦 hyperedge_index 为空直接报错；
      - 若有 per-sample 空图退化为 Identity 的需求，应在上游（例如
        Model._hyperconv_with_local_refine）里对该样本跳过调用本模块。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_attention: bool = True,
                 heads: int = 4,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.1,
                 bias: bool = False,
                 gamma: float = 0.2,
                 lambda_nc: float = 0.5):
        super(HypergraphConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = bool(use_attention)

        self.heads = int(heads) if self.use_attention else 1
        self.concat = bool(concat) if self.use_attention else True
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)

        # NHC 超参
        self.gamma = float(gamma)
        self.lambda_nc = float(lambda_nc)

        # 线性映射
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        # 注意力参数: [1, H, 2 * (C_out / H)]
        if self.use_attention:
            self.att = Parameter(
                torch.Tensor(1, self.heads, 2 * int(out_channels / self.heads))
            )
        else:
            self.register_parameter('att', None)

        # bias
        if bias and self.concat:
            self.bias = Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not self.concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.att is not None:
            glorot(self.att)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm, alpha):
        # norm: 按“目标端”索引的缩放因子
        out = norm[edge_index_i].view(-1, 1) * x_j
        if alpha is not None:
            out = alpha * out
        return out

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor):
        """
        x:  [B, N, C]
        hyperedge_index: [2, P], (src=node_idx, dst=edge_idx)
        """
        # ---------- 0. 线性变换 & 基本检查 ----------
        x = torch.matmul(x, self.weight)  # [B, N, C_out]
        if x.dim() != 3:
            raise ValueError(
                f"[HypergraphConv] x must be 3D [B,N,C], got {tuple(x.shape)}"
            )

        # 空图一律视为构图错误，由上游负责兜底处理
        if hyperedge_index is None:
            raise RuntimeError(
                "[HypergraphConv] hyperedge_index is None —— "
                "请在上游构图阶段保证非空，或在局部 refine 中自行退化为 Identity。"
            )
        if hyperedge_index.numel() == 0:
            raise RuntimeError(
                "[HypergraphConv] hyperedge_index 为空 —— "
                "当前路径不允许空超图，请检查 AHL/构图逻辑。"
            )

        if hyperedge_index.dim() != 2 or hyperedge_index.size(0) != 2:
            raise ValueError(
                f"[HypergraphConv] hyperedge_index must have shape [2, P], "
                f"got {tuple(hyperedge_index.shape)}"
            )

        # ---------- 1. 索引边界检查 ----------
        B, N_from_x, D = x.shape
        device = x.device
        dtype = x.dtype

        nodes = hyperedge_index[0].long()  # [P]
        edges = hyperedge_index[1].long()  # [P]

        if nodes.numel() == 0 or edges.numel() == 0:
            raise RuntimeError(
                "[HypergraphConv] hyperedge_index 在检查前即无有效 (node,edge)，"
                f"N_from_x={N_from_x}。"
            )

        n_min = int(nodes.min().item())
        n_max = int(nodes.max().item())
        if n_min < 0 or n_max >= N_from_x:
            raise RuntimeError(
                "[HypergraphConv] node index out of range: "
                f"found range [{n_min}, {n_max}] with N_from_x={N_from_x}"
            )

        e_min = int(edges.min().item())
        if e_min < 0:
            raise RuntimeError(
                f"[HypergraphConv] edge index contains negative values: min={e_min}"
            )
        Euniq = int(edges.max().item()) + 1 if edges.numel() > 0 else 0
        if Euniq <= 0:
            raise RuntimeError(
                "[HypergraphConv] 计算得到的超边数 Euniq<=0 —— "
                "请检查构图逻辑是否错误地传入了空超边集合。"
            )

        # ---------- 2. 统一到 [N, B, D] ----------
        # 约定 x 为 [B,N,C]，这里固定转成 [N,B,D]
        x_nb = x.permute(1, 0, 2).contiguous()  # [N, B, D]
        N, B_eff, D_eff = x_nb.shape  # N == N_from_x

        # ---------- 3. N → E：汇聚超边原型 ----------
        vals = x_nb.index_select(0, nodes)                # [P,B,D]
        proto = torch.zeros(Euniq, B_eff, D_eff,
                            device=device, dtype=dtype)   # [E,B,D]
        proto.index_add_(0, edges, vals)                  # ∑_{v∈e} x_v

        # ---------- 4. NHC 结构约束 ----------
        # 4.1 超边间约束 L_hyper
        proto_hat = F.normalize(proto, p=2, dim=2)  # [E,B,D]
        dists = []
        for b in range(B_eff):
            d_b = torch.cdist(proto_hat[:, b, :], proto_hat[:, b, :], p=2)
            dists.append(d_b.unsqueeze(0))
        dist = torch.stack(dists, dim=0)  # [B_eff,E,E]

        cos = (1.0 - 0.5 * dist.pow(2)).clamp(-1.0, 1.0)
        alpha01 = 0.5 * (cos + 1.0)  # ∈[0,1]

        loss_pos = alpha01 * dist
        loss_neg = (1.0 - alpha01) * F.relu(self.gamma - dist)
        loss_hyper = (loss_pos + loss_neg).mean()

        # 4.2 节点–超边一致性约束 L_node
        x_i = x_nb.index_select(0, nodes)   # [P,B,D]
        x_j = proto.index_select(0, edges)  # [P,B,D]
        if x_i.numel() > 0:
            constrain_raw = (x_i - x_j).mean()
            L_node = constrain_raw.abs()
        else:
            # 理论上不会到这里（nodes.numel() 之前已检查），加一个兜底
            L_node = x_nb.new_tensor(0.0)

        loss_const = L_node + self.lambda_nc * loss_hyper

        # ---------- 5. 注意力 α(v,e) ----------
        if self.use_attention:
            cat = torch.cat([x_i, x_j], dim=-1)  # [P,B,2D_eff]

            if self.heads == 1:
                att_vec = self.att.view(1, 1, -1)           # [1,1,2D_eff]
                alpha = (cat * att_vec).sum(dim=-1)         # [P,B]
            else:
                cat_h = cat.view(cat.size(0), cat.size(1), self.heads, -1)
                alpha = (cat_h * self.att).sum(dim=-1).mean(dim=-1)  # [P,B]

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, nodes, num_nodes=N)  # 按节点分组 softmax
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            alpha_bd = alpha.unsqueeze(-1).expand(-1, -1, D_eff).reshape(
                alpha.size(0), B_eff * D_eff
            )
        else:
            alpha_bd = None

        # ---------- 6. N → E → N 两跳传播（对称规范化） ----------
        x_flat = x_nb.reshape(N, B_eff * D_eff)  # [N,B*D_eff]

        he_index = torch.stack([nodes, edges], dim=0)  # [2,P]

        # 超边度 De^{-1}
        Edeg = 1.0 / degree(edges, Euniq, dtype=x_flat.dtype)
        Edeg[Edeg == float("inf")] = 0

        # 节点度 Dv^{-1/2}
        Ddeg = degree(nodes, N, dtype=x_flat.dtype)
        Ddeg_sqrt_inv = (Ddeg.clamp(min=1)).pow(-0.5)

        # 预缩放：D_v^{-1/2}
        x_flat = Ddeg_sqrt_inv.view(-1, 1) * x_flat

        # N → E
        self.flow = 'source_to_target'
        out = self.propagate(he_index, x=x_flat, norm=Edeg, alpha=alpha_bd)

        # E → N
        ones = x_flat.new_ones(N)
        self.flow = 'target_to_source'
        out = self.propagate(he_index, x=out, norm=ones, alpha=alpha_bd)

        # 后缩放：再乘一次 D_v^{-1/2}
        out = Ddeg_sqrt_inv.view(-1, 1) * out

        # ---------- 7. 还原为 [B,N,D_eff] + 可选 bias ----------
        out = out.view(N, B_eff, D_eff).transpose(0, 1).contiguous()  # [B,N,D_eff]

        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)

        return out, loss_const

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
