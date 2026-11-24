# -*- coding: utf-8 -*-
"""
models/sthyper/e2e/e2e_prop.py
Version: v0.7.8  (2025-12-30)

【版本号与修改记录（changelog）】
- v0.7.8 [Revert][FirstLogic+ManualThresholdBudget]
  1) 结构端完全回归“第一版思路”：
     receiver->sender attention → row-topk(fan-in) → 强制 self-loop → row-normalize。
  2) 将 v0.7.7 的 self-loop cap（_cap_self_loop_mass）替换为“手动阈值+手动预算”的确定性重分配：
     - tau：off-diag 概率阈值（<tau 置零，不新增连接）
     - self_cap：每行 self-loop 最大质量
     - neigh_min：每行 off-diag 最小总质量（必要时从 self-loop 转移给已存在的 off-diag）
     - off-diag 全 0 且需要转移时：可选 ring_fallback 注入 1 个邻居（可关）
  3) 其它功能保持不变：_MANUAL_CF / export / layer_kill / ProtoMemory 等不动。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.utils import _smoothness_loss
from typing import Optional, Dict, Any


# ======================================================================
# 手动反事实结构/内容干预配置（只改这里；不新增任何 CLI flag）
# ======================================================================
_MANUAL_CF: Dict[str, Any] = dict(
    enable=True,                 # 默认关闭，保证 baseline 可复现
    mode="rand_dense_noself",            # none | tied | row_shuffle | swap_batch | layer_mask | rand_dense_noself | layer_kill
    seed=1,                       # 随机干预用 seed（保证可复现）
    k=16,                         # row_shuffle 的 top-k（保持每行权重分布）
    layer_mask=[1, 1, 1, 1, 1],      # layer_mask/layer_kill 使用：0=干预，1=正常；长度>=层数即可
    apply_when="always",          # "eval"（仅eval/test生效）| "always"（训练也生效）
    tied_use_mean=False,          # tied 时是否用“在线均值结构”替代“第1层结构”（默认 False=用第1层）
)

# ======================================================================
# 手动分配阈值（你只改这里；不新增任何 CLI flag）
# ======================================================================
_MANUAL_ALLOC: Dict[str, Any] = dict(
    # (1) 稀疏阈值：对 off-diag 小于 tau 的连接直接置零（self-loop 不受影响）
    tau=0.0,            # 建议 0.0~0.05；0=不做阈值裁剪

    # (2) 质量预算：控制 self-loop 与邻居的质量分配（你手动设计的“分配阈值”）
    self_cap=0.3,      # self-loop 最大质量（建议 0.2~0.6）
    neigh_min=0.7,     # off-diag 最小总质量（建议 0.4~0.8）

    # (3) 若 off-diag 全 0 但又需要把 self-loop 质量转移出去，是否注入一个确定性邻居（ring）
    ring_fallback=True, # True 更稳定；False 则该行只能退化为纯 self-loop

    eps=1e-12,
)


class _ChebyMixin:
    @staticmethod
    def _cheby_filter(U, Srow, theta):
        """
        Chebyshev 多项式图滤波：
          U:    [..., E, C]
          Srow: [E, E] 行随机结构矩阵
          theta: [K+1]
        """
        K = theta.numel() - 1
        if K <= 0:
            return theta[0] * U
        T0 = U
        out = theta[0] * T0
        T1 = torch.matmul(Srow, U)
        out = out + theta[1] * T1
        if K == 1:
            return out
        Tk_minus_1, Tk = T0, T1
        for k in range(2, K + 1):
            Tk_plus_1 = 2.0 * torch.matmul(Srow, Tk) - Tk_minus_1
            out = out + theta[k] * Tk_plus_1
            Tk_minus_1, Tk = Tk, Tk_plus_1
        return out


# ======================================================================
# 工具：事件多样性 / 事件→节点重构
# ======================================================================

def _event_diversity_loss(Z: torch.Tensor) -> torch.Tensor:
    """
    事件语义多样性损失：
      - 将所有事件 embedding L2 归一化；
      - 计算两两余弦相似度；
      - 惩罚非对角元素的平方均值，鼓励事件“张开”。

    参数
    ----
    Z: [B, E, D]
    """
    if Z.dim() != 3:
        raise ValueError(f"_event_diversity_loss expects [B,E,D], got {Z.shape}")
    B, E, D = Z.shape
    N = B * E
    if N <= 1:
        return Z.new_zeros(())
    Z_flat = Z.reshape(N, D)
    Z_norm = F.normalize(Z_flat, p=2, dim=-1, eps=1e-12)   # [N, D]
    sim = Z_norm @ Z_norm.t()                              # [N, N]
    eye = torch.eye(N, device=Z.device, dtype=torch.bool)
    sim_off = sim[~eye]
    return (sim_off * sim_off).mean()


def _node_reconstruct_loss(
    Xn_mean: torch.Tensor,
    Ze_mean: torch.Tensor,
    mask: torch.Tensor,
    M: int,
    E: int,
) -> torch.Tensor:
    """
    事件→节点重构损失 L_node：
      - 利用“节点参与的事件”以及这些事件的 embedding，把节点特征重构出来，
        度量与原始节点特征的 MSE。

    参数
    ----
    Xn_mean: [M, D]，节点特征（对 batch 求平均后）
    Ze_mean: [E, D]，事件 embedding（对 batch 求平均后）
    mask:    [2, nnz]，节点→事件稀疏连接
    """
    if M <= 0 or E <= 0 or mask is None or mask.numel() == 0:
        return Xn_mean.new_zeros(())

    device = Xn_mean.device
    D = Xn_mean.size(-1)

    if Ze_mean.size(0) != E:
        raise ValueError(f"_node_reconstruct_loss: Ze_mean E={Ze_mean.size(0)} != E={E}")
    if Xn_mean.size(0) != M:
        raise ValueError(f"_node_reconstruct_loss: Xn_mean M={Xn_mean.size(0)} != M={M}")

    mask = mask.long().contiguous().to(device=device)
    if mask.dim() != 2 or mask.size(0) != 2:
        raise ValueError(f"_node_reconstruct_loss: mask must be [2, nnz], got {mask.shape}")
    n_idx = mask[0]
    e_idx = mask[1]
    if n_idx.numel() == 0:
        return Xn_mean.new_zeros(())

    deg_node = torch.bincount(n_idx, minlength=max(1, M)).to(Xn_mean.dtype).clamp_min(1.0)  # [M]
    w_ne = (1.0 / deg_node[n_idx]).unsqueeze(-1)                                            # [nnz, 1]

    feat_e = Ze_mean[e_idx]                                                                 # [nnz, D]
    contrib = w_ne * feat_e                                                                 # [nnz, D]

    hat_X = Xn_mean.new_zeros(M, D)
    hat_X.index_add_(0, n_idx, contrib)                                                     # [M, D]

    diff = hat_X - Xn_mean
    return (diff * diff).mean()


class SemanticEventPropV1(nn.Module):
    """
    V1：语义事件传播 + 样本级 gating（per-sample attention 构图）

    - A_base_b: [B,E,E]  每个样本自己算一次“语义事件图”（来自 Zm[b] 的 self-attention）
    - A_b:      [B,E,E]  最终用于传播的图：无 meta 时等于 A_base_b；有 meta 时在 log-space 上做 gating
    - last_A_base:  [E,E]  便于导出/可视化，取 A_base_b 的 batch 平均（更稳定）
    - last_A_batch: [B,E,E]
    """

    def __init__(
        self,
        configs,
        D: int,
        topk: int = 16,
        dropout: float = 0.1,
        expand: int = 2,
        ffn_ratio: float = 2.0,
        cheby_order: int = 2,
        num_heads: int = 4,
        mem_slots: int = 16,
        w_div: float = 0.5,
        w_node: float = 0.5,
        meta_dim: int = 0,
        meta_hidden: int = 32,
        gate_strength: float = 1.0,
        use_event_propagation: bool = True,
        hidden_dim: int = 128,
        cheby_use_batch: bool = True,
    ):
        super().__init__()
        self.topk = int(topk)
        self.drop = nn.Dropout(dropout)

        # 1) 事件塔内部维度 Dm_eff
        event_d_model = getattr(configs, "event_d_model", None)
        if event_d_model is None or event_d_model <= 0:
            event_d_model = int(expand * D)
        event_d_model = max(1, int(event_d_model))

        self.num_heads = max(1, int(num_heads))
        head_dim = int(math.ceil(event_d_model / self.num_heads))
        self.head_dim = max(1, head_dim)
        Dm_eff = self.num_heads * self.head_dim

        self.pre = nn.LayerNorm(D)
        self.up = nn.Linear(D, Dm_eff, bias=True)

        self.q = nn.Linear(Dm_eff, Dm_eff, bias=True)
        self.k = nn.Linear(Dm_eff, Dm_eff, bias=True)
        self.v = nn.Linear(Dm_eff, Dm_eff, bias=True)

        # 2) 多跳阶数（这里实际用于扩散混合）
        hop_order_cfg = getattr(configs, "cheby_order", cheby_order)
        self.cheby_order = int(max(0, hop_order_cfg))

        theta_init = torch.zeros(self.cheby_order + 1, dtype=torch.float32)
        if theta_init.numel() > 0:
            theta_init[0] = 6.0
        self.theta = nn.Parameter(theta_init)

        self.cheby_use_batch = bool(cheby_use_batch)

        # 3) 输出投影 + FFN
        self.down = nn.Linear(Dm_eff, D)
        self.post = nn.LayerNorm(D)

        event_ffn_dim = getattr(configs, "event_ffn_dim", None)
        if event_ffn_dim is None or event_ffn_dim <= 0:
            if hidden_dim is not None and hidden_dim > 0:
                event_ffn_dim = hidden_dim
            else:
                event_ffn_dim = int(ffn_ratio * D)
        hidden = max(1, int(event_ffn_dim))

        self.ffn = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, D),
        )

        # 4) 正则 & gating
        self.w_div = float(w_div)
        self.w_node = float(w_node)

        self.meta_dim = int(meta_dim)
        self.gate_strength = float(gate_strength)
        if self.meta_dim > 0:
            self.meta_proj = nn.Linear(self.meta_dim, int(meta_hidden))
            self.meta_gate = nn.Linear(int(meta_hidden), 1)
        else:
            self.meta_proj = None
            self.meta_gate = None

        self.use_event_propagation = bool(use_event_propagation)

        self.last_A_base = None   # [E,E]
        self.last_A_batch = None  # [B,E,E]

        # export cache
        self._last_content_export = None

        # --------- 手动结构干预：跨层 override（由 Tower 设置）---------
        self._override_A_b = None             # [B,E,E] or None
        self._override_A_base_mean = None     # [E,E] or None

    # ------------------------- manual counterfactual helpers -------------------------
    @torch.no_grad()
    def _set_override_adjacency(self, A_b: torch.Tensor = None, A_base_mean: torch.Tensor = None):
        """Tower 用于 tied / layer_mask：强行覆盖本层 A_b / A_base_mean。"""
        self._override_A_b = A_b
        self._override_A_base_mean = A_base_mean

    @staticmethod
    def _make_generator(seed: int, device):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) & 0x7FFFFFFF)
        return g

    @torch.no_grad()
    def _swap_batch(self, A_b: torch.Tensor, seed: int):
        """Cross-sample structure swap：batch 内随机置换 A_b[b]。"""
        if A_b is None or (not torch.is_tensor(A_b)) or A_b.numel() == 0:
            return A_b
        B = int(A_b.size(0))
        if B <= 1:
            return A_b
        device = A_b.device
        g = self._make_generator(seed, device=device)
        perm = torch.randperm(B, generator=g, device=device)
        return A_b.index_select(0, perm)

    @torch.no_grad()
    def _row_edge_shuffle_batch_keep_weights(self, A_b: torch.Tensor, k: int, seed: int):
        """
        Row-permutation / edge-shuffle（保权重分布，毁语义对应）
        - 对每行取 top-(k-1) 的“权重值集合”
        - 随机选择 (k-1) 个非 self 的节点承载这些权重
        - 固定 self-loop
        """
        if A_b is None or (not torch.is_tensor(A_b)) or A_b.numel() == 0:
            return A_b
        B, E, _ = A_b.shape
        if E <= 1:
            return A_b

        k = max(1, min(int(k), int(E)))
        device = A_b.device
        g = self._make_generator(seed, device=device)
        idx = torch.arange(E, device=device)

        A_new = torch.zeros_like(A_b)

        # 固定 self-loop 权重
        self_w = A_b[:, idx, idx].clone()
        A_new[:, idx, idx] = self_w

        if k == 1:
            A_new = torch.eye(E, device=device, dtype=A_b.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
            return A_new

        # 取 off-diagonal top-(k-1) 的权重“值”
        A_no_self = A_b.clone()
        A_no_self[:, idx, idx] = -1.0
        topv, _ = torch.topk(A_no_self, k=k - 1, dim=-1)  # [B,E,k-1]

        for b in range(B):
            for i in range(E):
                cand = torch.cat([idx[:i], idx[i + 1:]])  # [E-1]
                perm = cand[torch.randperm(E - 1, generator=g, device=device)[: (k - 1)]]
                A_new[b, i, perm] = topv[b, i]

        A_new = self._fix_zero_rows_square(A_new)
        A_new = A_new / A_new.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return A_new

    @torch.no_grad()
    def _rand_dense_adj(self, B: int, E: int, seed: int, break_self: bool = True) -> torch.Tensor:
        """
        生成随机稠密行随机矩阵 A_b（每行归一化），可选破坏自环（diag=0）。
        """
        device = next(self.parameters()).device
        g = self._make_generator(seed, device=device)

        if E <= 1:
            I = torch.eye(E, device=device, dtype=torch.float32)
            return I.unsqueeze(0).expand(B, -1, -1).contiguous()

        A = torch.rand((B, E, E), generator=g, device=device, dtype=torch.float32)

        if break_self:
            idx = torch.arange(E, device=device)
            A[:, idx, idx] = 0.0

        A = self._fix_zero_rows_square(A)

        if break_self:
            idx = torch.arange(E, device=device)
            A[:, idx, idx] = 0.0

        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return A

    # ------------------------- basic helpers -------------------------
    @staticmethod
    def _l2norm_lastdim(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.norm(X, p=2, dim=-1).clamp_min(eps)

    @staticmethod
    def _fix_zero_rows_square(A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            E = A.size(0)
            rs = A.sum(dim=-1)
            bad = (rs <= 0)
            if bad.any():
                idx = bad.nonzero(as_tuple=True)[0]
                A = A.clone()
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
            return A
        elif A.dim() == 3:
            rs = A.sum(dim=-1)
            bad = (rs <= 0)
            if bad.any():
                b_idx, e_idx = bad.nonzero(as_tuple=True)
                A = A.clone()
                A[b_idx, e_idx, :] = 0.0
                A[b_idx, e_idx, e_idx] = 1.0
            return A
        return A

    @staticmethod
    def _apply_offdiag_threshold_batched(A: torch.Tensor, tau: float, eps: float = 1e-12) -> torch.Tensor:
        """
        对 off-diag 做阈值裁剪：A_off < tau → 0；不影响 self-loop。
        仅裁剪，不做预算；预算由 _rebalance_self_loop_budget 负责。
        A: [B,E,E]
        """
        if A is None or (not torch.is_tensor(A)) or A.numel() == 0:
            return A
        if tau <= 0.0:
            return A
        if A.dim() != 3:
            raise ValueError(f"_apply_offdiag_threshold_batched expects [B,E,E], got {tuple(A.shape)}")
        B, E, _ = A.shape
        if E <= 1:
            return A
        A32 = A.to(torch.float32)
        idx = torch.arange(E, device=A32.device)
        off_mask = torch.ones_like(A32, dtype=torch.bool)
        off_mask[:, idx, idx] = False
        A32 = A32.masked_fill(off_mask & (A32 < float(tau)), 0.0)
        A32 = SemanticEventPropV1._fix_zero_rows_square(A32)
        A32 = A32 / A32.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        return A32.to(dtype=A.dtype)

    @staticmethod
    def _rebalance_self_loop_budget(
        A: torch.Tensor,
        self_cap: float,
        neigh_min: float,
        ring_fallback: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        第一版逻辑上的“后处理”：在不破坏 row-stochastic 的前提下，手动控制每行 self-loop 与邻居的质量分配。
        - 不新增连接：只在该行已有 off-diag 上按比例分配（除非 off-diag 全 0 且 ring_fallback=True）
        A: [B,E,E]，要求每行非负且已做过 row-normalize
        """
        if A is None or (not torch.is_tensor(A)) or A.numel() == 0:
            return A
        if A.dim() != 3:
            raise ValueError(f"_rebalance_self_loop_budget expects [B,E,E], got {tuple(A.shape)}")

        B, E, _ = A.shape
        if E <= 1:
            return A

        A32 = A.to(torch.float32)
        idx = torch.arange(E, device=A32.device)

        # 约束可行化：diag <= min(self_cap, 1-neigh_min)
        diag_cap = min(float(self_cap), 1.0 - float(neigh_min))
        diag_cap = max(0.0, min(1.0, diag_cap))

        diag = A32[:, idx, idx].clamp_min(0.0)                   # [B,E]
        off = A32.clone()
        off[:, idx, idx] = 0.0
        off_sum = off.sum(dim=-1)                                 # [B,E]

        # 需要从 diag 转移到 off 的质量
        target_diag = torch.minimum(diag, torch.full_like(diag, diag_cap))
        delta = (diag - target_diag).clamp_min(0.0)               # [B,E]

        if float(delta.max().item()) <= float(eps):
            return A32.to(dtype=A.dtype)

        # 若 off_sum==0 且需要转移：可选 ring 注入一个邻居承接
        if ring_fallback:
            bad = (off_sum <= eps) & (delta > eps)
            if bad.any():
                b_idx, e_idx = bad.nonzero(as_tuple=True)
                j = (e_idx + 1) % E
                off[b_idx, e_idx, j] = 1.0
                off_sum = off.sum(dim=-1)

        # 若仍无 off 支撑集：无法转移，则保持原样（该行退化为纯 self）
        safe = (off_sum > eps).to(A32.dtype)  # [B,E]
        delta = delta * safe

        # 按已有 off 权重比例分配 delta
        off_norm = off / off_sum.clamp_min(eps).unsqueeze(-1)     # [B,E,E]
        off_new = off_norm * (off_sum + delta).unsqueeze(-1)      # 把 delta 加到 off 总量上

        A_new = off_new
        A_new[:, idx, idx] = diag - delta

        # 最终再做一次严格归一化（数值安全）
        A_new = SemanticEventPropV1._fix_zero_rows_square(A_new)
        A_new = A_new / A_new.sum(dim=-1, keepdim=True).clamp_min(eps)
        return A_new.to(dtype=A.dtype)

    @staticmethod
    def _post_process_adj_batched(A_b: torch.Tensor) -> torch.Tensor:
        """
        统一后处理（用于 build / override / CF）：
        1) fix_zero_rows + row-normalize
        2) off-diag 阈值 tau
        3) self-loop 预算重分配（self_cap / neigh_min）
        """
        alloc = _MANUAL_ALLOC if isinstance(_MANUAL_ALLOC, dict) else {}
        eps = float(alloc.get("eps", 1e-12))
        tau = float(alloc.get("tau", 0.0))
        self_cap = float(alloc.get("self_cap", 1.0))
        neigh_min = float(alloc.get("neigh_min", 0.0))
        ring_fb = bool(alloc.get("ring_fallback", True))

        A32 = A_b.to(torch.float32)
        A32 = SemanticEventPropV1._fix_zero_rows_square(A32)
        A32 = A32 / A32.sum(dim=-1, keepdim=True).clamp_min(eps)

        if tau > 0.0:
            A32 = SemanticEventPropV1._apply_offdiag_threshold_batched(A32, tau=tau, eps=eps).to(torch.float32)

        A32 = SemanticEventPropV1._rebalance_self_loop_budget(
            A32, self_cap=self_cap, neigh_min=neigh_min, ring_fallback=ring_fb, eps=eps
        ).to(torch.float32)

        A32 = SemanticEventPropV1._fix_zero_rows_square(A32)
        A32 = A32 / A32.sum(dim=-1, keepdim=True).clamp_min(eps)
        return A32.to(dtype=A_b.dtype)

    @staticmethod
    def _post_process_adj_base(A: torch.Tensor) -> torch.Tensor:
        """
        base [E,E] 的统一后处理（沿用 batched 逻辑）。
        """
        if A is None or (not torch.is_tensor(A)) or A.numel() == 0:
            return A
        if A.dim() != 2:
            raise ValueError(f"_post_process_adj_base expects [E,E], got {tuple(A.shape)}")
        A1 = A.unsqueeze(0)  # [1,E,E]
        A1 = SemanticEventPropV1._post_process_adj_batched(A1)
        return A1.squeeze(0)

    # ------------------------- struct export helpers -------------------------
    @staticmethod
    def _row_topk_to_incidence_base(A: torch.Tensor, k: int):
        if A is None or (not torch.is_tensor(A)) or A.numel() == 0:
            return None
        E = int(A.size(0))
        k = max(1, min(int(k), E))
        topv, topi = torch.topk(A, k=k, dim=-1)  # [E,k]
        edge_idx = torch.arange(E, device=A.device).view(E, 1).expand(E, k).reshape(-1)
        node_idx = topi.reshape(-1)
        w = topv.reshape(-1)
        keep = (w > 0)
        mask2 = torch.stack([node_idx[keep], edge_idx[keep]], dim=0).long()
        return {
            "shape": (E, E),
            "topi": topi.detach().cpu().to(torch.int32),
            "topv": topv.detach().cpu().to(torch.float32),
            "incidence_mask": mask2.detach().cpu().to(torch.int32),
            "incidence_weight": w[keep].detach().cpu().to(torch.float32),
        }

    @staticmethod
    def _row_topk_to_incidence_batch(A_b: torch.Tensor, k: int):
        if A_b is None or (not torch.is_tensor(A_b)) or A_b.numel() == 0:
            return None
        B, E, _ = A_b.shape
        k = max(1, min(int(k), int(E)))
        topv, topi = torch.topk(A_b, k=k, dim=-1)  # [B,E,k]
        b_idx = torch.arange(B, device=A_b.device).view(B, 1, 1).expand(B, E, k).reshape(-1)
        edge_idx = torch.arange(E, device=A_b.device).view(1, E, 1).expand(B, E, k).reshape(-1)
        node_idx = topi.reshape(-1)
        w = topv.reshape(-1)
        keep = (w > 0)
        mask3 = torch.stack([b_idx[keep], node_idx[keep], edge_idx[keep]], dim=0).long()
        return {
            "shape": (int(B), int(E), int(E)),
            "topi": topi.detach().cpu().to(torch.int32),
            "topv": topv.detach().cpu().to(torch.float32),
            "incidence_mask": mask3.detach().cpu().to(torch.int32),
            "incidence_weight": w[keep].detach().cpu().to(torch.float32),
        }

    # ------------------------- per-sample semantic adj -------------------------
    def _build_semantic_adj_batched(self, Zm: torch.Tensor) -> torch.Tensor:
        """
        第一版思路（严格保持）：
        1) receiver->sender attention softmax（行随机）
        2) receiver 行 row-topk（fan-in 约束），强制包含 self-loop
        3) row-normalize
        4) (可选) 手动阈值 tau + 手动预算 self_cap/neigh_min
        """
        B, E, Dm_eff = Zm.shape
        H = self.num_heads
        dh = Dm_eff // H if H > 0 else Dm_eff

        Q = self.q(Zm).view(B, E, H, dh).permute(0, 2, 1, 3)  # [B,H,E,dh]
        K = self.k(Zm).view(B, E, H, dh).permute(0, 2, 1, 3)  # [B,H,E,dh]
        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(max(dh, 1))  # [B,H,E_receiver,E_sender]

        # 1) receiver->sender softmax（第一版语义）
        A = torch.softmax(logits, dim=-1).mean(dim=1)  # [B,E,E] row=receiver, col=sender

        # 2) receiver 行 row-topk（保留 k-1 个 off-diag + 强制 self-loop）
        if 0 < self.topk < E:
            k = int(self.topk)
            idx = torch.arange(E, device=Zm.device)

            if k == 1:
                A_new = torch.zeros_like(A)
            else:
                A_no_self = A.clone()
                A_no_self[:, idx, idx] = -1.0
                topv, topi = torch.topk(A_no_self, k=k - 1, dim=-1)  # [B,E,k-1]
                A_new = torch.zeros_like(A)
                A_new.scatter_(dim=-1, index=topi, src=topv)

            # 强制包含 self-loop：拷回原始对角
            A_new[:, idx, idx] = A[:, idx, idx]
            A = A_new

        # 3) fix + row-normalize
        A = self._fix_zero_rows_square(A)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(float(_MANUAL_ALLOC.get("eps", 1e-12)))

        # 4) 手动阈值 + 手动预算（你可直接改 _MANUAL_ALLOC）
        A = self._post_process_adj_batched(A.to(torch.float32)).to(dtype=A.dtype)
        return A

    # ------------------------- diffusion multi-hop (batched) -------------------------
    def _diffusion_filter_batched(self, X: torch.Tensor, A: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        K = int(theta.numel() - 1)
        if K <= 0:
            return X
        alpha = torch.softmax(theta.to(dtype=torch.float32), dim=0)
        Xk = X
        out = alpha[0] * Xk
        for k in range(1, K + 1):
            Xk = torch.bmm(A, Xk)
            out = out + alpha[k] * Xk
        return out

    def forward(
        self,
        Z: torch.Tensor,                  # [B,E,D]
        meta: torch.Tensor = None,        # [B,E,meta_dim] or None
        node_feat: torch.Tensor = None,   # [B,M,D] or [M,D] or None
        mask: torch.Tensor = None,        # [2,nnz] node→event or None
        M: int = None,
    ):
        dtype_in = Z.dtype
        Z32 = Z.to(torch.float32)
        if Z32.dim() != 3:
            raise ValueError(f"Z must be [B,E,D], got {Z32.shape}")
        B, Ecnt, D = Z32.shape

        # 1) LN + 升维
        Zm = self.up(self.pre(Z32))  # [B,E,Dm_eff]

        # 2) per-sample 语义图 A_base_b（receiver->sender）
        A_base_b = self._build_semantic_adj_batched(Zm)   # [B,E,E]
        A_base_mean = A_base_b.mean(dim=0)                # [E,E]

        # 3) 最终用于传播的样本级图 A_b（当前实现：不启用 meta gating）
        A_b = A_base_b
        A_b = self._post_process_adj_batched(A_b)         # 统一后处理（含阈值+预算）

        # =========================
        # 手动反事实结构干预（不新增 CLI；由 _MANUAL_CF 控制）
        # =========================
        cf = _MANUAL_CF if isinstance(_MANUAL_CF, dict) else {}
        enable = bool(cf.get("enable", False))
        apply_when = str(cf.get("apply_when", "always")).lower()
        do_apply = enable and (apply_when == "always" or (apply_when == "eval" and (not self.training)))

        if do_apply:
            mode = str(cf.get("mode", "none")).lower()
            seed = int(cf.get("seed", 1))

            # (Tower override 优先级最高)
            if (self._override_A_b is not None) and torch.is_tensor(self._override_A_b):
                A_b = self._override_A_b.to(device=Z.device, dtype=torch.float32)
                A_b = self._post_process_adj_batched(A_b)

                if (self._override_A_base_mean is not None) and torch.is_tensor(self._override_A_base_mean):
                    A_base_mean = self._override_A_base_mean.to(device=Z.device, dtype=torch.float32)
                    A_base_mean = self._post_process_adj_base(A_base_mean)
                else:
                    A_base_mean = A_b.mean(dim=0)
                    A_base_mean = self._post_process_adj_base(A_base_mean)

                # 用完即清，避免跨 batch 污染
                self._override_A_b = None
                self._override_A_base_mean = None

            else:
                if mode == "row_shuffle":
                    k_eff = int(cf.get("k", self.topk if self.topk > 0 else int(Ecnt)))
                    A_b = self._row_edge_shuffle_batch_keep_weights(
                        A_b.to(torch.float32), k=k_eff, seed=seed
                    )
                    A_b = self._post_process_adj_batched(A_b)
                    A_base_mean = self._post_process_adj_base(A_b.mean(dim=0))

                elif mode == "swap_batch":
                    A_b = self._swap_batch(A_b.to(torch.float32), seed=seed + 99991)
                    A_b = self._post_process_adj_batched(A_b)
                    A_base_mean = self._post_process_adj_base(A_b.mean(dim=0))

                elif mode == "rand_dense_noself":
                    A_b = self._rand_dense_adj(B=B, E=Ecnt, seed=seed + 12345, break_self=True)
                    A_b = self._post_process_adj_batched(A_b)
                    A_base_mean = self._post_process_adj_base(A_b.mean(dim=0))

        # cast back
        A_b = A_b.to(dtype_in)
        A_base_mean = A_base_mean.to(dtype_in)

        # 记录邻接（最终用于传播）
        self.last_A_base = A_base_mean
        self.last_A_batch = A_b

        # 4) 事件内部传播
        if self.use_event_propagation:
            H = self.num_heads
            dh = Zm.size(-1) // H

            V_raw = self.v(Zm).view(B, Ecnt, H, dh).permute(0, 2, 1, 3)
            V0 = V_raw.permute(0, 2, 1, 3).contiguous().view(B, Ecnt, H * dh)

            if self.cheby_order <= 0:
                U = torch.bmm(A_b.to(torch.float32), V0)
            else:
                if self.cheby_use_batch:
                    U = self._diffusion_filter_batched(V0, A_b.to(torch.float32), self.theta)
                else:
                    A_mean = A_base_mean.to(torch.float32).unsqueeze(0).expand(B, -1, -1)
                    A_mean = self._post_process_adj_batched(A_mean)
                    U = self._diffusion_filter_batched(V0, A_mean.to(torch.float32), self.theta)

            ee_smooth = _smoothness_loss(U, A_b.to(torch.float32))
        else:
            U = Zm
            ee_smooth = Z.new_zeros(())

        # 5) 下采样回 D 维 + 残差 + FFN
        Y = self.down(U)
        Y = self.post(Z32 + self.drop(Y))
        Y = Y + self.drop(self.ffn(Y))

        # 6) 正则项
        A_safe = A_b.to(torch.float32).clamp_min(1e-8)
        e2e_entropy = -(A_safe * torch.log(A_safe)).sum(dim=-1).mean()

        if self.w_div > 0.0:
            raw_div = _event_diversity_loss(Y.to(torch.float32))
            event_div = self.w_div * raw_div
        else:
            event_div = Z.new_zeros(())

        if self.w_node > 0.0 and (node_feat is not None) and (mask is not None) and (M is not None):
            Xn32 = node_feat.to(torch.float32)
            if Xn32.dim() == 3:
                Xn_mean = Xn32.mean(dim=0)
            elif Xn32.dim() == 2:
                Xn_mean = Xn32
            else:
                raise ValueError(f"node_feat must be [B,M,D] or [M,D], got {Xn32.shape}")

            if Xn_mean.size(0) != int(M):
                raise ValueError(f"node_feat M={Xn_mean.size(0)} != M={M}")
            if Xn_mean.size(-1) != Y.size(-1):
                raise ValueError(f"node_feat dim={Xn_mean.size(-1)} != event dim={Y.size(-1)}")

            Ze_mean = Y.to(torch.float32).mean(dim=0)
            raw_node = _node_reconstruct_loss(
                Xn_mean=Xn_mean,
                Ze_mean=Ze_mean,
                mask=mask,
                M=int(M),
                E=int(Ecnt),
            )
            event_node = self.w_node * raw_node
        else:
            event_node = Z.new_zeros(())

        reg = {
            "e2e": e2e_entropy.to(dtype_in),
            "ee_smooth": ee_smooth.to(dtype_in),
            "event_div": event_div.to(dtype_in),
            "event_node": event_node.to(dtype_in),
        }

        # ------------------------- export cache（保持 v0.7.5：always） -------------------------
        R_mean = Y.mean(dim=1).detach().to(torch.float32).cpu()
        R_max = Y.max(dim=1).values.detach().to(torch.float32).cpu()
        R_mean_l2 = self._l2norm_lastdim(R_mean)
        R_max_l2 = self._l2norm_lastdim(R_max)

        Y_tokens = Y.detach().to(torch.float16).cpu()
        Y_l2norm = self._l2norm_lastdim(Y.detach().to(torch.float32)).cpu()

        k_export = int(self.topk) if self.topk > 0 else int(Ecnt)
        k_export = max(1, min(k_export, int(Ecnt)))

        base_struct = self._row_topk_to_incidence_base(A_base_mean.to(torch.float32), k=k_export)
        batch_struct = self._row_topk_to_incidence_batch(A_b.to(torch.float32), k=k_export)

        alpha = torch.softmax(self.theta.detach().to(torch.float32), dim=0).cpu().to(torch.float32)
        event_norm_mean = torch.norm(Y.detach().to(torch.float32), dim=-1).mean(dim=0).cpu().to(torch.float32)

        self._last_content_export = {
            "repr": {
                "R_mean": R_mean,
                "R_max":  R_max,
                "R_mean_l2": R_mean_l2,
                "R_max_l2":  R_max_l2,
            },
            "tokens": {
                "Y_tokens": Y_tokens,
                "Y_l2norm": Y_l2norm,
            },
            "struct": {
                "A_base": base_struct,
                "A_batch": batch_struct,
            },
            "diffusion": {
                "alpha": alpha,
                "K": int(self.cheby_order),
            },
            "stats": {
                "event_norm_mean": event_norm_mean,
                "e2e_entropy": float(e2e_entropy.detach().cpu().to(torch.float32)),
            }
        }

        return Y.to(dtype_in), reg, A_base_mean.to(dtype_in)

    @torch.no_grad()
    def get_content_export(self):
        return self._last_content_export


class SemanticEventTowerV2(nn.Module):
    """
    V2：多层语义事件塔（堆叠多层 SemanticEventPropV1）
    保持 v0.7.5：Prop 永远导出；Tower 永远汇总。
    """

    def __init__(
        self,
        configs,
        D: int,
        num_layers: int = 2,
        topk: int = 16,
        dropout: float = 0.1,
        expand: int = 2,
        ffn_ratio: float = 2.0,
        cheby_order: int = 2,
        num_heads: int = 4,
        mem_slots: int = 16,
        w_div: float = 0.3,
        w_node: float = 0.3,
        meta_dim: int = 0,
        meta_hidden: int = 32,
        gate_strength: float = 1.0,
        w_hier: float = 0.3,
        use_event_propagation: bool = True,
        # ---------- 内容端层级约束相关 ----------
        use_content_reg: bool = True,
        content_feat_dim: int = 0,
        content_w_span: float = 1.0,
        content_w_hf: float = 1.0,
        content_w_season: float = 0.0,
        content_w_trend: float = 0.0,
        content_margin_span: float = 0.0,
        content_margin_hf: float = 0.0,
        content_margin_season: float = 0.0,
        content_margin_trend: float = 0.0,
        content_season_layer: int = None,
        content_trend_layer: int = None,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"SemanticEventTowerV2: num_layers must > 0, got {num_layers}")
        self.num_layers = int(num_layers)
        self.w_hier = float(w_hier)
        self.use_event_propagation = bool(use_event_propagation)

        self.use_content_reg = bool(use_content_reg)
        self.content_feat_dim = int(content_feat_dim)
        self.content_w_span = float(content_w_span)
        self.content_w_hf = float(content_w_hf)

        self.content_w_season = float(content_w_season)
        self.content_w_trend = float(content_w_trend)
        self.content_margin_span = float(content_margin_span)
        self.content_margin_hf = float(content_margin_hf)
        self.content_margin_season = float(content_margin_season)
        self.content_margin_trend = float(content_margin_trend)
        self.content_season_layer = content_season_layer
        self.content_trend_layer = content_trend_layer

        self._last_mu_span = None
        self._last_mu_hf = None
        self._last_layer_stats = None
        self._last_content_export = None

        layers = []
        for li in range(self.num_layers):
            w_node_l = w_node if li == 0 else 0.0
            layer = SemanticEventPropV1(
                configs=configs,
                D=D,
                topk=topk,
                dropout=dropout,
                expand=expand,
                ffn_ratio=ffn_ratio,
                cheby_order=cheby_order,
                num_heads=num_heads,
                mem_slots=mem_slots,
                w_div=w_div,
                w_node=w_node_l,
                meta_dim=meta_dim,
                meta_hidden=meta_hidden,
                gate_strength=gate_strength,
                use_event_propagation=self.use_event_propagation,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def _compute_content_hier_loss(self, tower_Z, content_feat: torch.Tensor, device, dtype) -> torch.Tensor:
        if (not self.use_content_reg) or (content_feat is None):
            self._last_mu_span = None
            self._last_mu_hf = None
            return torch.zeros((), device=device, dtype=dtype)

        if content_feat.dim() == 3:
            cf32 = content_feat.to(torch.float32).mean(dim=0)
        elif content_feat.dim() == 2:
            cf32 = content_feat.to(torch.float32)
        else:
            self._last_mu_span = None
            self._last_mu_hf = None
            raise ValueError(f"content_feat must be [E,F] or [B,E,F], got {content_feat.shape}")

        E, Fdim = cf32.shape
        if self.content_feat_dim > 0 and Fdim != self.content_feat_dim:
            self._last_mu_span = None
            self._last_mu_hf = None
            raise ValueError(f"content_feat_dim={self.content_feat_dim}, but got {Fdim}")
        if Fdim < 2:
            self._last_mu_span = None
            self._last_mu_hf = None
            return torch.zeros((), device=device, dtype=dtype)

        span_feat = cf32[:, 0]
        hf_feat = cf32[:, 1]

        num_layers = len(tower_Z)
        if num_layers <= 1:
            self._last_mu_span = None
            self._last_mu_hf = None
            return torch.zeros((), device=device, dtype=dtype)

        mu_span = []
        mu_hf = []
        for Y_l in tower_Z:
            Y32 = Y_l.to(torch.float32)
            w = Y32.abs().mean(dim=(0, 2))
            w = w / w.sum().clamp_min(1e-12)
            mu_span.append((w * span_feat).sum())
            mu_hf.append((w * hf_feat).sum())

        mu_span_t = torch.stack(mu_span, dim=0)
        mu_hf_t = torch.stack(mu_hf, dim=0)

        self._last_mu_span = mu_span_t.detach()
        self._last_mu_hf = mu_hf_t.detach()

        L_span = torch.zeros((), device=device, dtype=torch.float32)
        if self.content_w_span > 0.0:
            margin = float(self.content_margin_span)
            for li in range(num_layers - 1):
                L_span = L_span + F.relu(mu_span_t[li] + margin - mu_span_t[li + 1])

        L_hf = torch.zeros((), device=device, dtype=torch.float32)
        if self.content_w_hf > 0.0:
            margin = float(self.content_margin_hf)
            for li in range(num_layers - 1):
                L_hf = L_hf + F.relu(mu_hf_t[li + 1] + margin - mu_hf_t[li])

        L_content = self.content_w_span * L_span + self.content_w_hf * L_hf
        return L_content.to(dtype)

    def get_content_stats(self):
        mu_span = self._last_mu_span.detach().cpu() if self._last_mu_span is not None else None
        mu_hf = self._last_mu_hf.detach().cpu() if self._last_mu_hf is not None else None
        return {"mu_span": mu_span, "mu_hf": mu_hf}

    def get_layer_stats(self, include_content: bool = True):
        stats = {}
        if self._last_layer_stats is not None:
            for k, v in self._last_layer_stats.items():
                stats[k] = None if v is None else v.detach().cpu()
        if include_content:
            stats.update(self.get_content_stats())
        return stats

    @torch.no_grad()
    def get_content_export(self):
        return self._last_content_export

    @staticmethod
    @torch.no_grad()
    def _patch_layer_export_after_kill(layer: nn.Module, Y_killed: torch.Tensor):
        """
        layer_kill 后修补该层 export，使 repr/tokens/stats 与被 kill 的输出一致。
        """
        if (not hasattr(layer, "_last_content_export")) or layer._last_content_export is None:
            return
        exp = layer._last_content_export
        if not isinstance(exp, dict):
            return

        Y0 = Y_killed.detach()
        R_mean = Y0.mean(dim=1).to(torch.float32).cpu()
        R_max = Y0.max(dim=1).values.to(torch.float32).cpu()
        R_mean_l2 = torch.norm(R_mean, p=2, dim=-1).clamp_min(1e-12)
        R_max_l2 = torch.norm(R_max, p=2, dim=-1).clamp_min(1e-12)

        Y_tokens = Y0.to(torch.float16).cpu()
        Y_l2norm = torch.norm(Y0.to(torch.float32), p=2, dim=-1).clamp_min(1e-12).cpu()
        event_norm_mean = torch.norm(Y0.to(torch.float32), p=2, dim=-1).mean(dim=0).cpu()

        if "repr" in exp and isinstance(exp["repr"], dict):
            exp["repr"]["R_mean"] = R_mean
            exp["repr"]["R_max"] = R_max
            exp["repr"]["R_mean_l2"] = R_mean_l2
            exp["repr"]["R_max_l2"] = R_max_l2

        if "tokens" in exp and isinstance(exp["tokens"], dict):
            exp["tokens"]["Y_tokens"] = Y_tokens
            exp["tokens"]["Y_l2norm"] = Y_l2norm

        if "stats" in exp and isinstance(exp["stats"], dict):
            exp["stats"]["event_norm_mean"] = event_norm_mean

        layer._last_content_export = exp

    def forward(
        self,
        Z: torch.Tensor,                  # [B,E,D]
        meta: torch.Tensor = None,
        node_feat: torch.Tensor = None,
        mask: torch.Tensor = None,
        M: int = None,
        content_feat: torch.Tensor = None
    ):
        dtype_in = Z.dtype
        self._last_layer_stats = None
        self._last_content_export = None

        if Z.dim() != 3:
            raise ValueError(f"SemanticEventTowerV2: Z must be [B,E,D], got {Z.shape}")
        B, E, D = Z.shape

        if E == 0:
            zero = Z.new_zeros(())
            reg_zero = {k: zero for k in ["e2e","ee_smooth","event_div","event_node","tower_hier","tower_content"]}
            tower_empty = {
                "Z_layers": [],
                "A_base_list": [],
                "A_batch_list": [],
                "mu_span": None,
                "mu_hf": None,
                "content_export": None,
            }
            self._last_mu_span = None
            self._last_mu_hf = None
            self._last_layer_stats = None
            self._last_content_export = None
            return Z, reg_zero, tower_empty

        zero = Z.new_zeros(())
        reg_total = {k: zero for k in ["e2e","ee_smooth","event_div","event_node","tower_hier","tower_content"]}

        Z_l = Z
        tower_Z = []
        tower_A_base = []
        tower_A_batch = []

        prev_A_base = None
        L_hier_raw = zero

        layer_e2e = []
        layer_ee_smooth = []
        layer_event_div = []
        layer_event_node = []
        layer_amp_mean = []
        layer_amp_std = []

        L = len(self.layers)
        repr_layers_mean = [None] * L
        repr_layers_max = [None] * L
        tokens_layers_Y = [None] * L
        tokens_layers_norm = [None] * L
        layers_export = [None] * L

        # -------------------- 手动反事实（由 _MANUAL_CF 控制） --------------------
        cf = _MANUAL_CF if isinstance(_MANUAL_CF, dict) else {}
        enable = bool(cf.get("enable", False))
        apply_when = str(cf.get("apply_when", "always")).lower()
        use_cf = enable and (apply_when == "always" or (apply_when == "eval" and (not self.training)))

        mode = str(cf.get("mode", "none")).lower() if use_cf else "none"
        tied_use_mean = bool(cf.get("tied_use_mean", False)) if use_cf else False

        layer_mask = cf.get("layer_mask", None) if use_cf else None
        layer_mask = layer_mask if isinstance(layer_mask, (list, tuple)) else None

        tied_A_b = None
        tied_A_base = None

        mean_A_b_acc = None
        mean_cnt = 0

        def _identity_batched(B0: int, E0: int, device):
            I = torch.eye(E0, device=device, dtype=torch.float32)
            return I.unsqueeze(0).expand(B0, -1, -1).contiguous(), I

        for li, layer in enumerate(self.layers):

            # (d) layer_mask：指定层强制 Identity adjacency（结构屏蔽）
            if use_cf and mode == "layer_mask" and (layer_mask is not None) and li < len(layer_mask) and int(layer_mask[li]) == 0:
                I_b, I_base = _identity_batched(B, E, Z.device)
                if hasattr(layer, "_set_override_adjacency"):
                    layer._set_override_adjacency(A_b=I_b, A_base_mean=I_base)

            # (a) tied：非第 0 层强制用第 0 层（或“在线均值”）结构
            elif use_cf and mode == "tied" and li > 0:
                if tied_use_mean and (mean_A_b_acc is not None) and mean_cnt > 0:
                    mean_A_b = (mean_A_b_acc / float(mean_cnt)).detach()
                    mean_A_base = mean_A_b.mean(dim=0)
                    if hasattr(layer, "_set_override_adjacency"):
                        layer._set_override_adjacency(A_b=mean_A_b, A_base_mean=mean_A_base)
                elif (tied_A_b is not None) and (tied_A_base is not None):
                    if hasattr(layer, "_set_override_adjacency"):
                        layer._set_override_adjacency(A_b=tied_A_b, A_base_mean=tied_A_base)

            # --- forward ---
            if li == 0:
                Y_l, reg_l, A_base_l = layer(Z_l, meta=meta, node_feat=node_feat, mask=mask, M=M)
            else:
                Y_l, reg_l, A_base_l = layer(Z_l, meta=meta, node_feat=None, mask=None, M=None)

            # (NEW) layer_kill：强内容屏蔽（彻底断开串联信息流）
            if use_cf and mode == "layer_kill" and (layer_mask is not None) and li < len(layer_mask) and int(layer_mask[li]) == 0:
                Y_l = torch.zeros_like(Y_l)
                self._patch_layer_export_after_kill(layer, Y_l)

            tower_Z.append(Y_l)
            tower_A_base.append(A_base_l)
            A_batch_l = layer.last_A_batch
            tower_A_batch.append(A_batch_l)

            # tied：保存第 0 层结构 + 维护在线均值累加器
            if use_cf and mode == "tied":
                if li == 0:
                    tied_A_b = A_batch_l.detach().to(torch.float32)
                    tied_A_base = A_base_l.detach().to(torch.float32)
                if mean_A_b_acc is None:
                    mean_A_b_acc = A_batch_l.detach().to(torch.float32).clone()
                else:
                    mean_A_b_acc = mean_A_b_acc + A_batch_l.detach().to(torch.float32)
                mean_cnt += 1

            for k in ("e2e", "ee_smooth", "event_div", "event_node"):
                if k in reg_l:
                    reg_total[k] = reg_total[k] + reg_l[k]

            layer_e2e.append(reg_l.get("e2e", zero).detach().to(torch.float32))
            layer_ee_smooth.append(reg_l.get("ee_smooth", zero).detach().to(torch.float32))
            layer_event_div.append(reg_l.get("event_div", zero).detach().to(torch.float32))
            layer_event_node.append(reg_l.get("event_node", zero).detach().to(torch.float32))

            Y32 = Y_l.to(torch.float32)
            layer_amp_mean.append(Y32.abs().mean().detach())
            layer_amp_std.append(Y32.abs().std(unbiased=False).detach())

            exp_pack = layer.get_content_export() if hasattr(layer, "get_content_export") else None
            layers_export[li] = exp_pack

            if isinstance(exp_pack, dict):
                repr_dict = exp_pack.get("repr", None)
                if isinstance(repr_dict, dict):
                    repr_layers_mean[li] = repr_dict.get("R_mean", None)
                    repr_layers_max[li] = repr_dict.get("R_max", None)
                tok_dict = exp_pack.get("tokens", None)
                if isinstance(tok_dict, dict):
                    tokens_layers_Y[li] = tok_dict.get("Y_tokens", None)
                    tokens_layers_norm[li] = tok_dict.get("Y_l2norm", None)

            # tower_hier：A^(ℓ)≈(A^(ℓ-1))^2
            if prev_A_base is not None:
                A_prev = prev_A_base.to(torch.float32)
                A_curr = A_base_l.to(torch.float32)
                A_2 = torch.matmul(A_prev, A_prev)
                A_2 = A_2 / A_2.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                diff = A_curr - A_2
                L_hier_raw = L_hier_raw + (diff * diff).mean()

            prev_A_base = A_base_l
            Z_l = Y_l

        reg_total["tower_hier"] = reg_total["tower_hier"] + self.w_hier * L_hier_raw.to(dtype_in)

        L_content = self._compute_content_hier_loss(
            tower_Z=tower_Z,
            content_feat=content_feat,
            device=Z.device,
            dtype=dtype_in,
        )
        reg_total["tower_content"] = reg_total["tower_content"] + L_content

        def _stack_or_none(lst):
            if not lst:
                return None
            return torch.stack(lst, dim=0)

        self._last_layer_stats = {
            "e2e": _stack_or_none(layer_e2e),
            "ee_smooth": _stack_or_none(layer_ee_smooth),
            "event_div": _stack_or_none(layer_event_div),
            "event_node": _stack_or_none(layer_event_node),
            "feat_amp_mean": _stack_or_none(layer_amp_mean),
            "feat_amp_std": _stack_or_none(layer_amp_std),
        }

        # -------------------- 安全 stack：支持缺失层 padding + mask --------------------
        def _stack_with_pad_2d(lst):
            if lst is None or len(lst) == 0:
                return None, None
            valid = [x is not None for x in lst]
            mask_ = torch.tensor(valid, dtype=torch.bool)
            if not any(valid):
                return None, mask_
            x0 = next((x for x in lst if x is not None), None)
            if x0 is None:
                return None, mask_
            if x0.dim() != 2:
                raise ValueError(f"repr tensor must be [B,D], got {tuple(x0.shape)}")
            B0, D0 = int(x0.size(0)), int(x0.size(1))
            out = x0.new_zeros((len(lst), B0, D0))
            for i, x in enumerate(lst):
                if x is None:
                    continue
                if x.shape != (B0, D0):
                    raise ValueError(f"layer {i} repr shape mismatch: expect {(B0,D0)} but got {tuple(x.shape)}")
                out[i] = x
            return out, mask_

        def _stack_with_pad_tokens(lst_Y, lst_norm):
            if lst_Y is None or len(lst_Y) == 0:
                return None, None, None
            valid = [x is not None for x in lst_Y]
            mask_ = torch.tensor(valid, dtype=torch.bool)
            if not any(valid):
                return None, None, mask_

            y0 = next((x for x in lst_Y if x is not None), None)
            if y0 is None or y0.dim() != 3:
                return None, None, mask_
            B0, E0, D0 = int(y0.size(0)), int(y0.size(1)), int(y0.size(2))

            Y_out = y0.new_zeros((len(lst_Y), B0, E0, D0))
            N_out = None
            n0 = next((x for x in (lst_norm or []) if x is not None), None)
            if n0 is not None:
                N_out = n0.new_zeros((len(lst_Y), B0, E0))

            for i in range(len(lst_Y)):
                yi = lst_Y[i]
                if yi is not None:
                    if yi.shape != (B0, E0, D0):
                        raise ValueError(f"layer {i} tokens shape mismatch: expect {(B0,E0,D0)} but got {tuple(yi.shape)}")
                    Y_out[i] = yi
                if N_out is not None and lst_norm is not None:
                    ni = lst_norm[i]
                    if ni is not None:
                        if ni.shape != (B0, E0):
                            raise ValueError(f"layer {i} token-norm shape mismatch: expect {(B0,E0)} but got {tuple(ni.shape)}")
                        N_out[i] = ni
            return Y_out, N_out, mask_

        R_mean_stack, R_mean_mask = _stack_with_pad_2d(repr_layers_mean)
        R_max_stack, R_max_mask = _stack_with_pad_2d(repr_layers_max)
        Y_tokens_stack, Y_l2norm_stack, Y_tokens_mask = _stack_with_pad_tokens(tokens_layers_Y, tokens_layers_norm)

        content_export = {
            "R_mean_stack": R_mean_stack,
            "R_max_stack":  R_max_stack,
            "R_mean_mask":  R_mean_mask,
            "R_max_mask":   R_max_mask,
            "Y_tokens_stack": Y_tokens_stack,
            "Y_l2norm_stack": Y_l2norm_stack,
            "Y_tokens_mask":  Y_tokens_mask,
            "R_mean_list":  repr_layers_mean,
            "R_max_list":   repr_layers_max,
            "Y_tokens_list": tokens_layers_Y,
            "Y_l2norm_list": tokens_layers_norm,
            "layers_export": layers_export,
            "layer_ids": list(range(L)),
        }

        self._last_content_export = content_export

        tower = {
            "Z_layers": tower_Z,
            "A_base_list": tower_A_base,
            "A_batch_list": tower_A_batch,
            "mu_span": self._last_mu_span,
            "mu_hf": self._last_mu_hf,
            "content_export": content_export,
        }

        return Z_l.to(dtype_in), reg_total, tower


class ProtoMemoryHypergraph(nn.Module):
    """
    Proto memory as hyperedges (prototypes).

    输出:
      Z_proto: [B,K,D]
      extra["A_np"]: [B,M,K]
      extra["proto_export"]: dict
    """

    def __init__(
        self,
        D_in: int,
        D_mem: int = None,
        num_protos: int = 16,
        dropout: float = 0.1,
        topk_per_node: int = 8,
        w_proto_div: float = 0.3,
        w_proto_usage: float = 0.3,
    ):
        super().__init__()
        if D_in <= 0:
            raise ValueError(f"ProtoMemoryHypergraph: D_in must > 0, got {D_in}")
        self.D_in = int(D_in)
        self.D_mem = int(D_mem) if D_mem is not None else int(D_in)
        self.num_protos = int(num_protos)
        if self.num_protos <= 0:
            raise ValueError(f"ProtoMemoryHypergraph: num_protos must > 0, got {num_protos}")

        self.topk_per_node = int(topk_per_node)
        self.drop = nn.Dropout(dropout)

        self.q_node = nn.Linear(self.D_in, self.D_mem, bias=False)
        self.v_node = nn.Linear(self.D_in, self.D_mem, bias=False)
        self.proto = nn.Parameter(torch.randn(self.num_protos, self.D_mem) * 0.02)

        self.w_proto_div = float(w_proto_div)
        self.w_proto_usage = float(w_proto_usage)

        self.last_A_np = None
        self.last_Z_proto = None
        self.last_mass_bk = None
        self._last_reg = None

    def forward(self, Xn: torch.Tensor, mask_valid: torch.Tensor = None, return_sparse: bool = False):
        """
        Xn: [B,M,D_in]
        mask_valid: [B,M] (optional)
        """
        # ============================================================
        # 反事实推理模式（只在 eval() 生效）
        # ============================================================
        CF_MODE = "none"            # "none" / "struct_shuffle" / "routing_uniform" / "content_shuffle"
        CF_SEED = 0
        CF_KEEP_SUPPORT = False     # routing_uniform 用
        CF_CONTENT_KIND = "match_stats"  # "match_stats" / "resample"
        # ============================================================

        if Xn.dim() != 3:
            raise ValueError(f"ProtoMemoryHypergraph: Xn must be [B,M,D_in], got {Xn.shape}")
        B, M, D = Xn.shape
        if D != self.D_in:
            raise ValueError(f"ProtoMemoryHypergraph: Xn dim={D} != D_in={self.D_in}")

        device = Xn.device
        dtype_in = Xn.dtype
        K = self.num_protos

        # --- project ---
        X32 = Xn.to(torch.float32)
        Qn = self.q_node(X32)              # [B,M,Dm]
        Vn = self.v_node(X32)              # [B,M,Dm]
        P = self.proto.to(dtype=X32.dtype) # [K,Dm]

        # --- affinity ---
        scores = torch.matmul(Qn, P.t()) / math.sqrt(max(self.D_mem, 1))  # [B,M,K]

        if mask_valid is not None:
            if mask_valid.shape != (B, M):
                raise ValueError(f"ProtoMemoryHypergraph: mask_valid must be [B,M], got {mask_valid.shape}")
            mv = mask_valid.to(device=device, dtype=torch.bool)
            scores = scores.masked_fill(~mv.unsqueeze(-1), float("-inf"))

        A_np = torch.softmax(scores, dim=-1)   # [B,M,K]
        A_np = self.drop(A_np)

        # --- topk sparsify per node ---
        if 0 < self.topk_per_node < K:
            topv, topi = torch.topk(A_np, k=self.topk_per_node, dim=-1)   # [B,M,topk]
            A_sparse = torch.zeros_like(A_np)
            A_sparse.scatter_(-1, topi, topv)
            A_np = A_sparse

        # ============================================================
        # 结构/路由反事实：只在 eval() 时生效
        # ============================================================
        if (not self.training) and (CF_MODE is not None) and (CF_MODE != "none"):
            gen = torch.Generator(device=A_np.device)
            gen.manual_seed(int(CF_SEED))

            if CF_MODE == "struct_shuffle":
                w = torch.rand((B, M, K), device=A_np.device, generator=gen)
                A_np = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            elif CF_MODE == "routing_uniform":
                if 0 < self.topk_per_node < K:
                    if CF_KEEP_SUPPORT:
                        _, topi_u = torch.topk(A_np, k=self.topk_per_node, dim=-1)
                    else:
                        r = torch.rand((B, M, K), device=A_np.device, generator=gen)
                        _, topi_u = torch.topk(r, k=self.topk_per_node, dim=-1)

                    A_new = torch.zeros_like(A_np)
                    A_new.scatter_(-1, topi_u, 1.0 / float(self.topk_per_node))
                    A_np = A_new
                else:
                    A_np = torch.full_like(A_np, 1.0 / float(K))

            elif CF_MODE == "content_shuffle":
                pass
            else:
                raise ValueError(f"ProtoMemoryHypergraph: unknown CF_MODE={CF_MODE}")

        # cache
        self.last_A_np = A_np.detach()

        # --- content aggregation ---
        Z_num = torch.einsum("bmk,bmd->bkd", A_np, Vn)  # [B,K,Dm]
        denom_bk1 = A_np.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-12)  # [B,K,1]
        Z_proto = Z_num / denom_bk1  # [B,K,Dm]

        # ============================================================
        # 内容端反事实：distribution-matched（只在 eval() 时生效）
        # ============================================================
        if (not self.training) and (CF_MODE == "content_shuffle"):
            gen = torch.Generator(device=Z_proto.device)
            gen.manual_seed(int(CF_SEED))

            if CF_CONTENT_KIND == "match_stats":
                mu = Z_proto.mean(dim=(0, 1), keepdim=True)  # [1,1,D]
                std = Z_proto.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)  # [1,1,D]
                eps = torch.randn(Z_proto.shape, device=Z_proto.device, dtype=Z_proto.dtype, generator=gen)
                Z_proto = eps * std + mu

            elif CF_CONTENT_KIND == "resample":
                BK = B * K
                flat = Z_proto.reshape(BK, Z_proto.size(-1))  # [BK, D]
                r = torch.rand((BK,), device=Z_proto.device, generator=gen)
                idx = torch.argsort(r, dim=0)  # [BK]
                flat2 = flat.index_select(0, idx)
                Z_proto = flat2.reshape(B, K, Z_proto.size(-1))

            else:
                raise ValueError(f"ProtoMemoryHypergraph: unknown CF_CONTENT_KIND={CF_CONTENT_KIND}")

        # --- cache stats ---
        mass_bk = denom_bk1.squeeze(-1)  # [B,K]
        self.last_Z_proto = Z_proto.detach()
        self.last_mass_bk = mass_bk.detach()

        # --- regularizers ---
        if self.w_proto_div > 0.0:
            P_norm = F.normalize(P, p=2, dim=-1, eps=1e-12)
            sim = P_norm @ P_norm.t()
            eye = torch.eye(K, device=device, dtype=torch.bool)
            sim_off = sim[~eye]
            raw_div = (sim_off * sim_off).mean()
            proto_div = self.w_proto_div * raw_div
        else:
            proto_div = Xn.new_zeros(())

        p_use_global = None
        if self.w_proto_usage > 0.0:
            if mask_valid is not None:
                mask_f = mask_valid.to(A_np.dtype).unsqueeze(-1)
                A_eff = A_np * mask_f
            else:
                A_eff = A_np

            mass_global = A_eff.sum(dim=(0, 1))
            total_mass = mass_global.sum().clamp_min(1e-12)
            p_use_global = mass_global / total_mass

            logp = torch.log(p_use_global + 1e-12)
            entropy = -(p_use_global * logp).sum()
            if K > 1:
                entropy = entropy / math.log(float(K))
            raw_usage = 1.0 - entropy
            proto_usage = self.w_proto_usage * raw_usage
        else:
            proto_usage = Xn.new_zeros(())

        reg = {"proto_div": proto_div.to(dtype_in), "proto_usage": proto_usage.to(dtype_in)}

        row_sum = mass_bk.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_use_b = mass_bk / row_sum

        proto_export = {
            "A_np": A_np.to(dtype_in),
            "Z_proto": Z_proto.to(dtype_in),
            "proto_param": self.proto.to(dtype_in),
            "mass_per_proto_b": mass_bk.to(dtype_in),
            "denom_b": mass_bk.to(dtype_in),
            "p_use_b": p_use_b.to(dtype_in),
            "p_use_global": (p_use_global.to(dtype_in) if p_use_global is not None else None),
        }

        extra = {"A_np": A_np.to(dtype_in), "proto_export": proto_export}

        if return_sparse:
            if B != 1:
                raise ValueError("ProtoMemoryHypergraph: return_sparse=True 当前仅支持 B==1")
            A0 = A_np[0]
            nz = (A0 > 0)
            n_idx, k_idx = nz.nonzero(as_tuple=True)
            if n_idx.numel() == 0:
                mask_ne = torch.zeros(2, 0, device=device, dtype=torch.long)
            else:
                mask_ne = torch.stack([n_idx.long(), k_idx.long()], dim=0)
            extra["mask_ne"] = mask_ne

        self._last_reg = {k: v.detach() for k, v in reg.items()}
        return Z_proto.to(dtype_in), reg, extra

    def get_proto_stats(self):
        reg = {k: v.detach().cpu() for k, v in self._last_reg.items()} if self._last_reg is not None else None
        A_np = self.last_A_np.detach().cpu() if self.last_A_np is not None else None
        Zp = self.last_Z_proto.detach().cpu() if self.last_Z_proto is not None else None
        mass = self.last_mass_bk.detach().cpu() if self.last_mass_bk is not None else None
        P = self.proto.detach().cpu()
        return {"reg": reg, "A_np": A_np, "Z_proto": Zp, "mass_bk": mass, "proto_param": P}
