# -*- coding: utf-8 -*-
"""
models/sthyper/e2e/e2e_prop.py
Version: v0.9.0  (2026-01-01)

【版本号与修改记录（changelog）】
- v0.9.0  [Major][MemoryOnly-EventTower]
  1) 完全删除 attention 事件塔（SemanticEventPropV1 / E->E self-attn 构图与传播）及其相关损失：
     - 删除 event_diversity_loss / node_reconstruct_loss / smoothness_loss / content_hier 等 attention 路径依赖
  2) 事件塔改为“纯记忆网络（ProtoMemory 风格）”的层叠 slot 金字塔：
     - 每层用原型 P[K,D] + 路由 Π[B,E,K]（softmax + row-topk）完成事件->槽聚合；
     - 输出高阶事件 token S[B,K,D]，不做 S->E 回写，不做 E->E 注意力传播；
     - 提供 slot_div / slot_usage / slot_ent 三类正则；
  3) 显式结构导出改为稀疏 topk/incidence：
     - batch: topi/topv + incidence_mask[3,nnz](b,node,event_idx; edge,slot_idx) + incidence_weight[nnz]
     - base : 对 Π 的 batch mean 做 topk 显式化（mask[2,nnz](node,edge)）
  4) 为兼容旧训练器日志，tower 仍返回 reg_total 键：
     - event_div <- slot_div
     - tower_hier <- slot_usage
     - e2e <- slot_ent（可选）
     - ee_smooth/event_node/tower_content 恒为 0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# 工具函数（Memory-only）
# ======================================================================

def _row_topk_dense(Pi: torch.Tensor, k: int) -> torch.Tensor:
    """
    对 Π 的每行（每个 node/event）做 TopK：保留支持集，其余置 0，并行归一化。
    Pi: [B, E, K]
    """
    if k is None or k <= 0:
        return Pi
    B, E, K = Pi.shape
    if k >= K:
        return Pi
    topv, topi = torch.topk(Pi, k=int(k), dim=-1)  # [B,E,k]
    out = torch.zeros_like(Pi)
    out.scatter_(-1, topi, topv)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def _slot_div_loss(P: torch.Tensor) -> torch.Tensor:
    """
    原型多样性：惩罚非对角余弦相似度平方均值。
    P: [K, D]
    """
    K = P.size(0)
    if K <= 1:
        return P.new_zeros(())
    Pn = F.normalize(P, p=2, dim=-1, eps=1e-12)
    sim = Pn @ Pn.t()
    eye = torch.eye(K, device=P.device, dtype=torch.bool)
    off = sim[~eye]
    return (off * off).mean()


def _usage_loss(Pi: torch.Tensor, eps: float = 1e-12):
    """
    槽使用均匀性：usage loss = 1 - H(p)/logK
    Pi: [B, E, K]
    返回：
      raw_usage: 标量
      entropy_norm: 标量（0~1）
    """
    B, E, K = Pi.shape
    if K <= 1:
        z = Pi.new_zeros(())
        return z, z
    mass = Pi.sum(dim=(0, 1))  # [K]
    total = mass.sum().clamp_min(eps)
    p = mass / total
    entropy = -(p * torch.log(p + eps)).sum()
    entropy_norm = entropy / math.log(float(K))
    raw_usage = 1.0 - entropy_norm
    return raw_usage, entropy_norm


def _entropy_ratio(Pi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    路由行熵均值，归一化到 [0,1]：H(row)/logK
    Pi: [B, E, K]
    """
    B, E, K = Pi.shape
    if K <= 1:
        return Pi.new_zeros(())
    P = Pi.clamp_min(eps)
    H_row = -(P * torch.log(P)).sum(dim=-1)  # [B,E]
    return H_row.mean() / math.log(float(K))


def _build_incidence_from_topk(topi: torch.Tensor, topv: torch.Tensor):
    """
    由 topk 结果构造 incidence（batch 级）：
      topi: [B, E, k]  (slot index)
      topv: [B, E, k]  (weight)
    返回：
      mask:   [3, nnz]  (b, node=e, edge=slot)
      weight: [nnz]
    """
    if topi.numel() == 0:
        device = topi.device
        return torch.zeros(3, 0, device=device, dtype=torch.long), torch.zeros(0, device=device, dtype=topv.dtype)

    B, E, k = topi.shape
    device = topi.device

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, E, k)
    n_idx = torch.arange(E, device=device).view(1, E, 1).expand(B, E, k)
    e_idx = topi

    mask = torch.stack([b_idx.reshape(-1), n_idx.reshape(-1), e_idx.reshape(-1)], dim=0).long()
    weight = topv.reshape(-1)
    return mask, weight


def _build_base_from_Pi_mean(Pi_mean: torch.Tensor, k: int):
    """
    base 显式化：对 Pi_mean[E,K] 每行 topk，返回：
      base_topi: [E,k]
      base_topv: [E,k]
      base_mask: [2, nnz] (node=e, edge=slot)
      base_w:    [nnz]
    """
    if Pi_mean.numel() == 0:
        device = Pi_mean.device
        return None, None, torch.zeros(2, 0, device=device, dtype=torch.long), torch.zeros(0, device=device, dtype=Pi_mean.dtype)

    E, K = Pi_mean.shape
    kk = int(max(1, min(int(k), K))) if (k is not None and k > 0) else K
    topv, topi = torch.topk(Pi_mean, k=kk, dim=-1)  # [E,kk]

    device = Pi_mean.device
    n_idx = torch.arange(E, device=device).view(E, 1).expand(E, kk)
    e_idx = topi

    base_mask = torch.stack([n_idx.reshape(-1), e_idx.reshape(-1)], dim=0).long()
    base_w = topv.reshape(-1)
    return topi.detach(), topv.detach(), base_mask, base_w

def _inter_layer_proto_div_loss(Pa: torch.Tensor, Pb: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    层间多样性：鼓励相邻层 prototypes 彼此不相似。
    Pa: [Ka, D], Pb: [Kb, D]
    return: scalar (float32)
    """
    A = F.normalize(Pa.to(torch.float32), p=2, dim=-1, eps=eps)  # [Ka,D]
    B = F.normalize(Pb.to(torch.float32), p=2, dim=-1, eps=eps)  # [Kb,D]
    sim = A @ B.t()                                              # [Ka,Kb]
    return (sim * sim).mean()


class MemorySlotPropV2(nn.Module):
    """
    纯记忆网络层：输入事件 Z:[B,E,D] -> 输出槽 token S:[B,K,D]

    核心流程：
      1) 路由 Π[B,E,K] = softmax( Q_route(Z) · K_route(P)^T / sqrt(D_route) / tau )
         + row-topk 稀疏化（每个事件最多连接 topk_per_event 个槽）
      2) 证据聚合 u[B,K,D] = Σ_e Π[e,k] φ(z_e) / mass_k
      3) 槽更新：S = LN( P + eta * gate(u,P) * δ([u,P,u-P,u*P]) )

    正则：
      - slot_div   ：原型多样性
      - slot_usage ：使用分布熵（均匀性）
      - slot_ent   ：路由行熵目标（可选）

    v0.7.4：
      - pack["diag"]：新增坍缩诊断（结构+内容），用于离线查看是否坍缩。
    """

    def __init__(
        self,
        configs,
        D: int,
        num_slots: int,
        topk_per_event: int = 8,
        dropout: float = 0.1,
        tau: float = 1.0,
        eta: float = 1.0,
        hidden_dim: int = 0,
        w_slot_div: float = 0.3,
        w_slot_usage: float = 0.3,
        w_slot_ent: float = 0.0,
        ent_ratio_target: float = 0.0,
    ):
        super().__init__()
        self.D = int(D)
        self.num_slots = int(num_slots)
        if self.num_slots <= 0:
            raise ValueError(f"MemorySlotPropV2: num_slots must > 0, got {self.num_slots}")

        self.D_route = int(getattr(configs, "D_route", 128))

        self.topk_per_event = int(topk_per_event)
        self.drop = nn.Dropout(float(dropout))
        self.tau = float(tau)
        self.eta = float(eta)

        self.w_slot_div = float(w_slot_div)
        self.w_slot_usage = float(w_slot_usage)
        self.w_slot_ent = float(w_slot_ent)
        self.ent_ratio_target = float(ent_ratio_target)

        # prototypes / slots（仍在 D 空间）
        self.proto = nn.Parameter(torch.randn(self.num_slots, self.D) * 0.02)

        # ===== 路由打分：使用瓶颈维度 D_route =====
        self.q_proj_route = nn.Linear(self.D, self.D_route, bias=False)
        self.p_proj_route = nn.Linear(self.D, self.D_route, bias=False)

        # evidence mapping φ（保持在 D 空间，稳定）
        self.phi = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, self.D),
            nn.GELU(),
        )

        # update networks（保持）
        if hidden_dim is None or int(hidden_dim) <= 0:
            hidden = max(64, 2 * self.D)
        else:
            hidden = int(hidden_dim)

        self.delta_mlp = nn.Sequential(
            nn.Linear(4 * self.D, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.D),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.D, max(32, self.D // 2)),
            nn.GELU(),
            nn.Linear(max(32, self.D // 2), 1),
        )
        self.out_ln = nn.LayerNorm(self.D)

        # caches (detach for analysis/export)
        self.last_pack = None

    @staticmethod
    def _offdiag_cos_mean(X: torch.Tensor) -> torch.Tensor:
        """
        X: [N,D] -> mean cosine similarity over off-diagonal entries.
        """
        if X is None or X.numel() == 0:
            return X.new_zeros(())
        N = X.size(0)
        if N <= 1:
            return X.new_zeros(())
        Xn = F.normalize(X, dim=-1, eps=1e-12)
        sim = torch.matmul(Xn, Xn.t())  # [N,N]
        off = sim - torch.eye(N, device=sim.device, dtype=sim.dtype)
        return off.sum() / float(N * (N - 1))

    def forward(self, Z: torch.Tensor):
        """
        return:
          S:   [B,K,D]
          reg: dict{'slot_div','slot_usage','slot_ent'}
          pack: dict（稀疏显式结构 + 标量统计 + diag）
        """
        if Z.dim() != 3:
            raise ValueError(f"MemorySlotPropV2: Z must be [B,E,D], got {Z.shape}")
        B, E, D = Z.shape
        if D != self.D:
            raise ValueError(f"MemorySlotPropV2: Z dim={D} != D={self.D}")

        dtype_in = Z.dtype
        Z32 = Z.to(torch.float32)

        # -------- 1) routing Π（瓶颈打分） --------
        Qr = self.q_proj_route(Z32)                         # [B,E,D_route]
        P = self.proto.to(torch.float32)                    # [K,D]
        Kr = self.p_proj_route(P)                           # [K,D_route]

        scale = math.sqrt(max(int(self.D_route), 1))
        scores = torch.matmul(Qr, Kr.t()) / scale           # [B,E,K]

        tau = max(1e-4, float(self.tau))
        Pi = torch.softmax(scores / tau, dim=-1)            # [B,E,K]
        Pi = self.drop(Pi)

        Kcnt = self.num_slots
        k_row = int(self.topk_per_event)
        if k_row > 0 and k_row < Kcnt:
            Pi = _row_topk_dense(Pi, k_row)

        # 导出 topk 视图（用于结构显式化）
        if k_row > 0 and k_row < Kcnt:
            topv, topi = torch.topk(Pi, k=k_row, dim=-1)    # [B,E,k]
        else:
            k_view = min(16, Kcnt)
            topv, topi = torch.topk(Pi, k=k_view, dim=-1)

        inc_mask, inc_w = _build_incidence_from_topk(topi.detach(), topv.detach())

        # base: mean routing (for stable visualization)
        Pi_mean = Pi.mean(dim=0).detach()                   # [E,K]
        base_k = k_row if (k_row > 0) else min(16, Kcnt)
        base_topi, base_topv, base_mask, base_w = _build_base_from_Pi_mean(Pi_mean, k=base_k)

        # -------- 2) evidence aggregation u（保持在 D 空间） --------
        H = self.phi(Z32)                                   # [B,E,D]
        num = torch.einsum("bek,bed->bkd", Pi, H)           # [B,K,D]
        denom = Pi.sum(dim=1).unsqueeze(-1).clamp_min(1e-12)  # [B,K,1]
        u = num / denom                                     # [B,K,D]

        # -------- 3) slot update（保持在 D 空间） --------
        P_b = P.unsqueeze(0).expand(B, -1, -1)              # [B,K,D]
        feat = torch.cat([u, P_b, u - P_b, u * P_b], dim=-1)
        delta = self.delta_mlp(feat)                        # [B,K,D]
        gate = torch.sigmoid(self.gate_mlp(torch.cat([u, P_b], dim=-1)))  # [B,K,1]

        eta = float(self.eta)
        S = self.out_ln(P_b + eta * gate * delta).to(dtype_in)  # [B,K,D]

        # -------- 4) regularizers --------
        # slot_div
        if self.w_slot_div > 0.0:
            raw_div = _slot_div_loss(P)
            slot_div = (self.w_slot_div * raw_div).to(dtype_in)
        else:
            slot_div = Z.new_zeros(())

        # slot_usage
        if self.w_slot_usage > 0.0:
            raw_usage, ent_usage = _usage_loss(Pi)
            slot_usage = (self.w_slot_usage * raw_usage).to(dtype_in)
        else:
            slot_usage = Z.new_zeros(())
            ent_usage = Pi.new_zeros(())

        # slot_ent (optional entropy schedule)
        ent_ratio = _entropy_ratio(Pi)  # [0,1] scalar
        if self.w_slot_ent > 0.0 and self.ent_ratio_target > 0.0:
            tgt = float(self.ent_ratio_target)
            raw_ent = (ent_ratio - tgt) ** 2
            slot_ent = (self.w_slot_ent * raw_ent).to(dtype_in)
        else:
            slot_ent = Z.new_zeros(())

        reg = {"slot_div": slot_div, "slot_usage": slot_usage, "slot_ent": slot_ent}

        # =======================
        # v0.7.4：diag（结构+内容坍缩诊断）
        # =======================
        eps = 1e-12

        # 1) event-level routing entropy / top1 prob（用 row-normalized 的 Pi）
        row_sum = Pi.sum(dim=-1).clamp_min(eps)                      # [B,E]
        Pi_norm = Pi / row_sum.unsqueeze(-1)                         # [B,E,K]
        event_ent = -(Pi_norm.clamp_min(eps) * Pi_norm.clamp_min(eps).log()).sum(dim=-1)  # [B,E]
        event_top1_prob, event_top1_idx = Pi_norm.max(dim=-1)        # [B,E]

        event_ent_mean = event_ent.mean()
        event_ent_std = event_ent.std(unbiased=False)
        event_top1_prob_mean = event_top1_prob.mean()
        event_top1_prob_std = event_top1_prob.std(unbiased=False)

        row_sum_mean = row_sum.mean()
        row_sum_std = row_sum.std(unbiased=False)

        # top1 histogram + mode ratio（直接看是否“几乎都去同一个槽”）
        flat_top1 = event_top1_idx.reshape(-1)                       # [B*E]
        top1_counts = torch.bincount(flat_top1, minlength=Kcnt).to(torch.float32)  # [K]
        top1_total = top1_counts.sum().clamp_min(1.0)
        top1_counts_ratio = top1_counts / top1_total                 # [K]
        top1_mode_ratio = top1_counts.max() / top1_total             # scalar

        # 2) slot mass distribution（更稳：每槽总质量）
        slot_mass = Pi.sum(dim=1)                                    # [B,K]
        slot_mass_sum = slot_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
        slot_mass_norm = slot_mass / slot_mass_sum                   # [B,K]

        slot_mass_norm_mean = slot_mass_norm.mean(dim=0)             # [K]
        slot_mass_norm_std = slot_mass_norm.std(dim=0, unbiased=False)

        slot_mass_ent = -(slot_mass_norm.clamp_min(eps) * slot_mass_norm.clamp_min(eps).log()).sum(dim=-1)  # [B]
        slot_mass_ent_mean = slot_mass_ent.mean()
        slot_hhi = (slot_mass_norm ** 2).sum(dim=-1)                 # [B]
        slot_hhi_mean = slot_hhi.mean()
        eff_slots_hhi_mean = (1.0 / slot_hhi.clamp_min(eps)).mean()
        eff_slots_exp_mean = torch.exp(slot_mass_ent).mean()
        slot_max_mass_ratio_mean = slot_mass_norm.max(dim=-1).values.mean()
        active_slots_mean = (slot_mass_norm > 1e-3).to(torch.float32).sum(dim=-1).mean()

        # 3) content collapse：proto / slot token cosine off-diag mean
        proto_cos_offdiag_mean = self._offdiag_cos_mean(P)           # scalar float32
        S32 = S.to(torch.float32)
        S_mean = S32.mean(dim=0)                                     # [K,D]
        slot_token_cos_offdiag_mean = self._offdiag_cos_mean(S_mean)

        # 4) update dynamics：gate / delta / token norm
        gate_mean = gate.to(torch.float32).mean()
        gate_std = gate.to(torch.float32).std(unbiased=False)
        delta_norm_mean = delta.to(torch.float32).norm(dim=-1).mean()
        token_norm_mean = S32.norm(dim=-1).mean()
        token_norm_std = S32.norm(dim=-1).std(unbiased=False)

        diag = {
            # structure
            "row_sum_mean": row_sum_mean.detach(),
            "row_sum_std": row_sum_std.detach(),
            "event_entropy_mean": event_ent_mean.detach(),
            "event_entropy_std": event_ent_std.detach(),
            "event_top1_prob_mean": event_top1_prob_mean.detach(),
            "event_top1_prob_std": event_top1_prob_std.detach(),
            "top1_mode_ratio": top1_mode_ratio.detach(),
            "top1_counts_ratio": top1_counts_ratio.detach(),                 # [K]
            "slot_mass_norm_mean": slot_mass_norm_mean.detach(),             # [K]
            "slot_mass_norm_std": slot_mass_norm_std.detach(),               # [K]
            "slot_mass_entropy_mean": slot_mass_ent_mean.detach(),
            "slot_hhi_mean": slot_hhi_mean.detach(),
            "eff_slots_hhi_mean": eff_slots_hhi_mean.detach(),
            "eff_slots_exp_mean": eff_slots_exp_mean.detach(),
            "slot_max_mass_ratio_mean": slot_max_mass_ratio_mean.detach(),
            "active_slots_mean": active_slots_mean.detach(),

            # content
            "proto_cos_offdiag_mean": proto_cos_offdiag_mean.detach(),
            "slot_token_cos_offdiag_mean": slot_token_cos_offdiag_mean.detach(),
            "token_norm_mean": token_norm_mean.detach(),
            "token_norm_std": token_norm_std.detach(),

            # dynamics
            "gate_mean": gate_mean.detach(),
            "gate_std": gate_std.detach(),
            "delta_norm_mean": delta_norm_mean.detach(),
        }

        pack = {
            "shape": (int(B), int(E), int(Kcnt)),
            "proto": self.proto.detach().to(dtype_in),
            "tau": float(self.tau),
            "eta": float(self.eta),

            # route bottleneck
            "route_dim": int(self.D_route),

            # sparse batch struct
            "topi": topi.detach().to(torch.long),
            "topv": topv.detach().to(dtype_in),
            "incidence_mask": inc_mask.detach(),                  # [3, nnz] (b,node,edge)
            "incidence_weight": inc_w.detach().to(dtype_in),

            # base struct (mean over batch)
            "Pi_mean_shape": (int(E), int(Kcnt)),
            "base_topi": None if base_topi is None else base_topi.to(torch.long),
            "base_topv": None if base_topv is None else base_topv.to(dtype_in),
            "base_incidence_mask": base_mask.detach(),            # [2, nnz] (node,edge)
            "base_incidence_weight": base_w.detach().to(dtype_in),

            # scalars for analysis（保留原字段）
            "entropy_ratio": ent_ratio.detach().to(dtype_in),
            "usage_entropy": ent_usage.detach().to(dtype_in),

            # v0.7.4 diag（统一出口）
            "diag": {k: (v.to(dtype_in) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
                     for k, v in diag.items()},
        }
        self.last_pack = pack
        return S, reg, pack


# ======================================================================
# 事件塔（Memory-only）：层叠 slot 金字塔
# ======================================================================

class SemanticEventTowerV2(nn.Module):
    """
    Memory-only 事件塔：
      输入  Z:[B,E0,D]
      输出  Z_L:[B, K_last, D]

    - 每层是 MemorySlotPropV2：事件->槽聚合，不回写、不 E->E 注意力传播
    - reg_total 精简：
        event_div  <- Σ slot_div + Σ inter_layer_proto_div
        tower_hier <- Σ slot_usage
        e2e        <- Σ slot_ent

    v0.7.4：
      - 汇总每层 pack["diag"] 到 self._last_layer_stats（标量 stack + 变长向量 list）
    """

    def __init__(
        self,
        configs,
        D: int,
        num_layers: int = 2,
        topk: int = 16,
        dropout: float = 0.1,
        w_div: float = 1,
        w_hier: float = 1,   # usage 权重（兼容旧命名）
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"SemanticEventTowerV2: num_layers must > 0, got {num_layers}")
        self.num_layers = int(num_layers)
        self.D = int(D)

        # slot schedule (base -> ratio^l)
        slot_base = int(getattr(configs, "num_event_protos", 64))
        slot_ratio = float(getattr(configs, "slot_ratio", 0.5))
        slot_min = int(getattr(configs, "slot_min", 2))

        K_list = [max(1, slot_base)]
        for _ in range(1, self.num_layers):
            nxt = int(round(K_list[-1] * slot_ratio))
            nxt = max(slot_min, nxt)
            K_list.append(nxt)
        self.slot_num_list = [int(k) for k in K_list]

        # tau/eta schedule (optional)
        self.slot_tau0 = float(getattr(configs, "slot_tau0", 1.0))
        self.slot_tau_decay = float(getattr(configs, "slot_tau_decay", 0.7))
        self.slot_eta0 = float(getattr(configs, "slot_eta0", 1.0))
        self.slot_eta_decay = float(getattr(configs, "slot_eta_decay", 1.0))

        # entropy target schedule (optional)
        self.slot_ent_ratio_hi = float(getattr(configs, "slot_ent_ratio_hi", 0.90))
        self.slot_ent_ratio_lo = float(getattr(configs, "slot_ent_ratio_lo", 0.30))

        # loss weights
        self.w_slot_div = float(getattr(configs, "w_slot_div", w_div))
        self.w_slot_usage = float(getattr(configs, "w_slot_usage", w_hier))
        self.w_slot_ent = float(getattr(configs, "w_slot_ent", 0.0))  # 默认 0

        self.slot_hidden_dim = int(getattr(configs, "slot_hidden_dim", 128))

        # ratio topk（外部已有参数 event_topk_ratio）
        self.event_topk_ratio = float(getattr(configs, "event_topk_ratio", 0.0))
        # fallback topk（旧参数）
        self.topk_per_event = int(topk)

        layers = []
        for li in range(self.num_layers):
            K_l = int(self.slot_num_list[li])
            tau_l = self.slot_tau0 * (self.slot_tau_decay ** li)
            eta_l = self.slot_eta0 * (self.slot_eta_decay ** li)

            if self.num_layers <= 1:
                ent_ratio_tgt = self.slot_ent_ratio_lo
            else:
                t = li / float(self.num_layers - 1)
                ent_ratio_tgt = (1.0 - t) * self.slot_ent_ratio_hi + t * self.slot_ent_ratio_lo

            # 每个事件最多连接多少槽：优先 ratio，否则 fallback 到旧 topk 语义
            if self.event_topk_ratio > 0.0:
                k_row = int(round(self.event_topk_ratio * K_l))
                k_row = max(1, min(k_row, K_l))
            else:
                if self.topk_per_event <= 0:
                    k_row = max(1, int(round(0.5 * K_l)))
                else:
                    k_row = min(int(self.topk_per_event), K_l)

            layers.append(
                MemorySlotPropV2(
                    configs=configs,
                    D=self.D,
                    num_slots=int(K_l),
                    topk_per_event=int(k_row),
                    dropout=float(dropout),
                    tau=float(tau_l),
                    eta=float(eta_l),
                    hidden_dim=int(self.slot_hidden_dim),
                    w_slot_div=float(self.w_slot_div),
                    w_slot_usage=float(self.w_slot_usage),
                    w_slot_ent=float(self.w_slot_ent),
                    ent_ratio_target=float(ent_ratio_tgt),
                )
            )
        self.layers = nn.ModuleList(layers)

        # caches for analysis
        self._last_layer_stats = None

    def get_layer_stats(self):
        """
        返回 CPU 上的统计：
          - 标量序列：tensor [L]
          - 变长向量：list[tensor]，每层一个（例如 top1_counts_ratio / slot_mass_norm_mean）
        """
        stats = {}
        if self._last_layer_stats is None:
            return stats

        for k, v in self._last_layer_stats.items():
            if v is None:
                stats[k] = None
            elif isinstance(v, (list, tuple)):
                out = []
                for it in v:
                    out.append(None if it is None else it.detach().cpu())
                stats[k] = out
            else:
                stats[k] = v.detach().cpu()
        return stats

    def forward(self, Z: torch.Tensor, **kwargs):
        """
        kwargs 会被忽略（兼容旧调用）
        return:
          Z_out: [B, K_last, D]
          reg_total: dict（精简键）
          tower: dict（结构导出）
        """
        zero = Z.new_zeros(())
        reg_total = {
            "event_div": zero,   # slot_div + inter_layer_div
            "tower_hier": zero,  # slot_usage
            "e2e": zero,         # slot_ent
        }

        tower_Z = []
        tower_A_base = []   # None placeholders
        tower_A_batch = []  # None placeholders

        tower_H_base = []
        tower_H_batch = []
        slot_pack_list = []

        prev_proto = None

        # ===== v0.7.4：新增 diag 汇总容器 =====
        diag_scalar = {
            "row_sum_mean": [],
            "event_entropy_mean": [],
            "event_top1_prob_mean": [],
            "top1_mode_ratio": [],
            "slot_hhi_mean": [],
            "eff_slots_hhi_mean": [],
            "slot_mass_entropy_mean": [],
            "slot_max_mass_ratio_mean": [],
            "active_slots_mean": [],
            "proto_cos_offdiag_mean": [],
            "slot_token_cos_offdiag_mean": [],
            "token_norm_mean": [],
            "gate_mean": [],
            "delta_norm_mean": [],
        }
        diag_vec = {
            "top1_counts_ratio": [],      # list of [K_l]
            "slot_mass_norm_mean": [],    # list of [K_l]
        }

        layer_scalar = {
            "slot_entropy_ratio": [],
            "slot_usage_entropy": [],
            "feat_amp_mean": [],
            "feat_amp_std": [],
            "slot_div": [],
            "slot_usage": [],
            "slot_ent": [],
            "inter_proto_div": [],
        }

        Z_l = Z
        for li, layer in enumerate(self.layers):
            S_l, reg_l, pack = layer(Z_l)   # S_l: [B,K,D]
            tower_Z.append(S_l)
            slot_pack_list.append(pack)

            H_base = {
                "shape": pack["Pi_mean_shape"],
                "topi": pack["base_topi"],
                "topv": pack["base_topv"],
                "incidence_mask": pack["base_incidence_mask"],      # [2,nnz] (node,edge)
                "incidence_weight": pack["base_incidence_weight"],
            }
            H_batch = {
                "shape": pack["shape"],
                "topi": pack["topi"],
                "topv": pack["topv"],
                "incidence_mask": pack["incidence_mask"],           # [3,nnz] (b,node,edge)
                "incidence_weight": pack["incidence_weight"],
            }
            tower_H_base.append(H_base)
            tower_H_batch.append(H_batch)

            # -------- 核心：累加需要的 loss --------
            reg_total["e2e"] = reg_total["e2e"] + reg_l["slot_ent"]
            reg_total["event_div"] = reg_total["event_div"] + reg_l["slot_div"]
            reg_total["tower_hier"] = reg_total["tower_hier"] + reg_l["slot_usage"]

            # -------- 层间多样性（相邻层 proto 去相关）--------
            if prev_proto is not None:
                raw_inter = _inter_layer_proto_div_loss(prev_proto, layer.proto)  # float32 scalar
                inter_div = (self.w_slot_div * raw_inter).to(Z_l.dtype)
            else:
                inter_div = Z_l.new_zeros(())
            reg_total["event_div"] = reg_total["event_div"] + inter_div

            # stats（保留你原来的）
            layer_scalar["slot_entropy_ratio"].append(pack["entropy_ratio"].detach().to(torch.float32))
            layer_scalar["slot_usage_entropy"].append(pack["usage_entropy"].detach().to(torch.float32))
            layer_scalar["slot_div"].append(reg_l["slot_div"].detach().to(torch.float32))
            layer_scalar["slot_usage"].append(reg_l["slot_usage"].detach().to(torch.float32))
            layer_scalar["slot_ent"].append(reg_l["slot_ent"].detach().to(torch.float32))
            layer_scalar["inter_proto_div"].append(inter_div.detach().to(torch.float32))

            S32 = S_l.to(torch.float32)
            layer_scalar["feat_amp_mean"].append(S32.abs().mean().detach())
            layer_scalar["feat_amp_std"].append(S32.abs().std(unbiased=False).detach())

            # ===== v0.7.4：收集 diag =====
            diag = pack.get("diag", {})
            for k in diag_scalar.keys():
                if k in diag and torch.is_tensor(diag[k]):
                    diag_scalar[k].append(diag[k].detach().to(torch.float32))
                else:
                    diag_scalar[k].append(Z_l.new_zeros(()).detach().to(torch.float32))

            # 变长向量：直接存 list
            if "top1_counts_ratio" in diag and torch.is_tensor(diag["top1_counts_ratio"]):
                diag_vec["top1_counts_ratio"].append(diag["top1_counts_ratio"].detach().to(torch.float32))
            else:
                diag_vec["top1_counts_ratio"].append(None)

            if "slot_mass_norm_mean" in diag and torch.is_tensor(diag["slot_mass_norm_mean"]):
                diag_vec["slot_mass_norm_mean"].append(diag["slot_mass_norm_mean"].detach().to(torch.float32))
            else:
                diag_vec["slot_mass_norm_mean"].append(None)

            tower_A_base.append(None)
            tower_A_batch.append(None)

            prev_proto = layer.proto
            Z_l = S_l

        def _stack(lst):
            return torch.stack(lst, dim=0) if lst else None

        # 汇总缓存：标量 stack；向量 list
        self._last_layer_stats = {
            # 原有
            "slot_entropy_ratio": _stack(layer_scalar["slot_entropy_ratio"]),
            "slot_usage_entropy": _stack(layer_scalar["slot_usage_entropy"]),
            "slot_div": _stack(layer_scalar["slot_div"]),
            "slot_usage": _stack(layer_scalar["slot_usage"]),
            "slot_ent": _stack(layer_scalar["slot_ent"]),
            "inter_proto_div": _stack(layer_scalar["inter_proto_div"]),
            "feat_amp_mean": _stack(layer_scalar["feat_amp_mean"]),
            "feat_amp_std": _stack(layer_scalar["feat_amp_std"]),

            # v0.7.4 diag（标量）
            "diag_row_sum_mean": _stack(diag_scalar["row_sum_mean"]),
            "diag_event_entropy_mean": _stack(diag_scalar["event_entropy_mean"]),
            "diag_event_top1_prob_mean": _stack(diag_scalar["event_top1_prob_mean"]),
            "diag_top1_mode_ratio": _stack(diag_scalar["top1_mode_ratio"]),
            "diag_slot_hhi_mean": _stack(diag_scalar["slot_hhi_mean"]),
            "diag_eff_slots_hhi_mean": _stack(diag_scalar["eff_slots_hhi_mean"]),
            "diag_slot_mass_entropy_mean": _stack(diag_scalar["slot_mass_entropy_mean"]),
            "diag_slot_max_mass_ratio_mean": _stack(diag_scalar["slot_max_mass_ratio_mean"]),
            "diag_active_slots_mean": _stack(diag_scalar["active_slots_mean"]),
            "diag_proto_cos_offdiag_mean": _stack(diag_scalar["proto_cos_offdiag_mean"]),
            "diag_slot_token_cos_offdiag_mean": _stack(diag_scalar["slot_token_cos_offdiag_mean"]),
            "diag_token_norm_mean": _stack(diag_scalar["token_norm_mean"]),
            "diag_gate_mean": _stack(diag_scalar["gate_mean"]),
            "diag_delta_norm_mean": _stack(diag_scalar["delta_norm_mean"]),

            # v0.7.4 diag（变长向量）
            "diag_top1_counts_ratio_list": diag_vec["top1_counts_ratio"],
            "diag_slot_mass_norm_mean_list": diag_vec["slot_mass_norm_mean"],
        }

        tower = {
            "Z_layers": tower_Z,
            "A_base_list": tower_A_base,   # 兼容字段
            "A_batch_list": tower_A_batch, # 兼容字段
            "H_base_list": tower_H_base,
            "H_batch_list": tower_H_batch,
            "slot_pack_list": slot_pack_list,
        }
        return Z_l, reg_total, tower


# ======================================================================
# ProtoMemoryHypergraph（保留：节点->原型事件，供事件塔输入）
# ======================================================================

class ProtoMemoryHypergraph(nn.Module):
    """
    节点 -> 原型事件（第一层事件发现器）
    v0.3.0：引入显式非线性变换（value-side phi + proto-anchored post update），降低坍缩风险。
    v0.3.2：加入内容收集功能，便于测试可视化/离线分析。
    """
    def __init__(
        self,
        configs,
        D_in: int,
        D_mem: int = None,
        num_protos: int = 16,
        dropout: float = 0.1,
        topk_per_node: int = 8,
        w_proto_div: float = 0.3,
        w_proto_usage: float = 0.3,
        # ===== v0.3.0 新增：路由瓶颈与非线性更新超参（可纳入搜索空间）=====
        D_route: int = 128,
        hidden_dim: int = 0,
        eta: float = 1.0,
    ):
        super().__init__()
        if D_in <= 0:
            raise ValueError(f"ProtoMemoryHypergraph: D_in must > 0, got {D_in}")
        self.D_in = int(D_in)
        self.D_mem = int(D_mem) if D_mem is not None else int(D_in)

        self.num_protos = int(num_protos)
        if self.num_protos <= 0:
            raise ValueError(f"ProtoMemoryHypergraph: num_protos must > 0, got {self.num_protos}")

        self.topk_per_node = int(topk_per_node)
        self.drop = nn.Dropout(float(dropout))

        # ===== 路由瓶颈维度（可搜索）=====
        self.D_route = int(D_route) if int(D_route) > 0 else int(self.D_mem)
        self.eta = float(eta)

        # -------- 路由打分（Q/P 在 D_route 空间）--------
        self.q_proj_route = nn.Linear(self.D_in, self.D_route, bias=False)
        self.p_proj_route = nn.Linear(self.D_mem, self.D_route, bias=False)

        # -------- value 端：线性映射 + 非线性证据变换 --------
        self.v_node = nn.Linear(self.D_in, self.D_mem, bias=False)
        self.v_phi = nn.Sequential(
            nn.LayerNorm(self.D_mem),
            nn.Linear(self.D_mem, self.D_mem),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # prototypes（事件原型）
        self.proto = nn.Parameter(torch.randn(self.num_protos, self.D_mem) * 0.02)

        # -------- 聚合后：原型锚定的非线性更新（轻量版 slot update）--------
        # 说明：你这里用了 configs.slot_hidden_dim，这会成为后续超参搜索候选项之一
        # （建议后续改成 proto_hidden_dim，但目前先兼容你的工程字段）
        hidden = getattr(configs, "slot_hidden_dim", 32)

        self.delta_mlp = nn.Sequential(
            nn.Linear(4 * self.D_mem, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.D_mem),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.D_mem, max(32, self.D_mem // 2)),
            nn.GELU(),
            nn.Linear(max(32, self.D_mem // 2), 1),
        )
        self.out_ln = nn.LayerNorm(self.D_mem)

        # regs
        self.w_proto_div = float(w_proto_div)
        self.w_proto_usage = float(w_proto_usage)

        # ---- caches for analysis/export ----
        self.last_A_np = None
        self.last_Z_raw = None     # NEW(v0.3.2): 聚合后的原始事件（非线性更新前）
        self.last_Z_proto = None   # NEW(v0.3.2): 最终事件 token（非线性更新后）
        self._last_reg = None

    def forward(self, Xn: torch.Tensor, mask_valid: torch.Tensor = None, return_sparse: bool = False):
        if Xn.dim() != 3:
            raise ValueError(f"ProtoMemoryHypergraph: Xn must be [B,M,D_in], got {Xn.shape}")
        B, M, D = Xn.shape
        if D != self.D_in:
            raise ValueError(f"ProtoMemoryHypergraph: Xn dim={D} != D_in={self.D_in}")

        device = Xn.device
        dtype_in = Xn.dtype

        # 用 float32 做数值更稳（尤其 softmax / LN / GELU）
        X32 = Xn.to(torch.float32)

        # ===== 1) routing scores（瓶颈空间）=====
        Qr = self.q_proj_route(X32)          # [B,M,D_route]
        P = self.proto.to(torch.float32)     # [K,D_mem]
        Pr = self.p_proj_route(P)            # [K,D_route]
        K = self.num_protos

        scale = math.sqrt(max(int(self.D_route), 1))
        scores = torch.matmul(Qr, Pr.t()) / scale  # [B,M,K]

        mv = None
        if mask_valid is not None:
            if mask_valid.shape != (B, M):
                raise ValueError(f"mask_valid must be [B,M], got {mask_valid.shape}")
            mv = mask_valid.to(device=device, dtype=torch.bool)
            # 不做 -inf masked_fill，避免 softmax(-inf) -> NaN；
            # 下面对无效行 A_np 直接置 0。

        # soft assignment
        A_np = torch.softmax(scores, dim=-1)  # [B,M,K]

        # mask invalid rows -> 0（避免 NaN/污染聚合）
        if mv is not None:
            A_np = A_np.masked_fill(~mv.unsqueeze(-1), 0.0)

        # dropout（在概率上加噪，仍会再行归一化）
        A_np = self.drop(A_np)

        # ===== 2) TopK 稀疏化 + 行归一化（保持概率语义）=====
        if 0 < self.topk_per_node < K:
            topv, topi = torch.topk(A_np, k=self.topk_per_node, dim=-1)  # [B,M,k]
            A_sparse = torch.zeros_like(A_np)
            A_sparse.scatter_(-1, topi, topv)
            A_np = A_sparse

        # 行归一化：每个节点的分配和为 1（无效节点行保持 0）
        row_sum = A_np.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B,M,1]
        A_np = A_np / row_sum
        if mv is not None:
            A_np = A_np * mv.unsqueeze(-1).to(A_np.dtype)

        # ===== v0.3.2：缓存 A_np（detach，避免计算图引用）=====
        self.last_A_np = A_np.detach()

        # ===== 3) value-side 非线性证据映射 =====
        Vn = self.v_node(X32)      # [B,M,D_mem]
        Hn = self.v_phi(Vn)        # [B,M,D_mem]  <-- 非线性在这里引入

        # ===== 4) 聚合得到原始事件（线性汇聚）=====
        Z_num = torch.einsum("bmk,bmd->bkd", A_np, Hn)  # [B,K,D_mem]
        denom = A_np.sum(dim=1).unsqueeze(-1).clamp_min(1e-12)  # [B,K,1]
        Z_raw = Z_num / denom  # [B,K,D_mem]

        # ===== v0.3.2：缓存 Z_raw（detach）=====
        self.last_Z_raw = Z_raw.detach()

        # ===== 5) 聚合后：原型锚定的非线性更新（显式非线性事件生成）=====
        P_b = P.unsqueeze(0).expand(B, -1, -1)  # [B,K,D_mem]
        feat = torch.cat([Z_raw, P_b, Z_raw - P_b, Z_raw * P_b], dim=-1)  # [B,K,4D]
        delta = self.delta_mlp(feat)  # [B,K,D]
        gate = torch.sigmoid(self.gate_mlp(torch.cat([Z_raw, P_b], dim=-1)))  # [B,K,1]

        eta = float(self.eta)
        Z_proto = self.out_ln(P_b + eta * gate * delta)  # [B,K,D_mem]

        # ===== v0.3.2：缓存 Z_proto（detach）=====
        self.last_Z_proto = Z_proto.detach()

        # ===== regs =====
        # proto_div（原型多样性）
        if self.w_proto_div > 0.0:
            P_norm = F.normalize(P, p=2, dim=-1, eps=1e-12)
            sim = P_norm @ P_norm.t()
            eye = torch.eye(K, device=device, dtype=torch.bool)
            sim_off = sim[~eye]
            raw_div = (sim_off * sim_off).mean()
            proto_div = self.w_proto_div * raw_div
        else:
            proto_div = Xn.new_zeros(()).to(torch.float32)

        # proto_usage（使用均匀性）
        if self.w_proto_usage > 0.0:
            if mv is not None:
                mask_f = mv.to(A_np.dtype).unsqueeze(-1)
                A_eff = A_np * mask_f
            else:
                A_eff = A_np
            mass = A_eff.sum(dim=(0, 1))  # [K]
            total = mass.sum().clamp_min(1e-12)
            p_use = mass / total
            entropy = -(p_use * torch.log(p_use + 1e-12)).sum()
            if K > 1:
                entropy = entropy / math.log(float(K))
            raw_usage = 1.0 - entropy
            proto_usage = self.w_proto_usage * raw_usage
        else:
            proto_usage = Xn.new_zeros(()).to(torch.float32)

        reg = {
            "proto_div": proto_div.to(dtype_in),
            "proto_usage": proto_usage.to(dtype_in),
        }

        extra = {
            "A_np": A_np.to(dtype_in),
            "Z_raw": Z_raw.to(dtype_in),   # 你现有可视化/消融很可能会用到
        }

        if return_sparse:
            if B != 1:
                raise ValueError("return_sparse=True 仅支持 B==1；多 batch 请外部循环。")
            A0 = A_np[0]  # [M,K]
            nz = (A0 > 0)
            n_idx, k_idx = nz.nonzero(as_tuple=True)
            if n_idx.numel() == 0:
                mask_ne = torch.zeros(2, 0, device=device, dtype=torch.long)
            else:
                mask_ne = torch.stack([n_idx.long(), k_idx.long()], dim=0)
            extra["mask_ne"] = mask_ne

        self._last_reg = {k: v.detach() for k, v in reg.items()}
        return Z_proto.to(dtype_in), reg, extra

    @torch.no_grad()
    def get_proto_stats(self):
        """
        用于离线分析/导出（v0.3.2）：
          - reg
          - A_np（node->proto 分配矩阵）
          - Z_raw（非线性更新前的事件 token）
          - Z_proto（非线性更新后的事件 token）
          - proto_param（可学习原型参数）
        """
        out = {
            "reg": None,
            "A_np": None,
            "Z_raw": None,
            "Z_proto": None,
            "proto_param": None,
        }

        if self._last_reg is not None:
            out["reg"] = {k: v.detach().cpu() for k, v in self._last_reg.items()}
        if self.last_A_np is not None:
            out["A_np"] = self.last_A_np.detach().cpu()
        if self.last_Z_raw is not None:
            out["Z_raw"] = self.last_Z_raw.detach().cpu()
        if self.last_Z_proto is not None:
            out["Z_proto"] = self.last_Z_proto.detach().cpu()
        if self.proto is not None:
            out["proto_param"] = self.proto.detach().cpu()

        return out
