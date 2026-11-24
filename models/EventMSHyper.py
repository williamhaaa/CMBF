# -*- coding: utf-8 -*-
"""
models/EventMSHyper.py
Version: v0.8.3  (2026-01-03)

【版本号与修改记录（changelog）】
- v0.8.3 [Add][HGStore-ExportPipeline-MemoryTower]
  1) 在主模型 Model 中加入统一导出函数 _append_hg_store（eval/test）：
     - 兼容 Memory-only 事件塔：优先读取 tower_time['H_base_list'/'H_batch_list'/'slot_pack_list']，
       直接写入 hg_store['tower'] 与 hg_store['tower_base']；
     - 将每层 slot_pack_list[l]['diag']（坍缩诊断指标）写入 hg_store['tower'][...]['layers'][l]['diag']；
     - 可选接收 tower_time['_layer_stats'] 写入 hg_store['tower_meta']，用于快速可视化。
  2) __init__ 初始化 self._hg_store=None，并提供 get_hg_store/clear_hg_store 访问接口。
  3) forward 在 eval/test 末尾调用 _append_hg_store，收集：
     - ProtoMemory(A_np) incidence
     - Tower(H_base/H_batch) incidence + diag
     - time_hg(多尺度 HGConv 前/后 token)
     - branches(y_lin/y_nodes/y_event + 若干 token)
     - tower_tokens(raw/proj)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 构图/卷积模块（与你工程保持一致） =====
from .Layers import (
    Bottleneck_Construct,
    Conv_Construct,
    MaxPooling_Construct,
    AvgPooling_Construct,
)

CONSTRUCTORS = {
    'Bottleneck_Construct': Bottleneck_Construct,
    'Conv_Construct':       Conv_Construct,
    'MaxPooling_Construct': MaxPooling_Construct,
    'AvgPooling_Construct': AvgPooling_Construct,
}

from .core.utils import (
    _as_int, _as_float, get_mask,
)
from .ops.hypergraph_conv import HypergraphConv
from .structure.time_builder import multi_adaptive_hypergraoh_prob

# 事件塔：ProtoMemory + SemanticEventTowerV2（Memory-only 路线B）
from .e2e.e2e_propV1 import ProtoMemoryHypergraph, SemanticEventTowerV2


@torch.no_grad()
def _build_segment_ring_mask(M: int, E: int, min_card: int = 2,
                             device=None, dtype=torch.long):
    """
    兜底构图：把 M 个节点按 E 份近似均分，每条超边至少包含 min_card 个成员；
    若局部成员不足则采用环式补齐（wrap-around）。
    返回：mask: LongTensor[2, nnz]
    """
    device = device or torch.device('cpu')
    E = max(int(E), 0)
    if M <= 0 or E <= 0:
        return torch.zeros(2, 0, dtype=dtype, device=device)

    n_per_edge = max(M // E, 1)
    src_list, dst_list = [], []
    for e in range(E):
        start = (e * n_per_edge) % M
        end = ((e + 1) * n_per_edge) % M
        if end > start:
            nodes = list(range(start, end))
        else:
            nodes = list(range(start, M)) + list(range(0, end))

        while len(nodes) < min_card:
            nodes.append((nodes[-1] + 1) % M if nodes else (e % M))

        nodes = list(dict.fromkeys(nodes))[:max(min_card, len(nodes))]
        src_list.extend(nodes)
        dst_list.extend([e] * len(nodes))

    mask = torch.stack([
        torch.tensor(src_list, dtype=dtype, device=device),
        torch.tensor(dst_list, dtype=dtype, device=device)
    ], dim=0)
    return mask


class Model(nn.Module):

    class _Grouping:
        @staticmethod
        def penalty(X_nodes: torch.Tensor,
                    E_edges: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
            """
            简单的 Grouping penalty：约束“同一高阶事件”内的节点不要差太多。
            （当前版本事件分支为 ProtoMemory + SemanticTower，默认不使用该项，可保留兼容。）
            """
            if mask is None or mask.numel() == 0 or \
               E_edges is None or E_edges.size(1) == 0:
                return X_nodes.new_zeros(())
            src = mask[0].long()
            dst = mask[1].long()
            member = X_nodes.index_select(1, src)         # [B, P, C]
            center_expand = E_edges.index_select(1, dst)  # [B, P, C]
            return torch.mean(torch.abs(member - center_expand))

    @staticmethod
    def _edge_centers_batched(X: torch.Tensor,
                             mask: torch.Tensor,
                             M: int,
                             E: int) -> torch.Tensor:
        """
        根据节点特征 X 和稀疏超图 mask 计算每条超边的中心特征（简单平均）。

        X    : [B, M, C]
        mask : [2, P]    (node_idx, edge_idx)
        返回 : [B, E, C]
        """
        if X is None:
            raise RuntimeError("[edge_centers_batched] X is None")

        device = X.device
        B, M_x, C = X.shape

        if M_x != M:
            raise RuntimeError(
                f"[edge_centers_batched] M={M} 与 X.size(1)={M_x} 不一致，"
                f"请检查 time-branch 里传入的 M_i 是否等于 Xi.size(1)。"
            )

        if mask is None:
            raise RuntimeError(f"[edge_centers_batched] mask is None (M={M}, E={E})")

        if mask.numel() == 0:
            raise RuntimeError(
                f"[edge_centers_batched] mask 为空 (M={M}, E={E}, mask_shape={tuple(mask.shape)})"
            )

        if mask.dim() != 2 or mask.size(0) != 2:
            raise RuntimeError(
                f"[edge_centers_batched] mask 形状非法，期望 [2, P]，实际 {tuple(mask.shape)}"
            )

        if E <= 0:
            raise RuntimeError(
                f"[edge_centers_batched] E<=0 (E={E})，请检查 hyper_num1 / AHL 输出。"
            )

        mask_cpu = mask.detach().to(device="cpu", dtype=torch.long)
        n_idx_cpu = mask_cpu[0]
        e_idx_cpu = mask_cpu[1]

        n_min = int(n_idx_cpu.min().item())
        n_max = int(n_idx_cpu.max().item())
        e_min = int(e_idx_cpu.min().item())
        e_max = int(e_idx_cpu.max().item())

        bad_n = (n_idx_cpu < 0) | (n_idx_cpu >= M)
        bad_e = (e_idx_cpu < 0) | (e_idx_cpu >= E)

        if bool(bad_n.any()) or bool(bad_e.any()):
            bad_pairs = []
            bad_any = bad_n | bad_e
            bad_pos = torch.nonzero(bad_any, as_tuple=False).view(-1)

            max_show = min(int(bad_pos.numel()), 10)
            for i in range(max_show):
                pos = int(bad_pos[i].item())
                n_val = int(n_idx_cpu[pos].item())
                e_val = int(e_idx_cpu[pos].item())
                bad_pairs.append((pos, n_val, e_val))

            raise RuntimeError(
                "[edge_centers_batched] 检测到越界索引，已在 CPU 侧停止计算以避免 CUDA index out of bounds：\n"
                f"  - M={M}, E={E}\n"
                f"  - 原始 node_idx 范围 = [{n_min}, {n_max}]\n"
                f"  - 原始 edge_idx 范围 = [{e_min}, {e_max}]\n"
                f"  - bad_n_count = {int(bad_n.sum().item())}, bad_e_count = {int(bad_e.sum().item())}\n"
                f"  - 前 {max_show} 个越界样本 (pos, node_idx, edge_idx) = {bad_pairs}\n"
                "请检查：当前尺度使用的节点数 M / 超边数 E 与 AHL 返回的 mask 是否对齐。"
            )

        n_idx = n_idx_cpu.to(device=device, dtype=torch.long)
        e_idx = e_idx_cpu.to(device=device, dtype=torch.long)

        edge_centers = X.new_zeros(B, E, C)

        for b in range(B):
            x_b = X[b]  # [M, C]
            x_sel = x_b.index_select(0, n_idx)  # [P, C]

            agg = x_b.new_zeros(E, C)
            agg.index_add_(0, e_idx, x_sel)

            cnt = torch.bincount(e_idx, minlength=E).clamp(min=1).to(agg.dtype).unsqueeze(-1)
            edge_centers[b] = agg / cnt

        return edge_centers

    @staticmethod
    def _sanitize_time_scale(
        M_feat: int,
        E_conf: int,
        raw_info,
        min_edge_card: int = 2,
        use_ring_fallback: bool = False,
        scale_idx: int = -1,
        device=None,
    ):
        """
        清洗 AHL 单尺度超图，统一输出合法 (M_i, E_eff, mask_clean)
        """
        import torch

        if device is None:
            if isinstance(raw_info, dict):
                m_ = raw_info.get('mask', None)
                if isinstance(m_, torch.Tensor):
                    device = m_.device
            if device is None:
                device = torch.device('cpu')

        M_i = int(M_feat)
        E_i_conf = int(E_conf) if E_conf is not None else 0

        if M_i <= 0:
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] M_feat={M_i} <= 0 —— 当前尺度没有节点。")

        mask = None
        prob_val = None

        if isinstance(raw_info, dict):
            M_raw = int(raw_info.get('M', M_i))
            E_raw = int(raw_info.get('E', E_i_conf))
            mask = raw_info.get('mask', None)
            prob_val = raw_info.get('prob_val', None)
            if M_raw != M_i:
                raise RuntimeError(
                    f"[TimeBranch][Scale {scale_idx}] AHL 输出的 M={M_raw} 与特征 M_feat={M_i} 不一致。"
                )
            E_i = E_raw if E_raw > 0 else E_i_conf
        elif isinstance(raw_info, torch.Tensor):
            mask = raw_info
            E_i = E_i_conf
        else:
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] raw_info 类型非法：{type(raw_info)}")

        if (mask is None) or (not isinstance(mask, torch.Tensor)):
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回的 mask 为空或非法：M_feat={M_i}, E_conf={E_conf}。")

        mask = mask.to(device=device, dtype=torch.long)
        if mask.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回空超图：M_feat={M_i}, E_conf={E_conf}。")

        if mask.dim() != 2 or mask.size(0) != 2:
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回的 mask 形状非法：{tuple(mask.shape)}")

        node_idx = mask[0]
        edge_idx = mask[1]

        if edge_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL mask 中无任何超边索引。")

        max_edge_id = int(edge_idx.max().item())
        E_i = max(int(E_i_conf), max_edge_id + 1, 1)

        valid = (node_idx >= 0) & (node_idx < M_i) & (edge_idx >= 0) & (edge_idx < E_i)
        if not bool(valid.any()):
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            n_min, n_max = int(node_idx.min().item()), int(node_idx.max().item())
            e_min, e_max = int(edge_idx.min().item()), int(edge_idx.max().item())
            raise RuntimeError(
                f"[TimeBranch][Scale {scale_idx}] AHL 输出全部 membership 越界："
                f"M_feat={M_i}, E_conf={E_conf}, E_i={E_i}, "
                f"node_idx=[{n_min},{n_max}], edge_idx=[{e_min},{e_max}]。"
            )

        node_idx = node_idx[valid]
        edge_idx = edge_idx[valid]

        if node_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 过滤越界后 mask 为空。")

        E_raw = max(int(edge_idx.max().item()) + 1, 1)
        deg = torch.bincount(edge_idx, minlength=E_raw)
        min_edge_card = max(1, int(min_edge_card))
        keep_edges = deg >= min_edge_card

        if not bool(keep_edges.any()):
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 所有超边节点数都 < min_edge_card={min_edge_card}。")

        keep_flags = keep_edges[edge_idx]
        node_idx = node_idx[keep_flags]
        edge_idx = edge_idx[keep_flags]

        if node_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] min_edge_card 过滤后 membership 为空。")

        uniq_edges, new_eidx = torch.unique(edge_idx, return_inverse=True)
        E_eff = int(uniq_edges.numel())
        if E_eff <= 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 清洗后 E_eff={E_eff}。")

        mask_clean = torch.stack([node_idx, new_eidx.to(device=device, dtype=torch.long)], dim=0).contiguous()
        if mask_clean.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 清洗后 mask_clean 为空。")

        return {'M': M_i, 'E': E_eff, 'mask': mask_clean, 'prob_val': prob_val}

    def _hyperconv_with_local_refine(
        self,
        conv,
        Xi: torch.Tensor,
        base_mask: torch.Tensor,
        M_i: int,
        E_i: int,
        scale_idx: int,
    ):
        """
        每样本局部裁剪边集合后再 HGConv（保持 v0.6.6 逻辑不变）
        """
        device = Xi.device
        B, M_feat, C = Xi.shape
        if M_feat != M_i:
            raise RuntimeError(f"[LocalHG][Scale {scale_idx}] M_feat={M_feat} 与 M_i={M_i} 不一致。")

        if base_mask is None or base_mask.numel() == 0:
            return Xi, Xi.new_zeros(())

        base_mask = base_mask.to(device=device, dtype=torch.long)
        edge_centers = Model._edge_centers_batched(Xi, base_mask, M_i, E_i)  # [B,E_i,C]

        if self.edge_refiners is not None and scale_idx < len(self.edge_refiners):
            refiner = self.edge_refiners[scale_idx]
            gate_raw = refiner(edge_centers)            # [B,E_i,1]
            gate = torch.sigmoid(gate_raw).squeeze(-1)  # [B,E_i]
        else:
            gate = Xi.new_ones(B, E_i)

        if hasattr(self, "_last_edge_gates"):
            if len(self._last_edge_gates) <= scale_idx:
                self._last_edge_gates.extend([None] * (scale_idx + 1 - len(self._last_edge_gates)))
            self._last_edge_gates[scale_idx] = gate.detach().cpu()

        Xi_conv_list = []
        loss_total = Xi.new_zeros(())

        node_idx_global = base_mask[0]
        edge_idx_global = base_mask[1]

        for b in range(B):
            x_b = Xi[b:b + 1]
            g_b = gate[b]

            if self.local_refine_topk > 0:
                k = min(self.local_refine_topk, E_i)
                _, idx_edges = torch.topk(g_b, k=k, dim=0)
                active_edges = idx_edges
            elif self.local_refine_threshold > 0.0:
                active_edges = torch.nonzero(g_b > self.local_refine_threshold, as_tuple=False).view(-1)
                if active_edges.numel() == 0:
                    _, idx_edges = torch.topk(g_b, k=1, dim=0)
                    active_edges = idx_edges
            else:
                active_edges = torch.arange(E_i, device=device)

            if active_edges.numel() == E_i:
                m_b = base_mask
            else:
                keep_edge_flags = torch.zeros(E_i, dtype=torch.bool, device=device)
                keep_edge_flags[active_edges] = True
                keep_nnz_flags = keep_edge_flags[edge_idx_global]
                if not bool(keep_nnz_flags.any()):
                    m_b = base_mask[:, :0]
                else:
                    m_b = torch.stack([node_idx_global[keep_nnz_flags], edge_idx_global[keep_nnz_flags]], dim=0)

            if m_b.numel() == 0:
                x_conv_b = x_b
                loss_b = Xi.new_zeros(())
            else:
                x_conv_b, loss_b = conv(x_b, m_b)

            Xi_conv_list.append(x_conv_b)
            loss_total = loss_total + loss_b

        Xi_conv = torch.cat(Xi_conv_list, dim=0)
        return Xi_conv, loss_total

    def _build_event_content_feat(self, Xn_time_union: torch.Tensor, A_np: torch.Tensor) -> torch.Tensor:
        """
        span/hf 内容特征：
        - 路线A（attention tower）用于 use_content_reg
        - 路线B（memory-only）当前不使用该项，但可用于导出分析（你需要看“事件内容”时很有用）
        """
        if Xn_time_union is None or A_np is None:
            return None
        if Xn_time_union.dim() != 3 or A_np.dim() != 3:
            return None

        B, M_union, C = Xn_time_union.shape
        B_a, M_a, K = A_np.shape
        if B_a != B or M_a != M_union or K <= 0 or M_union <= 0:
            return None

        device = Xn_time_union.device
        dtype = Xn_time_union.dtype

        if M_union == 1:
            pos = torch.zeros(1, device=device, dtype=dtype)
        else:
            pos = torch.linspace(0.0, 1.0, steps=M_union, device=device, dtype=dtype)
        pos2 = pos * pos

        if M_union > 1:
            diff = (Xn_time_union[:, 1:, :] - Xn_time_union[:, :-1, :]).pow(2).mean(dim=-1)  # [B,M-1]
            diff = torch.cat([diff, diff.new_zeros(B, 1)], dim=1)  # [B,M]
        else:
            diff = Xn_time_union.new_zeros(B, M_union)

        A_np32 = A_np.to(dtype)
        denom = A_np32.sum(dim=1, keepdim=True).clamp_min(1e-12)  # [B,1,K]
        w = A_np32 / denom  # p(j|k)  [B,M,K]

        E_t = torch.einsum("bmk,m->bk", w, pos)
        E_t2 = torch.einsum("bmk,m->bk", w, pos2)
        var_t = (E_t2 - E_t.pow(2)).clamp_min(0.0)
        span = torch.sqrt(var_t + 1e-12)

        hf_raw = torch.einsum("bmk,bm->bk", w, diff)
        mean_hf = hf_raw.mean(dim=1, keepdim=True).clamp_min(1e-6)
        hf = hf_raw / mean_hf

        return torch.stack([span, hf], dim=-1)  # [B,K,2]

    @staticmethod
    def _approx_span_by_length(M: int, L: int, device):
        """
        给长度为 M 的尺度节点序列，构造其在原始 lookback 长度 L 上的近似覆盖区间：
          node j -> [ floor(j*L/M), floor((j+1)*L/M)-1 ]
        返回：
          t_start: LongTensor[M]
          t_end  : LongTensor[M]
        """
        import torch
        M = int(M)
        L = int(L)
        if M <= 0 or L <= 0:
            z = torch.zeros(0, dtype=torch.long, device=device)
            return z, z

        idx = torch.arange(M, device=device, dtype=torch.long)
        t_start = (idx * L) // max(M, 1)
        t_end = ((idx + 1) * L) // max(M, 1) - 1

        t_start = t_start.clamp(min=0, max=L - 1)
        t_end = t_end.clamp(min=0, max=L - 1)
        t_end = torch.maximum(t_end, t_start)  # 保证 end>=start
        return t_start, t_end

    def _viz_reset(self):
        self._viz_H0 = None
        self._viz_H_layers = []
        self._viz_enabled_this_forward = False
        self._viz_time_stats = []
        self._viz_spatial_stats = []
        self._last_edge_gates = []

    # ============================================================
    #  新增：hg_store 访问接口（不引入任何新 flag）
    # ============================================================
    @torch.no_grad()
    def get_hg_store(self, clear: bool = False):
        hs = getattr(self, "_hg_store", None)
        if clear:
            self._hg_store = None
        return hs

    @torch.no_grad()
    def clear_hg_store(self):
        self._hg_store = None

    # ============================================================
    #  新增/替换：统一导出函数（eval/test only）
    # ============================================================
    # -*- coding: utf-8 -*-
    """
    models/EventMSHyper.py
    Version: v0.8.4  (2026-01-03)

    【版本号与修改记录（changelog）】
    - v0.8.4 [Add][Viz-RaincloudReady]
      1) ProtoMemory(A_np) 导出新增 proto 侧“分配诊断 diag”，用于箱线图/云雨图：
         - proto_mass_norm_mean/std[K]、proto_top1_counts_ratio[K]、proto_top1_mode_ratio
         - node_entropy_mean/std、node_top1_prob_mean/std
         - proto_mass_entropy_mean、proto_hhi_mean、eff_protos_hhi_mean/exp_mean、proto_max_mass_ratio_mean、active_protos_mean
         - 可选：proto_param_cos_offdiag_mean（若 proto_time 存在 proto/protos 参数）
         - 同时附带 proto_mass_norm[B,K] 与 node_top1_idx[B,M]（用于更细粒度可视化）
      2) Memory-only Tower（H_batch_list）每层导出新增“上一层→当前层”的 top1 转移诊断 transition：
         - transition_top1_hist[E_prev,K]（E_prev=上一层 slot 数；每行是该 prev-slot 在 batch 内映射到各 curr-slot 的频率）
         - transition_row_mode_ratio[E_prev] / transition_row_entropy[E_prev]（用于箱线/云雨）
      3) 不新增任何 flags；仅增强 eval/test 导出的 hg_store 内容。
    """

    @torch.no_grad()
    def _append_hg_store(
            self,
            A_np,
            tower_time,
            content_feat_time=None,
            branch_pack=None,
            time_hg_pack=None,
            tower_token_pack=None,
    ):
        """
        models/EventMSHyper.py
        Version: v0.8.6  (2026-01-04)

        【changelog】
        - v0.8.6 [Add][MS-Node-Similarity-PreInput]
          1) 在不新增任何 flags 的前提下，在 hg_store['time_hg'] 内新增多尺度节点“进入前(pre)”的相似度统计：
             - 每尺度：pre/post 的 node-token 余弦相似度 off-diagonal mean/std + 直方图（可用于箱线/云雨）
             - 跨尺度：scale-level pooled mean token 的 cosine matrix（pre/post）
             - union(pre)：把所有尺度 pre token 拼接后计算整体 off-diagonal 相似度（用于判断“原始就很相似”的情况）
          2) 计算默认在 CPU float32 上做，且对 token 做 deterministic subsample，避免显存/内存膨胀。
          3) 只改动 _append_hg_store 本函数，其余训练/forward 逻辑不动。
        """
        import torch
        import torch.nn.functional as F

        if self.training:
            return

        required_list_keys = [
            "proto",
            "tower",
            "tower_repr",
            "content",
            "branches",
            "time_hg",
            "tower_tokens",
            "tower_meta",
        ]

        if self._hg_store is None or (not isinstance(self._hg_store, dict)):
            self._hg_store = {
                "proto": [],
                "tower": [],
                "tower_base": None,
                "tower_repr": [],
                "content": [],
                "branches": [],
                "time_hg": [],
                "tower_tokens": [],
                "tower_meta": [],
            }
        else:
            for k in required_list_keys:
                if k not in self._hg_store or (not isinstance(self._hg_store[k], list)):
                    self._hg_store[k] = []
            if "tower_base" not in self._hg_store:
                self._hg_store["tower_base"] = None
            else:
                tb = self._hg_store["tower_base"]
                if tb is not None and (not isinstance(tb, list)):
                    self._hg_store["tower_base"] = None

        # 对齐 list 长度（保持你原逻辑）
        max_len = 0
        for k in required_list_keys:
            max_len = max(max_len, len(self._hg_store.get(k, [])))
        for k in required_list_keys:
            lst = self._hg_store[k]
            if len(lst) < max_len:
                lst.extend([None] * (max_len - len(lst)))

        def _cpu(x, dtype=None):
            if x is None or (not torch.is_tensor(x)):
                return None
            y = x.detach()
            if dtype is not None:
                y = y.to(dtype)
            return y.cpu().clone()

        def _to_layer_list(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return list(v)
            if torch.is_tensor(v):
                if v.dim() == 3:
                    return [v[i] for i in range(int(v.size(0)))]
                if v.dim() == 2:
                    return [v]
            return None

        def _pick_ce(ce: dict, keys):
            if not isinstance(ce, dict):
                return None
            for k in keys:
                if k in ce:
                    v = ce.get(k, None)
                    if v is not None:
                        return v
            return None

        def _subsample_tokens(X: torch.Tensor, max_tokens: int = 256):
            if X is None or (not torch.is_tensor(X)) or X.numel() == 0 or X.dim() != 3:
                return None, None
            B, T, D = X.shape
            if T <= max_tokens:
                idx = torch.arange(T, device=X.device)
                return X, idx
            idx = torch.linspace(0, T - 1, steps=max_tokens, device=X.device).long()
            Xs = X.index_select(1, idx)
            return Xs, idx

        # ---------- small math utils for diag ----------
        eps = 1e-12

        def _entropy(p: torch.Tensor, dim: int = -1):
            p = p.clamp_min(eps)
            return -(p * p.log()).sum(dim=dim)

        def _offdiag_cos_mean(X: torch.Tensor) -> torch.Tensor:
            if X is None or (not torch.is_tensor(X)) or X.numel() == 0:
                return torch.zeros((), dtype=torch.float32)
            if X.dim() != 2:
                X = X.reshape(X.size(0), -1)
            N = int(X.size(0))
            if N <= 1:
                return torch.zeros((), dtype=torch.float32)
            Xn = F.normalize(X.to(torch.float32), dim=-1, eps=1e-12)
            sim = torch.matmul(Xn, Xn.t())  # [N,N]
            off = sim - torch.eye(N, device=sim.device, dtype=sim.dtype)
            return off.sum() / float(N * (N - 1))

        def _mass_diag_from_mass_norm(mass_norm: torch.Tensor):
            """
            mass_norm: [B,K] already normalized
            return dict scalars + vectors
            """
            B, K = mass_norm.shape
            mass_norm = mass_norm.to(torch.float32)

            mean = mass_norm.mean(dim=0)  # [K]
            std = mass_norm.std(dim=0, unbiased=False)  # [K]
            ent = _entropy(mass_norm, dim=-1)  # [B]
            ent_mean = ent.mean()
            hhi = (mass_norm ** 2).sum(dim=-1)  # [B]
            hhi_mean = hhi.mean()
            eff_hhi_mean = (1.0 / hhi.clamp_min(eps)).mean()
            eff_exp_mean = torch.exp(ent).mean()
            max_ratio_mean = mass_norm.max(dim=-1).values.mean()
            active_mean = (mass_norm > 1e-3).to(torch.float32).sum(dim=-1).mean()

            return {
                "mass_norm_mean": mean,
                "mass_norm_std": std,
                "mass_entropy_mean": ent_mean,
                "hhi_mean": hhi_mean,
                "eff_hhi_mean": eff_hhi_mean,
                "eff_exp_mean": eff_exp_mean,
                "max_mass_ratio_mean": max_ratio_mean,
                "max_ratio_mean": max_ratio_mean,
                "active_mean": active_mean,
            }

        def _top1_hist_from_top1_idx(top1_idx: torch.Tensor, K: int):
            """
            top1_idx: [N] long
            return ratio [K] and mode_ratio
            """
            top1_idx = top1_idx.to(torch.long).clamp_min(0).clamp_max(int(K) - 1)
            counts = torch.bincount(top1_idx, minlength=int(K)).to(torch.float32)
            total = counts.sum().clamp_min(1.0)
            ratio = counts / total
            mode_ratio = counts.max() / total
            return ratio, mode_ratio

        # ============================================================
        # NEW: 多尺度节点相似度（pre/post）统计工具
        # ============================================================
        def _cos_offdiag_stats_batched(X_bnd_cpu32: torch.Tensor, hist_bins: int = 41):
            """
            X_bnd_cpu32: [B,N,D] on CPU float32
            返回 dict: cos_offdiag_mean/std + hist_ratio[hist_bins]
            """
            if X_bnd_cpu32 is None or (not torch.is_tensor(X_bnd_cpu32)) or X_bnd_cpu32.numel() == 0:
                return None
            if X_bnd_cpu32.dim() != 3:
                return None
            Bc, Nc, Dc = X_bnd_cpu32.shape
            if Nc <= 1:
                return {
                    "num_nodes_sampled": int(Nc),
                    "cos_offdiag_mean": float("nan"),
                    "cos_offdiag_std": float("nan"),
                    "cos_hist_ratio": torch.zeros(hist_bins, dtype=torch.float32),
                    "hist_bins": int(hist_bins),
                    "hist_min": -1.0,
                    "hist_max": 1.0,
                }

            Xn = F.normalize(X_bnd_cpu32, dim=-1, eps=1e-12)  # [B,N,D]
            sim = torch.matmul(Xn, Xn.transpose(1, 2))  # [B,N,N]

            eye = torch.eye(Nc, dtype=torch.bool, device=sim.device).unsqueeze(0)  # [1,N,N]
            off = sim.masked_select(~eye).view(Bc, Nc * (Nc - 1))  # [B, N*(N-1)]

            # per-sample stats，再做 batch mean（避免 batch size 变化影响）
            mean_b = off.mean(dim=-1)  # [B]
            std_b = off.std(dim=-1, unbiased=False)  # [B]

            off_all = off.reshape(-1)
            # 直方图（ratio）
            hist = torch.histc(off_all, bins=hist_bins, min=-1.0, max=1.0)  # [bins]
            hist = hist / hist.sum().clamp_min(1.0)

            return {
                "num_nodes_sampled": int(Nc),
                "cos_offdiag_mean": float(mean_b.mean().item()),
                "cos_offdiag_std": float(std_b.mean().item()),
                "cos_hist_ratio": hist.to(torch.float32),
                "hist_bins": int(hist_bins),
                "hist_min": -1.0,
                "hist_max": 1.0,
            }

        def _scale_pool_cos_matrix(scale_pool_list_cpu32):
            """
            scale_pool_list_cpu32: list of [B,D] CPU float32
            return [S,S] CPU float32 cosine matrix averaged over batch
            """
            if not isinstance(scale_pool_list_cpu32, (list, tuple)) or len(scale_pool_list_cpu32) == 0:
                return None
            mats = []
            for v in scale_pool_list_cpu32:
                if v is None or (not torch.is_tensor(v)) or v.numel() == 0 or v.dim() != 2:
                    return None
                mats.append(v)
            Z = torch.stack(mats, dim=1)  # [B,S,D]
            Z = F.normalize(Z, dim=-1, eps=1e-12)
            A = torch.matmul(Z, Z.transpose(1, 2))  # [B,S,S]
            return A.mean(dim=0).to(torch.float32)  # [S,S]

        # ============================================================
        # 0) 取 ProtoMemory 的 stats（保持你原逻辑）
        # ============================================================
        ds = int(getattr(self, "viz_downsample", 1) or 1)
        ds = max(1, ds)

        proto_stats = None
        if hasattr(self, "proto_time") and hasattr(self.proto_time, "get_proto_stats"):
            try:
                proto_stats = self.proto_time.get_proto_stats()
            except Exception:
                proto_stats = None

        Z_proto_live = None
        if hasattr(self, "proto_time") and hasattr(self.proto_time, "last_Z_proto"):
            v = getattr(self.proto_time, "last_Z_proto", None)
            if torch.is_tensor(v) and v.numel() > 0:
                Z_proto_live = v  # [B,K,D] on device

        if Z_proto_live is None and isinstance(proto_stats, dict):
            v = proto_stats.get("Z_proto", None)
            if torch.is_tensor(v) and v.numel() > 0:
                Z_proto_live = v  # CPU fallback

        proto_param = None
        if hasattr(self, "proto_time") and hasattr(self.proto_time, "proto"):
            v = getattr(self.proto_time, "proto", None)
            if torch.is_tensor(v) and v.numel() > 0:
                proto_param = v
        if proto_param is None and isinstance(proto_stats, dict):
            v = proto_stats.get("proto_param", None)
            if torch.is_tensor(v) and v.numel() > 0:
                proto_param = v

        proto_reg = None
        if isinstance(proto_stats, dict):
            r = proto_stats.get("reg", None)
            if isinstance(r, dict) and len(r) > 0:
                proto_reg = {}
                for kk, vv in r.items():
                    proto_reg[kk] = _cpu(vv, torch.float32) if torch.is_tensor(vv) else vv

        # ============================================================
        # 1) ProtoMemory (A_np) —— 结构 + 内容（保持你原逻辑）
        # ============================================================
        A_in = A_np
        if A_in is None and hasattr(self, "proto_time") and hasattr(self.proto_time, "last_A_np"):
            v = getattr(self.proto_time, "last_A_np", None)
            if torch.is_tensor(v) and v.numel() > 0:
                A_in = v
        if A_in is None and isinstance(proto_stats, dict):
            v = proto_stats.get("A_np", None)
            if torch.is_tensor(v) and v.numel() > 0:
                A_in = v

        if A_in is not None and torch.is_tensor(A_in) and A_in.numel() > 0:
            A = A_in
            Bp, M, K = A.shape

            k_node = int(getattr(self.proto_time, "topk_per_node", 8)) if hasattr(self, "proto_time") else 8
            k_node = max(1, min(k_node, K))

            topv, topi = torch.topk(A, k=k_node, dim=-1)  # [B,M,k]

            b_idx = torch.arange(Bp, device=topi.device).view(Bp, 1, 1).expand(Bp, M, k_node).reshape(-1)
            n_idx = torch.arange(M, device=topi.device).view(1, M, 1).expand(Bp, M, k_node).reshape(-1)
            e_idx = topi.reshape(-1)
            w = topv.reshape(-1)

            keep = w > 0
            mask = torch.stack([b_idx[keep], n_idx[keep], e_idx[keep]], dim=0).long()
            weight = w[keep]

            mass = A.to(torch.float32).sum(dim=1)  # [B,K]
            mass_sum = mass.sum(dim=-1, keepdim=True).clamp_min(eps)
            mass_norm = mass / mass_sum  # [B,K]
            mass_pack = _mass_diag_from_mass_norm(mass_norm)

            tv = topv.to(torch.float32)  # [B,M,k]
            row_sum = tv.sum(dim=-1).clamp_min(eps)  # [B,M]
            tv_norm = tv / row_sum.unsqueeze(-1)  # [B,M,k]
            node_ent = _entropy(tv_norm, dim=-1)  # [B,M]
            node_top1_prob = tv_norm[..., 0]  # [B,M]

            node_ent_mean = node_ent.mean()
            node_ent_std = node_ent.std(unbiased=False)
            node_top1_prob_mean = node_top1_prob.mean()
            node_top1_prob_std = node_top1_prob.std(unbiased=False)

            node_top1_idx = topi[..., 0].reshape(-1)  # [B*M]
            top1_counts_ratio, top1_mode_ratio = _top1_hist_from_top1_idx(node_top1_idx, K=int(K))

            proto_param_cos = _offdiag_cos_mean(proto_param) if torch.is_tensor(proto_param) else torch.zeros((),
                                                                                                              dtype=torch.float32)

            max_full_elems = 2_000_000
            Z_proto_export = None
            Z_proto_export_mode = "none"
            Z_proto_full = None
            Z_proto_sampled = None

            Z_proto_mean = None
            Z_proto_pool_mean = None
            proto_token_cos = torch.zeros((), dtype=torch.float32)
            proto_token_norm_mean = None
            proto_token_norm_std = None

            if Z_proto_live is not None and torch.is_tensor(Z_proto_live) and Z_proto_live.numel() > 0:
                ok_shape = (Z_proto_live.dim() == 3 and int(Z_proto_live.size(0)) == int(Bp) and int(
                    Z_proto_live.size(1)) == int(K))
                if ok_shape:
                    Z_proto_mean = Z_proto_live.to(torch.float32).mean(dim=0)  # [K,D]
                    Z_proto_pool_mean = Z_proto_live.to(torch.float32).mean(dim=1)  # [B,D]
                    proto_token_cos = _offdiag_cos_mean(Z_proto_mean)

                    norms = Z_proto_live.to(torch.float32).norm(dim=-1)  # [B,K]
                    proto_token_norm_mean = norms.mean()
                    proto_token_norm_std = norms.std(unbiased=False)

                    if int(Z_proto_live.numel()) <= int(max_full_elems):
                        Z_proto_full = Z_proto_live
                    Z_proto_sampled = Z_proto_live[::ds].contiguous()

                    if Z_proto_full is not None:
                        Z_proto_export = Z_proto_full
                        Z_proto_export_mode = "full"
                    elif Z_proto_sampled is not None:
                        Z_proto_export = Z_proto_sampled
                        Z_proto_export_mode = f"sampled_ds{ds}"

            proto_diag = {
                "node_entropy_mean": node_ent_mean.detach(),
                "node_entropy_std": node_ent_std.detach(),
                "node_top1_prob_mean": node_top1_prob_mean.detach(),
                "node_top1_prob_std": node_top1_prob_std.detach(),
                "top1_mode_ratio": top1_mode_ratio.detach(),
                "top1_counts_ratio": top1_counts_ratio.detach(),
                "proto_mass_norm_mean": mass_pack["mass_norm_mean"].detach(),
                "proto_mass_norm_std": mass_pack["mass_norm_std"].detach(),
                "proto_mass_entropy_mean": mass_pack["mass_entropy_mean"].detach(),
                "proto_hhi_mean": mass_pack["hhi_mean"].detach(),
                "eff_protos_hhi_mean": mass_pack["eff_hhi_mean"].detach(),
                "eff_protos_exp_mean": mass_pack["eff_exp_mean"].detach(),
                "proto_max_mass_ratio_mean": mass_pack["max_mass_ratio_mean"].detach(),
                "active_protos_mean": mass_pack["active_mean"].detach(),
                "proto_param_cos_offdiag_mean": proto_param_cos.detach(),
                "proto_token_cos_offdiag_mean": proto_token_cos.detach(),
            }
            if proto_token_norm_mean is not None:
                proto_diag["proto_token_norm_mean"] = proto_token_norm_mean.detach()
            if proto_token_norm_std is not None:
                proto_diag["proto_token_norm_std"] = proto_token_norm_std.detach()

            self._hg_store["proto"].append({
                "shape": (int(Bp), int(M), int(K)),
                "topi": _cpu(topi, torch.int32),
                "topv": _cpu(topv, torch.float32),
                "incidence_mask": _cpu(mask, torch.int32),
                "incidence_weight": _cpu(weight, torch.float32),

                "proto_mass_norm": _cpu(mass_norm, torch.float32),
                "node_top1_idx": _cpu(topi[..., 0], torch.int32),

                "diag": {k: (_cpu(v, torch.float32) if torch.is_tensor(v) else v) for k, v in proto_diag.items()},

                "proto_reg": proto_reg,
                "proto_param": _cpu(proto_param, torch.float32) if torch.is_tensor(proto_param) else None,

                "Z_proto": _cpu(Z_proto_export, torch.float32) if torch.is_tensor(Z_proto_export) else None,
                "Z_proto_export_mode": Z_proto_export_mode,
                "Z_proto_mean": _cpu(Z_proto_mean, torch.float32) if torch.is_tensor(Z_proto_mean) else None,
                "Z_proto_pool_mean": _cpu(Z_proto_pool_mean, torch.float32) if torch.is_tensor(
                    Z_proto_pool_mean) else None,

                "Z_proto_full": _cpu(Z_proto_full, torch.float32) if torch.is_tensor(Z_proto_full) else None,
                "Z_proto_sampled": _cpu(Z_proto_sampled, torch.float32) if torch.is_tensor(Z_proto_sampled) else None,
            })
        else:
            self._hg_store["proto"].append(None)

        # ============================================================
        # 2) EventTower：结构 + repr + meta（保持你原逻辑）
        # ============================================================
        A_base_list = None
        A_batch_list = None

        H_base_list = None
        H_batch_list = None
        slot_pack_list = None

        repr_layers_mean = None
        repr_layers_max = None
        repr_mean_mask = None
        repr_max_mask = None
        layer_ids = None

        mu_span = None
        mu_hf = None
        layers_export = None

        layer_stats_pack = None

        if isinstance(tower_time, dict):
            A_base_list = tower_time.get("A_base_list", None)
            A_batch_list = tower_time.get("A_batch_list", None)

            H_base_list = tower_time.get("H_base_list", None)
            H_batch_list = tower_time.get("H_batch_list", None)
            slot_pack_list = tower_time.get("slot_pack_list", None)

            mu_span = tower_time.get("mu_span", None)
            mu_hf = tower_time.get("mu_hf", None)

            layer_stats_pack = tower_time.get("_layer_stats", None)

            ce = tower_time.get("content_export", None)
            if isinstance(ce, dict):
                repr_layers_mean = _pick_ce(ce, ["R_mean_list", "R_mean_stack", "R_mean"])
                repr_layers_max = _pick_ce(ce, ["R_max_list", "R_max_stack", "R_max"])
                repr_mean_mask = ce.get("R_mean_mask", None)
                repr_max_mask = ce.get("R_max_mask", None)
                layer_ids = ce.get("layer_ids", None)
                layers_export = ce.get("layers_export", None)

            repr_layers_mean = _to_layer_list(repr_layers_mean)
            repr_layers_max = _to_layer_list(repr_layers_max)

        # ---- 优先：Memory-only tower 的 H_base/H_batch ----
        if isinstance(H_batch_list, (list, tuple)) and len(H_batch_list) > 0:
            tower_pack = {"layers": []}

            for l in range(len(H_batch_list)):
                layer = {}

                # base（只保存一次到 tower_base）
                if isinstance(H_base_list, (list, tuple)) and l < len(H_base_list):
                    hb0 = H_base_list[l]
                    if isinstance(hb0, dict):
                        if self._hg_store["tower_base"] is None:
                            self._hg_store["tower_base"] = []
                        while len(self._hg_store["tower_base"]) <= l:
                            self._hg_store["tower_base"].append(None)
                        if self._hg_store["tower_base"][l] is None:
                            self._hg_store["tower_base"][l] = {
                                "shape": tuple(hb0.get("shape")) if hb0.get("shape", None) is not None else None,
                                "topi": _cpu(hb0.get("topi", None), torch.int32),
                                "topv": _cpu(hb0.get("topv", None), torch.float32),
                                "incidence_mask": _cpu(hb0.get("incidence_mask", None), torch.int32),
                                "incidence_weight": _cpu(hb0.get("incidence_weight", None), torch.float32),
                            }

                # batch（每个 batch 都写到 tower）
                hb = H_batch_list[l]
                if isinstance(hb, dict):
                    topi_cpu = _cpu(hb.get("topi", None), torch.int32)  # [B,E,k]
                    topv_cpu = _cpu(hb.get("topv", None), torch.float32)

                    layer["batch"] = {
                        "shape": tuple(hb.get("shape")) if hb.get("shape", None) is not None else None,
                        "topi": topi_cpu,
                        "topv": topv_cpu,
                        "incidence_mask": _cpu(hb.get("incidence_mask", None), torch.int32),
                        "incidence_weight": _cpu(hb.get("incidence_weight", None), torch.float32),
                    }

                    # transition（上一层 slot -> 当前层 slot 的 top1 转移统计）
                    if l > 0 and topi_cpu is not None and torch.is_tensor(topi_cpu) and topi_cpu.numel() > 0:
                        Bc, Eprev, kview = topi_cpu.shape
                        shape = hb.get("shape", None)
                        Kcurr = int(shape[2]) if (shape is not None and len(shape) >= 3) else int(
                            topi_cpu.max().item() + 1)

                        top1 = topi_cpu[..., 0].to(torch.long)  # [B,Eprev]
                        top1 = top1.clamp_min(0).clamp_max(Kcurr - 1)

                        e_ids = torch.arange(Eprev, dtype=torch.long).view(1, Eprev).expand(Bc, Eprev)  # [B,E]
                        flat = (e_ids * Kcurr + top1).reshape(-1)  # [B*E]
                        counts = torch.bincount(flat, minlength=Eprev * Kcurr).to(torch.float32).reshape(Eprev, Kcurr)
                        hist = counts / float(max(int(Bc), 1))  # [Eprev,Kcurr]

                        row = hist / hist.sum(dim=-1, keepdim=True).clamp_min(eps)
                        row_mode = row.max(dim=-1).values  # [Eprev]
                        row_ent = _entropy(row, dim=-1)  # [Eprev]

                        layer["transition"] = {
                            "shape": (int(Eprev), int(Kcurr)),
                            "transition_top1_hist": _cpu(hist, torch.float32),
                            "transition_row_mode_ratio": _cpu(row_mode, torch.float32),
                            "transition_row_entropy": _cpu(row_ent, torch.float32),
                            "transition_row_mode_ratio_mean": float(row_mode.mean().item()),
                            "transition_row_entropy_mean": float(row_ent.mean().item()),
                        }

                # diag（坍缩诊断）
                if isinstance(slot_pack_list, (list, tuple)) and l < len(slot_pack_list):
                    sp = slot_pack_list[l]
                    if isinstance(sp, dict):
                        diag = sp.get("diag", None)
                        if isinstance(diag, dict):
                            diag_cpu = {}
                            for kk, vv in diag.items():
                                diag_cpu[kk] = _cpu(vv, torch.float32) if torch.is_tensor(vv) else vv
                            layer["diag"] = diag_cpu

                tower_pack["layers"].append(layer)

            self._hg_store["tower"].append(tower_pack)

        # ---- fallback：Attention-tower 的 A_base/A_batch（旧路保留兼容）----
        elif isinstance(A_base_list, (list, tuple)) and len(A_base_list) > 0:
            topk_row = int(getattr(self.event_tower_time, "topk", 16))
            tower_pack = {"layers": []}

            for l, A_base in enumerate(A_base_list):
                layer = {}

                if A_base is not None and torch.is_tensor(A_base) and A_base.numel() > 0:
                    E = int(A_base.size(0))
                    k = max(1, min(topk_row, E))
                    topv, topi = torch.topk(A_base, k=k, dim=-1)
                    edge_idx = torch.arange(E, device=A_base.device).view(E, 1).expand(E, k).reshape(-1)
                    node_idx = topi.reshape(-1)
                    w = topv.reshape(-1)

                    keep = w > 0
                    mask2 = torch.stack([node_idx[keep], edge_idx[keep]], dim=0).long()
                    weight2 = w[keep]

                    if self._hg_store["tower_base"] is None:
                        self._hg_store["tower_base"] = []
                    while len(self._hg_store["tower_base"]) <= l:
                        self._hg_store["tower_base"].append(None)

                    if self._hg_store["tower_base"][l] is None:
                        self._hg_store["tower_base"][l] = {
                            "shape": (E, E),
                            "topi": _cpu(topi, torch.int32),
                            "topv": _cpu(topv, torch.float32),
                            "incidence_mask": _cpu(mask2, torch.int32),
                            "incidence_weight": _cpu(weight2, torch.float32),
                        }

                if isinstance(A_batch_list, (list, tuple)) and l < len(A_batch_list):
                    A_b = A_batch_list[l]
                    if A_b is not None and torch.is_tensor(A_b) and A_b.numel() > 0:
                        Bb, E1, E2 = A_b.shape
                        k = max(1, min(topk_row, E2))
                        topv_b, topi_b = torch.topk(A_b, k=k, dim=-1)

                        b_idx = torch.arange(Bb, device=A_b.device).view(Bb, 1, 1).expand(Bb, E1, k).reshape(-1)
                        edge_idx = torch.arange(E1, device=A_b.device).view(1, E1, 1).expand(Bb, E1, k).reshape(-1)
                        node_idx = topi_b.reshape(-1)
                        w = topv_b.reshape(-1)

                        keep = w > 0
                        mask3 = torch.stack([b_idx[keep], node_idx[keep], edge_idx[keep]], dim=0).long()
                        weight3 = w[keep]

                        layer["batch"] = {
                            "shape": (int(Bb), int(E1), int(E2)),
                            "topi": _cpu(topi_b, torch.int32),
                            "topv": _cpu(topv_b, torch.float32),
                            "incidence_mask": _cpu(mask3, torch.int32),
                            "incidence_weight": _cpu(weight3, torch.float32),
                        }

                if isinstance(layers_export, (list, tuple)) and l < len(layers_export):
                    le = layers_export[l]
                    if isinstance(le, dict):
                        export_pack = {}

                        diff = le.get("diffusion", None)
                        if isinstance(diff, dict):
                            export_pack["diffusion"] = {
                                "K": int(diff.get("K", -1)) if diff.get("K", None) is not None else None,
                                "alpha": _cpu(diff.get("alpha", None), torch.float32) if torch.is_tensor(
                                    diff.get("alpha", None)) else None,
                            }

                        stats = le.get("stats", None)
                        if isinstance(stats, dict):
                            stats_pack = {}
                            for kk, vv in stats.items():
                                stats_pack[kk] = _cpu(vv, torch.float32) if torch.is_tensor(vv) else vv
                            export_pack["stats"] = stats_pack

                        struct = le.get("struct", None)
                        if isinstance(struct, dict):
                            export_pack["struct"] = struct

                        if len(export_pack) > 0:
                            layer["export"] = export_pack

                tower_pack["layers"].append(layer)

            self._hg_store["tower"].append(tower_pack)

        else:
            self._hg_store["tower"].append(None)

        # ============================================================
        # 2.3 repr（pooled）
        # ============================================================
        if isinstance(repr_layers_mean, (list, tuple)) and len(repr_layers_mean) > 0:
            repr_pack = {
                "layer_ids": list(layer_ids) if isinstance(layer_ids, (list, tuple)) else None,
                "mean_mask": _cpu(repr_mean_mask, torch.bool) if torch.is_tensor(repr_mean_mask) else None,
                "max_mask": _cpu(repr_max_mask, torch.bool) if torch.is_tensor(repr_max_mask) else None,
                "mean": [_cpu(x, torch.float32) for x in repr_layers_mean],
                "max": [_cpu(x, torch.float32) for x in (repr_layers_max or [])],
            }
            self._hg_store["tower_repr"].append(repr_pack)
        else:
            self._hg_store["tower_repr"].append(None)

        # ============================================================
        # 2.4 tower_meta：
        # ============================================================
        meta_pack = {}
        if mu_span is not None or mu_hf is not None:
            if torch.is_tensor(mu_span):
                meta_pack["mu_span"] = _cpu(mu_span, torch.float32)
            elif isinstance(mu_span, (list, tuple)):
                meta_pack["mu_span"] = [float(x) for x in mu_span]
            else:
                meta_pack["mu_span"] = mu_span

            if torch.is_tensor(mu_hf):
                meta_pack["mu_hf"] = _cpu(mu_hf, torch.float32)
            elif isinstance(mu_hf, (list, tuple)):
                meta_pack["mu_hf"] = [float(x) for x in mu_hf]
            else:
                meta_pack["mu_hf"] = mu_hf

        if isinstance(layer_stats_pack, dict) and len(layer_stats_pack) > 0:
            ls_cpu = {}
            for kk, vv in layer_stats_pack.items():
                if torch.is_tensor(vv):
                    ls_cpu[kk] = _cpu(vv, torch.float32)
                elif isinstance(vv, (list, tuple)):
                    tmp = []
                    for x in vv:
                        tmp.append(_cpu(x, torch.float32) if torch.is_tensor(x) else x)
                    ls_cpu[kk] = tmp
                else:
                    ls_cpu[kk] = vv
            meta_pack["layer_stats"] = ls_cpu

        self._hg_store["tower_meta"].append(meta_pack if len(meta_pack) > 0 else None)

        # ============================================================
        # 3) Content feat (span/hf)
        # ============================================================
        if content_feat_time is not None and torch.is_tensor(content_feat_time) and content_feat_time.numel() > 0:
            self._hg_store["content"].append({
                "shape": tuple(content_feat_time.shape),
                "content_feat": _cpu(content_feat_time, torch.float32),
            })
        else:
            self._hg_store["content"].append(None)

        # ============================================================
        # 4) Branch pack
        # ============================================================
        if isinstance(branch_pack, dict) and len(branch_pack) > 0:
            bp = {}
            for k in ["y_lin", "y_nodes", "y_event"]:
                if k in branch_pack:
                    bp[k] = _cpu(branch_pack.get(k), torch.float32)

            for k in ["Z_nodes", "decoded", "context_tokens", "Z_event_all"]:
                if k in branch_pack:
                    v = branch_pack.get(k)
                    bp[k] = _cpu(v, torch.float32) if (torch.is_tensor(v) and v.numel() > 0) else None

            for k in ["context_tokens_idx"]:
                if k in branch_pack:
                    v = branch_pack.get(k)
                    bp[k] = _cpu(v, torch.int32) if torch.is_tensor(v) else None

            self._hg_store["branches"].append(bp)
        else:
            self._hg_store["branches"].append(None)

        # ============================================================
        # 5) Time HG pack：pre/post HGConv 的节点 token（多尺度）
        #    + NEW: 多尺度节点“进入前(pre)”相似度统计
        # ============================================================
        if isinstance(time_hg_pack, dict) and len(time_hg_pack) > 0:
            max_tokens = int(getattr(self, "viz_max_pred", 256) or 256)

            # 相似度统计的 subsample 上限（不新增 flag，仅用常量 + viz_max_pred）
            sim_max_nodes = int(min(max_tokens, 256))
            sim_hist_bins = 41

            pack = {"scales": []}

            # 用于跨尺度 pooled cosine matrix
            pre_pool_list_cpu32 = []
            post_pool_list_cpu32 = []

            # 用于 union(pre/post) 相似度：拼接各尺度 subsampled tokens
            union_pre_list = []
            union_post_list = []

            scales = time_hg_pack.get("scales", None)
            if isinstance(scales, (list, tuple)):
                for item in scales:
                    if not isinstance(item, dict):
                        continue
                    Xi_pre = item.get("pre", None)
                    Xi_post = item.get("post", None)
                    scale_i = item.get("scale", None)

                    entry = {"scale": int(scale_i) if scale_i is not None else None}

                    # -------- pre --------
                    if torch.is_tensor(Xi_pre) and Xi_pre.numel() > 0:
                        entry["pre_shape"] = tuple(Xi_pre.shape)

                        # pooled
                        pre_pool = Xi_pre.mean(dim=1).detach().to(torch.float32).cpu()  # [B,D]
                        entry["pre_pool_mean"] = _cpu(Xi_pre.mean(dim=1), torch.float32)
                        entry["pre_pool_max"] = _cpu(Xi_pre.max(dim=1).values, torch.float32)
                        pre_pool_list_cpu32.append(pre_pool)

                        # subsample tokens (for export)
                        Xs, idx = _subsample_tokens(Xi_pre, max_tokens=max_tokens)
                        entry["pre_tokens"] = _cpu(Xs, torch.float32) if Xs is not None else None
                        entry["pre_tokens_idx"] = _cpu(idx, torch.int32) if idx is not None else None

                        # NEW: similarity stats on CPU float32 (deterministic subsample to sim_max_nodes)
                        Xs_sim, _ = _subsample_tokens(Xi_pre, max_tokens=sim_max_nodes)
                        if Xs_sim is not None and torch.is_tensor(Xs_sim) and Xs_sim.numel() > 0:
                            Xs_sim_cpu = Xs_sim.detach().to(torch.float32).cpu()  # [B,N,D]
                            sim_stats = _cos_offdiag_stats_batched(Xs_sim_cpu, hist_bins=sim_hist_bins)
                            if sim_stats is not None:
                                entry["pre_sim"] = {
                                    "num_nodes_total": int(Xi_pre.size(1)),
                                    **{k: (_cpu(v, torch.float32) if torch.is_tensor(v) else v) for k, v in
                                       sim_stats.items()},
                                }
                            union_pre_list.append(Xs_sim_cpu)
                        else:
                            entry["pre_sim"] = None
                    else:
                        entry.update({
                            "pre_shape": None,
                            "pre_pool_mean": None,
                            "pre_pool_max": None,
                            "pre_tokens": None,
                            "pre_tokens_idx": None,
                            "pre_sim": None,
                        })

                    # -------- post --------
                    if torch.is_tensor(Xi_post) and Xi_post.numel() > 0:
                        entry["post_shape"] = tuple(Xi_post.shape)

                        post_pool = Xi_post.mean(dim=1).detach().to(torch.float32).cpu()  # [B,D]
                        entry["post_pool_mean"] = _cpu(Xi_post.mean(dim=1), torch.float32)
                        entry["post_pool_max"] = _cpu(Xi_post.max(dim=1).values, torch.float32)
                        post_pool_list_cpu32.append(post_pool)

                        Xs, idx = _subsample_tokens(Xi_post, max_tokens=max_tokens)
                        entry["post_tokens"] = _cpu(Xs, torch.float32) if Xs is not None else None
                        entry["post_tokens_idx"] = _cpu(idx, torch.int32) if idx is not None else None

                        Xs_sim, _ = _subsample_tokens(Xi_post, max_tokens=sim_max_nodes)
                        if Xs_sim is not None and torch.is_tensor(Xs_sim) and Xs_sim.numel() > 0:
                            Xs_sim_cpu = Xs_sim.detach().to(torch.float32).cpu()
                            sim_stats = _cos_offdiag_stats_batched(Xs_sim_cpu, hist_bins=sim_hist_bins)
                            if sim_stats is not None:
                                entry["post_sim"] = {
                                    "num_nodes_total": int(Xi_post.size(1)),
                                    **{k: (_cpu(v, torch.float32) if torch.is_tensor(v) else v) for k, v in
                                       sim_stats.items()},
                                }
                            union_post_list.append(Xs_sim_cpu)
                        else:
                            entry["post_sim"] = None
                    else:
                        entry.update({
                            "post_shape": None,
                            "post_pool_mean": None,
                            "post_pool_max": None,
                            "post_tokens": None,
                            "post_tokens_idx": None,
                            "post_sim": None,
                        })

                    pack["scales"].append(entry)

            # NEW: pack-level similarity summaries
            sim_pack = {}

            # (a) 跨尺度 pooled mean cosine matrix（pre/post）
            pre_cos_mat = _scale_pool_cos_matrix(pre_pool_list_cpu32)
            post_cos_mat = _scale_pool_cos_matrix(post_pool_list_cpu32)
            sim_pack["scale_pool_cos_pre"] = _cpu(pre_cos_mat, torch.float32) if torch.is_tensor(pre_cos_mat) else None
            sim_pack["scale_pool_cos_post"] = _cpu(post_cos_mat, torch.float32) if torch.is_tensor(
                post_cos_mat) else None

            # (b) union(pre/post) overall off-diagonal similarity
            if len(union_pre_list) > 0:
                # union by concat on node-dim (requires same B,D)
                try:
                    union_pre = torch.cat(union_pre_list, dim=1)  # [B, sumN, D]
                    sim_pack["union_pre_sim"] = _cos_offdiag_stats_batched(union_pre, hist_bins=sim_hist_bins)
                    if sim_pack["union_pre_sim"] is not None:
                        # hist tensor already CPU float32
                        pass
                except Exception:
                    sim_pack["union_pre_sim"] = None
            else:
                sim_pack["union_pre_sim"] = None

            if len(union_post_list) > 0:
                try:
                    union_post = torch.cat(union_post_list, dim=1)
                    sim_pack["union_post_sim"] = _cos_offdiag_stats_batched(union_post, hist_bins=sim_hist_bins)
                except Exception:
                    sim_pack["union_post_sim"] = None
            else:
                sim_pack["union_post_sim"] = None

            # 标注参数（便于你离线脚本知道采样尺度）
            sim_pack["sim_max_nodes_per_scale"] = int(sim_max_nodes)
            sim_pack["sim_hist_bins"] = int(sim_hist_bins)

            pack["ms_similarity"] = sim_pack

            self._hg_store["time_hg"].append(pack)
        else:
            self._hg_store["time_hg"].append(None)

        # ============================================================
        # 6) Tower tokens：事件塔各层 token（保持你原逻辑）
        # ============================================================
        if isinstance(tower_token_pack, dict) and len(tower_token_pack) > 0:
            tp = {}

            Z_layers_raw = tower_token_pack.get("Z_layers_raw", None)
            Z_layers_proj = tower_token_pack.get("Z_layers_proj", None)

            if isinstance(Z_layers_raw, (list, tuple)) and len(Z_layers_raw) > 0:
                tp["Z_layers_raw"] = [_cpu(z, torch.float32) if torch.is_tensor(z) else None for z in Z_layers_raw]
                tp["layer_num_raw"] = int(len(Z_layers_raw))
                tp["Z_layers_raw_pool_mean"] = [
                    _cpu(z.mean(dim=1), torch.float32) if (torch.is_tensor(z) and z.numel() > 0) else None
                    for z in Z_layers_raw
                ]
            else:
                tp["Z_layers_raw"] = None
                tp["Z_layers_raw_pool_mean"] = None
                tp["layer_num_raw"] = 0

            if isinstance(Z_layers_proj, (list, tuple)) and len(Z_layers_proj) > 0:
                tp["Z_layers_proj"] = [_cpu(z, torch.float32) if torch.is_tensor(z) else None for z in Z_layers_proj]
                tp["layer_num_proj"] = int(len(Z_layers_proj))
                tp["Z_layers_proj_pool_mean"] = [
                    _cpu(z.mean(dim=1), torch.float32) if (torch.is_tensor(z) and z.numel() > 0) else None
                    for z in Z_layers_proj
                ]
            else:
                tp["Z_layers_proj"] = None
                tp["Z_layers_proj_pool_mean"] = None
                tp["layer_num_proj"] = 0

            z_top = tower_token_pack.get("Z_top_raw", None)
            tp["Z_top_raw_pool_mean"] = _cpu(z_top.mean(dim=1), torch.float32) if (
                        torch.is_tensor(z_top) and z_top.numel() > 0) else None

            self._hg_store["tower_tokens"].append(tp)
        else:
            self._hg_store["tower_tokens"].append(None)

    # ============================================================
    #  原 __init__：仅加入 self._hg_store=None（其余保持你贴的版本）
    # ============================================================
    def __init__(self, configs):
        super().__init__()

        # ================== 通用基础参数 ==================
        self.seq_len = _as_int(getattr(configs, 'seq_len', 96), 96)
        self.pred_len = _as_int(getattr(configs, 'pred_len', 96), 96)
        self.channels = _as_int(getattr(configs, 'enc_in', 7), 7)
        self.individual = bool(getattr(configs, 'individual', True))
        self.dropout = float(getattr(configs, 'dropout', 0.1))

        # ========= cycle_index 位置编码 =========
        self.use_cycle_index = bool(getattr(configs, 'use_cycle_index', True))
        self.cycle_period = int(getattr(configs, 'cycle', 0))
        self.cycle_encoding = getattr(configs, 'cycle_encoding', 'sincos')

        if self.use_cycle_index:
            _cycle_in_dim = 2 if self.cycle_encoding == 'sincos' else 1
            self.cycle_proj = nn.Linear(_cycle_in_dim, self.channels)
        else:
            self.cycle_proj = None

        # ========== 可视化控制 ==========
        self.viz_auto_eval = bool(getattr(configs, 'viz_auto_eval', True))
        self.viz_max_pred = getattr(configs, 'viz_max_pred', None)
        self.viz_downsample = int(getattr(configs, 'viz_downsample', 2))
        self._viz_reset()

        # ========== hg_store（新增初始化） ==========
        self._hg_store = None

        # ========== 正则项权重 ==========
        def _get_float(name, default):
            raw_val = getattr(configs, name, default)
            try:
                return float(raw_val)
            except (TypeError, ValueError):
                return default

        self.w_e2e = _get_float('w_e2e', 1e-1)
        self.w_gp = _get_float('w_gp', 5e-1)
        self.w_ee_smooth = _get_float('w_ee_smooth', 1e-1)
        self.w_event_content = _get_float('w_event_content', 0.3)

        # ======================= 时间分支（唯一主干） =======================
        ctor_name = getattr(configs, 't_cscm', getattr(configs, 'CSCM', 'Bottleneck_Construct'))
        ctor = CONSTRUCTORS.get(ctor_name)
        if ctor is None:
            raise NameError(f'Unknown CSCM={ctor_name}. Choose from {list(CONSTRUCTORS.keys())}')

        raw_window_size = getattr(configs, 't_window_size', None)
        if raw_window_size is None:
            raise ValueError("[STHyperModel] configs.t_window_size is None，请显式配置 t_window_size。")
        if isinstance(raw_window_size, int):
            raw_window_size = [int(raw_window_size)]
        else:
            raw_window_size = [int(x) for x in raw_window_size]
        self.window_size = raw_window_size

        all_size = get_mask(self.seq_len, self.window_size)
        self.all_size = [int(m) for m in (all_size if isinstance(all_size, (list, tuple)) else [all_size])]
        if len(self.all_size) == 0:
            raise ValueError(f'[STHyperModel][TimeBranch] get_mask 返回空：seq_len={self.seq_len}, window_size={self.window_size}')

        raw_hyper_num = getattr(configs, 't_hyper_num', None)
        if raw_hyper_num is None:
            raw_hyper_num = getattr(configs, 'hyper_num', [50])
        if isinstance(raw_hyper_num, int):
            raw_hyper_num = [int(raw_hyper_num)]
        else:
            raw_hyper_num = [int(x) for x in raw_hyper_num]

        n_hyper = len(raw_hyper_num)
        if n_hyper == 0:
            raise ValueError(f'[STHyperModel][TimeBranch] t_hyper_num/hyper_num 为空：{raw_hyper_num}')
        if n_hyper > len(self.all_size):
            raise ValueError(
                f'[STHyperModel][TimeBranch] len(t_hyper_num)={n_hyper} > len(get_mask)={len(self.all_size)}；'
                f'seq_len={self.seq_len}, t_window_size={self.window_size}, t_hyper_num={raw_hyper_num}'
            )

        self.hyper_num1 = [max(0, int(e)) for e in raw_hyper_num]
        self.n_time_scales = n_hyper

        self.t_d_model = getattr(configs, 't_d_model', 128)
        self.conv_layers = ctor(self.channels, self.window_size, self.t_d_model)

        time_M_list = self.all_size[:self.n_time_scales]
        self.Ms_length = int(sum(time_M_list))

        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        with torch.no_grad():
            self.out_tran.weight.copy_((1.0 / max(1, self.Ms_length)) * torch.ones(self.pred_len, self.Ms_length))
            if self.out_tran.bias is not None:
                self.out_tran.bias.zero_()

        self.multiadphyper = multi_adaptive_hypergraoh_prob(configs)

        heads_time = int(getattr(configs, 'hyper_heads_time', 4))
        neg_slope_time = float(getattr(configs, 'hyper_negative_slope_time', 0.2))
        dropout_time = float(getattr(configs, 'hyper_dropout_time', 0.3))
        gamma_time = float(getattr(configs, 't_gamma_dist', 0.2))
        lambda_nc_time = float(getattr(configs, 'lambda_nc_time', getattr(configs, 'lambda_nc', 0.5)))
        self.lambda_nc_time = lambda_nc_time

        self.time_hgconvs = nn.ModuleList([
            HypergraphConv(
                in_channels=self.t_d_model,
                out_channels=self.t_d_model,
                use_attention=True,
                heads=heads_time,
                concat=True,
                negative_slope=neg_slope_time,
                dropout=dropout_time,
                bias=False,
                gamma=gamma_time,
                lambda_nc=lambda_nc_time,
            )
            for _ in range(len(self.hyper_num1))
        ])

        self.t_min_edge_card = int(getattr(configs, "t_min_edge_card", 2))
        self.t_use_ring_fallback = bool(getattr(configs, "t_use_ring_fallback", False))

        self.use_local_hg_refine = bool(getattr(configs, "use_local_hg_refine", False))
        self.local_refine_topk = int(getattr(configs, "local_refine_topk", 0))
        self.local_refine_threshold = float(getattr(configs, "local_refine_threshold", 0.0))

        if self.use_local_hg_refine:
            self.edge_refiners = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.t_d_model, self.t_d_model),
                    nn.GELU(),
                    nn.Linear(self.t_d_model, 1),
                )
                for _ in range(len(self.hyper_num1))
            ])
        else:
            self.edge_refiners = None

        # ================== ProtoMemory + 事件塔 + QueryDecoder ==================
        self.event_dim = int(getattr(configs, "event_dim", self.channels))
        num_protos = int(getattr(configs, "num_event_protos", 32))
        proto_topk_ratio = float(getattr(configs, "event_topk_ratio", 0.0))
        proto_topk_int = int(round(proto_topk_ratio * num_protos))
        proto_topk_int = max(1, min(proto_topk_int, num_protos))

        self.proto_time = ProtoMemoryHypergraph(
            configs =configs,
            D_in=self.t_d_model,
            D_mem=self.event_dim,
            num_protos=num_protos,
            dropout=float(getattr(configs, "event_dropout", self.dropout)),
            topk_per_node=proto_topk_int,
            w_proto_div=float(getattr(configs, "w_proto_div", 0.3)),
            w_proto_usage=float(getattr(configs, "w_proto_usage", 0.3)),
        )

        self.event_tower_time = SemanticEventTowerV2(
            configs=configs,
            D=self.event_dim,
            num_layers=int(getattr(configs, "event_layers", 2)),
            topk=int(getattr(configs, "event_topk_ratio", 0.5)),
            dropout=float(getattr(configs, "event_dropout", self.dropout)),
            w_div=float(getattr(configs, "w_event_div", 0.1)),
            w_hier=float(getattr(configs, "w_event_hier", 1.0)),
        )

        self.event_proj = nn.Linear(self.event_dim, self.t_d_model) if self.event_dim != self.t_d_model else None
        self.token_type_embed = nn.Embedding(2, self.t_d_model)
        self.event_token_ln = nn.LayerNorm(self.t_d_model)

        dec_d_model = self.t_d_model
        dec_heads = int(getattr(configs, "dec_heads", 4))
        dec_layers = int(getattr(configs, "dec_layers", 1))
        dec_ffn_dim = int(getattr(configs, "dec_ffn_dim", 2 * dec_d_model))
        dec_dropout = float(getattr(configs, "dec_dropout", self.dropout))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dec_d_model,
            nhead=dec_heads,
            dim_feedforward=dec_ffn_dim,
            dropout=dec_dropout,
            batch_first=False,
        )
        self.query_decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        self.query_embed = nn.Parameter(torch.randn(self.pred_len, dec_d_model) * 0.02)
        self.dec_out_proj = nn.Linear(dec_d_model, self.channels)

        # ================== Linear baseline + node head ==================
        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.node_out_proj = nn.Linear(self.t_d_model, self.channels)
        self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        fuse_in = 3 * self.channels
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, self.channels, bias=True),
        )
        with torch.no_grad():
            lin = self.fuse_mlp[1]
            lin.weight.zero_()
            lin.bias.zero_()
            C = self.channels
            for c in range(C):
                lin.weight[c, c] = 1.0
                lin.weight[c, C + c] = 1.0
                lin.weight[c, 2 * C + c] = 1.0

        self._last_branch_stats = {}
        self.collect_node_stats = bool(getattr(configs, "collect_node_stats", True))
        self._last_node_stats = {}

        self.use_event = bool(getattr(configs, 'use_event', True))
        self.use_node = bool(getattr(configs, 'use_node', True))
        self.use_linear = bool(getattr(configs, 'use_linear', True))

    # ================== Debug / Viz APIs ==================
    def set_viz(self, enable: bool, max_pred=None, downsample: int = 1):
        return

    def get_debug(self, clear: bool = True):
        stats = getattr(self, "_last_branch_stats", {})
        if clear:
            self._last_branch_stats = {}
        return stats

    def clear_debug(self):
        self._last_branch_stats = {}

    @torch.no_grad()
    def get_hypergraph_init_and_layers(self):
        return self._viz_H0, list(self._viz_H_layers) if self._viz_H_layers is not None else []

    @torch.no_grad()
    def get_edge_gates(self):
        return getattr(self, "_last_edge_gates", [])

    @torch.no_grad()
    def get_node_stats(self, clear: bool = True):
        stats = getattr(self, "_last_node_stats", {})
        if clear:
            self._last_node_stats = {}
        return stats

    # ================== forward ==================
    def forward(self, x, x_mark_enc=None, cycle_index=None):
        """
        x: [B, L, C]
        return:
          y  : [B, pred_len, C]
          reg: dict
        """
        self._viz_reset()

        # ===== 1) 标准化 =====
        mean_enc = x.mean(1, keepdim=True).detach()  # [B,1,C]
        x_norm = x - mean_enc
        std_enc = torch.sqrt(torch.var(x_norm, dim=1, keepdim=False).mean(dim=1, keepdim=True) + 1e-5).detach()  # [B,1]
        std_enc = std_enc.unsqueeze(1)  # [B,1,1]
        x_norm = x_norm / std_enc

        B, L, Cref = x_norm.shape
        device = x_norm.device
        zero = x_norm.new_zeros(())

        # ===== 1.x) cycle_index 注入 =====
        if self.use_cycle_index and (cycle_index is not None) and (self.cycle_proj is not None):
            if not torch.is_tensor(cycle_index):
                cycle_index = torch.as_tensor(cycle_index, device=device)
            else:
                cycle_index = cycle_index.to(device)

            while cycle_index.dim() > 2 and cycle_index.size(-1) == 1:
                cycle_index = cycle_index.squeeze(-1)

            if cycle_index.dim() == 1:
                if cycle_index.size(0) != B:
                    raise RuntimeError(f"[STHyperModel] cycle_index {tuple(cycle_index.shape)} != B={B}")
                cycle_index = cycle_index.view(B, 1).expand(B, L)
            elif cycle_index.dim() == 2:
                if cycle_index.size(0) != B:
                    raise RuntimeError(f"[STHyperModel] cycle_index {tuple(cycle_index.shape)} != B={B}")
                if cycle_index.size(1) == 1:
                    cycle_index = cycle_index.expand(B, L)
                elif cycle_index.size(1) != L:
                    raise RuntimeError(f"[STHyperModel] cycle_index 期望 [B,L]=({B},{L})，当前 {tuple(cycle_index.shape)}")
            else:
                raise RuntimeError(f"[STHyperModel] cycle_index dim={cycle_index.dim()} 非法")

            idx = cycle_index.float()
            if self.cycle_encoding == 'raw':
                idx_min = idx.min(dim=1, keepdim=True).values
                idx_max = idx.max(dim=1, keepdim=True).values
                denom = (idx_max - idx_min).clamp_min(1.0)
                pos = (idx - idx_min) / denom
                feat = pos.unsqueeze(-1)
            else:
                if self.cycle_period <= 0:
                    raise RuntimeError(f"[STHyperModel] cycle_encoding='{self.cycle_encoding}' 但 configs.cycle<=0")
                T = float(self.cycle_period)
                phase_idx = torch.remainder(idx, T)
                phase = 2.0 * math.pi * phase_idx / T
                if self.cycle_encoding == 'sincos':
                    feat = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
                else:
                    feat = torch.sin(phase).unsqueeze(-1)

            x_norm = x_norm + self.cycle_proj(feat)

        # ===== 分支开关 =====
        use_linear = bool(getattr(self, "use_linear", True))
        use_node = bool(getattr(self, "use_node", True))
        use_event = bool(getattr(self, "use_event", True))
        if (not use_linear) and (not use_node) and (not use_event):
            use_linear = True

        # ===== 2) Linear baseline =====
        if self.individual:
            y_lin = x_norm.new_zeros(B, self.pred_len, self.channels)
            for i in range(self.channels):
                y_lin[:, :, i] = self.Linear[i](x_norm[:, :, i])
        else:
            tmp = self.Linear(x_norm.permute(0, 2, 1))
            y_lin = tmp.permute(0, 2, 1)

        # 仅 linear
        if (not use_node) and (not use_event):
            y_time = y_lin if use_linear else x_norm.new_zeros(B, self.pred_len, self.channels)

            # ===== eval/test 导出（即使只有 linear，也写 branches 便于对齐）=====
            if not self.training:
                branch_pack = {"y_lin": y_lin, "y_nodes": None, "y_event": None}
                self._append_hg_store(
                    A_np=None, tower_time=None, content_feat_time=None,
                    branch_pack=branch_pack, time_hg_pack=None, tower_token_pack=None
                )

            with torch.no_grad():
                self._last_branch_stats = {
                    "y_lin_mean_abs": y_lin.detach().abs().mean().item(),
                    "y_nodes_mean_abs": 0.0,
                    "y_event_mean_abs": 0.0,
                    "y_fuse_mean_abs": y_time.detach().abs().mean().item(),
                }
            y_time = self.Linear_Tran(y_time.permute(0, 2, 1)).permute(0, 2, 1)
            y = y_time * std_enc + mean_enc
            reg = {
                'e2e': zero, 'gp': zero, 'hyper': zero, 'ee_smooth': zero,
                'const_node_time': zero, 'const_time': zero,
                'event_div': zero, 'event_node': zero, 'tower_hier': zero,
                'proto_div': zero, 'proto_usage': zero, 'tower_content': zero,
            }
            return y, reg

        # ===== 3) 多尺度时间特征 + AHL 构图 =====
        raw = self.conv_layers(x_norm)
        seq_list = raw if isinstance(raw, (list, tuple)) else [raw]

        adj_info_ext = self.multiadphyper(x_norm)
        if not isinstance(adj_info_ext, (list, tuple)):
            adj_info_ext = [adj_info_ext]

        n_expected = len(self.hyper_num1)
        if len(seq_list) < n_expected:
            raise RuntimeError(f"[TimeBranch] conv_layers 输出尺度数={len(seq_list)} < 超边尺度数={n_expected}")
        if len(adj_info_ext) < n_expected:
            raise RuntimeError(f"[TimeBranch] multiadphyper 输出尺度数={len(adj_info_ext)} < 超边尺度数={n_expected}")

        # 3.1 清洗每尺度超图
        adj_info = []
        for i in range(n_expected):
            Xi = seq_list[i]
            info_i = Model._sanitize_time_scale(
                M_feat=Xi.size(1),
                E_conf=int(self.hyper_num1[i]),
                raw_info=adj_info_ext[i],
                min_edge_card=int(self.t_min_edge_card),
                use_ring_fallback=bool(self.t_use_ring_fallback),
                scale_idx=i,
                device=device,
            )
            adj_info.append(info_i)

        # ===== 4) Node branch：多尺度 HGConv -> 预测 =====
        X_time_ms_list = []
        loss_const_time = x_norm.new_zeros(())
        if use_node:
            for i in range(n_expected):
                Xi = seq_list[i]
                info_i = adj_info[i]
                M_i, E_i = int(info_i['M']), int(info_i['E'])
                m = info_i['mask']
                if M_i <= 0 or E_i <= 0 or m is None or m.numel() == 0:
                    raise RuntimeError(f"[TimeBranch][Scale {i}] 清洗后超图非法：M={M_i}, E={E_i}")
                m = m.to(device)

                if self.use_local_hg_refine:
                    Xi_conv, loss_i = self._hyperconv_with_local_refine(self.time_hgconvs[i], Xi, m, M_i, E_i, scale_idx=i)
                else:
                    Xi_conv, loss_i = self.time_hgconvs[i](Xi, m)

                X_time_ms_list.append(Xi_conv)
                loss_const_time = loss_const_time + loss_i

            # 节点统计（保持）
            if self.collect_node_stats and len(X_time_ms_list) > 0:
                per_node_energy_ms, per_node_var_ms, per_node_meanabs_ms, per_node_diff_ms = [], [], [], []
                scale_mean_abs, scale_std, scale_max_abs, scale_l2, scale_diff_energy = [], [], [], [], []

                for Xi_conv in X_time_ms_list:
                    B_i, M_i, C_i = Xi_conv.shape
                    energy_i = Xi_conv.pow(2).mean(dim=-1, keepdim=True)
                    var_i = Xi_conv.var(dim=-1, unbiased=False, keepdim=True)
                    meanabs_i = Xi_conv.abs().mean(dim=-1, keepdim=True)
                    if M_i > 1:
                        diff_i = (Xi_conv[:, 1:, :] - Xi_conv[:, :-1, :]).pow(2).mean(dim=-1, keepdim=True)
                        diff_i = torch.cat([diff_i, Xi_conv.new_zeros(B_i, 1, 1)], dim=1)
                    else:
                        diff_i = Xi_conv.new_zeros(B_i, M_i, 1)

                    per_node_energy_ms.append(energy_i.detach().cpu())
                    per_node_var_ms.append(var_i.detach().cpu())
                    per_node_meanabs_ms.append(meanabs_i.detach().cpu())
                    per_node_diff_ms.append(diff_i.detach().cpu())

                    scale_mean_abs.append(Xi_conv.abs().mean().item())
                    scale_std.append(Xi_conv.std(unbiased=False).item())
                    scale_max_abs.append(Xi_conv.abs().max().item())
                    scale_l2.append(Xi_conv.pow(2).mean().sqrt().item())
                    if M_i > 1:
                        scale_diff_energy.append((Xi_conv[:, 1:, :] - Xi_conv[:, :-1, :]).pow(2).mean().item())
                    else:
                        scale_diff_energy.append(0.0)

                self._last_node_stats = {
                    "per_node": {"energy": per_node_energy_ms, "var": per_node_var_ms, "mean_abs": per_node_meanabs_ms, "diff_energy": per_node_diff_ms},
                    "per_scale": {"mean_abs": scale_mean_abs, "std": scale_std, "max_abs": scale_max_abs, "l2": scale_l2, "diff_energy": scale_diff_energy},
                }
            else:
                self._last_node_stats = {}

            X_ms_nodes_concat = torch.cat(X_time_ms_list, dim=1) if len(X_time_ms_list) > 0 else x_norm[:, :0, :]
            if X_ms_nodes_concat.size(1) > 0:
                nodes_concat_T = X_ms_nodes_concat.permute(0, 2, 1)
                if nodes_concat_T.size(-1) != self.Ms_length:
                    if nodes_concat_T.size(-1) < self.Ms_length:
                        nodes_concat_T = F.pad(nodes_concat_T, (0, self.Ms_length - nodes_concat_T.size(-1)))
                    else:
                        nodes_concat_T = F.adaptive_avg_pool1d(nodes_concat_T, self.Ms_length)
                Z_nodes = self.out_tran(nodes_concat_T).permute(0, 2, 1)  # [B,pred_len,t_d_model]
            else:
                Z_nodes = x_norm.new_zeros(B, self.pred_len, self.t_d_model)

            y_nodes = self.node_out_proj(Z_nodes)
            loss_node_const_time = loss_const_time
        else:
            self._last_node_stats = {}
            Z_nodes = None
            y_nodes = x_norm.new_zeros(B, self.pred_len, self.channels)
            loss_const_time = zero
            loss_node_const_time = zero

        # ===== 5) union 节点（事件分支输入）=====
        Xn_time_union_list, mask_parts = [], []
        node_base, edge_offset, E_time_init = 0, 0, 0
        for i in range(n_expected):
            Xi = seq_list[i]
            info = adj_info[i]
            M_i, E_i = int(info['M']), int(info['E'])
            m = info['mask']
            Xn_time_union_list.append(Xi)
            if m is not None and m.numel() > 0:
                m = m.to(device)
                mask_parts.append(torch.stack([m[0] + node_base, m[1] + edge_offset], dim=0))
            node_base += M_i
            edge_offset += E_i
            E_time_init += E_i

        Xn_time_union = torch.cat(Xn_time_union_list, dim=1) if len(Xn_time_union_list) > 0 else x_norm[:, :0, :]
        mask_time_union = torch.cat(mask_parts, dim=1) if len(mask_parts) > 0 else torch.zeros(2, 0, dtype=torch.long, device=device)
        M_time_union = int(sum(int(adj_info[i]['M']) for i in range(n_expected))) if n_expected > 0 else 0

        # 可视化 H0（保持）
        if (M_time_union > 0 and E_time_init > 0 and mask_time_union.numel() > 0):
            vals = torch.ones(mask_time_union.size(1), device=device, dtype=x_norm.dtype)
            try:
                H0 = torch.sparse_coo_tensor(mask_time_union.long(), vals, size=(M_time_union, E_time_init))
                self._viz_H0 = H0.coalesce()
            except Exception:
                self._viz_H0 = None
        else:
            self._viz_H0 = None

        # ===== 6) Event branch：ProtoMemory -> Tower -> Decoder =====
        A_np = None
        content_feat_time = None
        tower_time = None
        tower_token_pack = None
        decoded = None
        context_tokens = None
        Z_event_all = None

        if use_event and (M_time_union > 0):
            Z_proto, reg_proto_time, extra_proto_time = self.proto_time(Xn_time_union, mask_valid=None, return_sparse=False)
            A_np = extra_proto_time.get("A_np", None)
            content_feat_time = self._build_event_content_feat(Xn_time_union, A_np) if (A_np is not None) else None

            if Z_proto.size(1) > 0:
                Z_event_top, reg_tower_time, tower_time = self.event_tower_time(Z_proto)
                # 额外塞入 layer_stats（用于 _append_hg_store -> tower_meta.layer_stats）
                if hasattr(self.event_tower_time, "get_layer_stats"):
                    try:
                        tower_time["_layer_stats"] = self.event_tower_time.get_layer_stats()
                    except Exception:
                        tower_time["_layer_stats"] = None
            else:
                Z_event_top = Z_proto
                reg_tower_time = {"e2e": zero, "event_div": zero, "tower_hier": zero}
                tower_time = {"Z_layers": [], "A_base_list": [], "A_batch_list": [], "H_base_list": [], "H_batch_list": [], "slot_pack_list": []}

            # 将 tower 的各层 token 投影到 t_d_model，并在“拼接之前”做 LN
            proj_layers = []
            raw_layers = []
            if Z_proto.size(1) > 0:
                Z_layers = tower_time.get("Z_layers", None)
                for Z_l in (Z_layers or []):
                    raw_layers.append(Z_l)
                    if Z_l.size(-1) != self.t_d_model:
                        Z_l = self.event_proj(Z_l) if self.event_proj is not None else Z_l[..., :self.t_d_model]
                    Z_l = self.event_token_ln(Z_l)
                    proj_layers.append(Z_l)
                Z_event_all = torch.cat(proj_layers, dim=1) if len(proj_layers) > 0 else x_norm.new_zeros(B, 0, self.t_d_model)
            else:
                Z_event_all = x_norm.new_zeros(B, 0, self.t_d_model)

            # Decoder：只用事件 token 做 memory
            if Z_event_all.size(1) == 0:
                y_event = x_norm.new_zeros(B, self.pred_len, self.channels)
            else:
                event_type_emb = self.token_type_embed(torch.ones(1, dtype=torch.long, device=device))[0]  # [D]
                context_tokens = Z_event_all + event_type_emb.view(1, 1, -1)  # [B,S,D]
                tgt = self.query_embed.unsqueeze(0).expand(B, -1, -1).transpose(0, 1)   # [Tq,B,D]
                memory = context_tokens.transpose(0, 1)                                  # [S,B,D]
                decoded = self.query_decoder(tgt=tgt, memory=memory).transpose(0, 1)     # [B,Tq,D]
                y_event = self.dec_out_proj(decoded)                                     # [B,pred,C]

            # tower_token_pack（用于 hg_store['tower_tokens']）
            tower_token_pack = {
                "Z_layers_raw": raw_layers,
                "Z_layers_proj": proj_layers,
                "Z_top_raw": Z_event_top,
            }

        else:
            y_event = x_norm.new_zeros(B, self.pred_len, self.channels)
            reg_proto_time = {"proto_div": zero, "proto_usage": zero}
            reg_tower_time = {"e2e": zero, "event_div": zero, "tower_hier": zero}
            tower_time = None

        # ===== 7) 三分支融合：concat+MLP =====
        y_lin_use = y_lin if use_linear else x_norm.new_zeros(B, self.pred_len, self.channels)
        y_nodes_use = y_nodes if use_node else x_norm.new_zeros(B, self.pred_len, self.channels)
        y_event_use = y_event if use_event else x_norm.new_zeros(B, self.pred_len, self.channels)

        y_cat = torch.cat([y_lin_use, y_nodes_use, y_event_use], dim=-1)   # [B,pred,3C]
        y_time = self.fuse_mlp(y_cat)                                      # [B,pred,C]

        with torch.no_grad():
            self._last_branch_stats = {
                "y_lin_mean_abs": y_lin.detach().abs().mean().item(),
                "y_nodes_mean_abs": y_nodes.detach().abs().mean().item() if use_node else 0.0,
                "y_event_mean_abs": y_event.detach().abs().mean().item() if use_event else 0.0,
                "y_fuse_mean_abs": y_time.detach().abs().mean().item(),
            }

        y_time = self.Linear_Tran(y_time.permute(0, 2, 1)).permute(0, 2, 1)

        # ===== 8) 反标准化 =====
        y = y_time * std_enc + mean_enc

        # ===== 9) reg 归档（保持你现有键）=====
        loss_e2e_time = reg_tower_time.get("e2e", zero)
        loss_event_div = reg_tower_time.get("event_div", zero)
        loss_tower_hier = reg_tower_time.get("tower_hier", zero)

        loss_e2e_smooth_time = zero
        loss_event_node = zero
        loss_tower_content = zero

        loss_proto_div = reg_proto_time.get("proto_div", zero)
        loss_proto_usage = reg_proto_time.get("proto_usage", zero)

        reg = {
            'e2e': self.w_e2e * loss_e2e_time,
            'gp': self.w_gp * zero,
            'hyper': zero,
            'ee_smooth': self.w_ee_smooth * loss_e2e_smooth_time,

            'const_node_time': loss_const_time if use_node else zero,
            'const_time': loss_node_const_time if use_node else zero,

            'event_div': loss_event_div if use_event else zero,
            'event_node': loss_event_node if use_event else zero,
            'tower_hier': loss_tower_hier if use_event else zero,

            'proto_div': loss_proto_div if use_event else zero,
            'proto_usage': loss_proto_usage if use_event else zero,

            'tower_content': (self.w_event_content * loss_tower_content) if use_event else zero,
        }

        # ============================================================
        #  eval/test：统一导出 hg_store（关键新增调用点）
        # ============================================================
        if not self.training:
            # time_hg_pack：多尺度 HGConv 前/后 token（pre=conv输出，post=HGConv输出）
            time_hg_pack = {"scales": []}
            for i in range(n_expected):
                pre = seq_list[i]
                post = X_time_ms_list[i] if (use_node and i < len(X_time_ms_list)) else None
                time_hg_pack["scales"].append({"scale": i, "pre": pre, "post": post})

            # branch_pack：尽量不把超大 token 全量塞进去；这里对 context_tokens 做一个简单下采样
            branch_pack = {"y_lin": y_lin_use, "y_nodes": y_nodes_use, "y_event": y_event_use}
            if Z_nodes is not None and torch.is_tensor(Z_nodes) and Z_nodes.numel() > 0:
                branch_pack["Z_nodes"] = Z_nodes
            if decoded is not None and torch.is_tensor(decoded) and decoded.numel() > 0:
                branch_pack["decoded"] = decoded

            # 对 context_tokens / Z_event_all 做稀疏抽样（避免 hg_store 暴涨）
            max_tok = int(self.viz_max_pred or 256)
            if context_tokens is not None and torch.is_tensor(context_tokens) and context_tokens.numel() > 0 and context_tokens.dim() == 3:
                Bt, Tt, Dt = context_tokens.shape
                if Tt > max_tok:
                    idx = torch.linspace(0, Tt - 1, steps=max_tok, device=context_tokens.device).long()
                    branch_pack["context_tokens"] = context_tokens.index_select(1, idx)
                    branch_pack["context_tokens_idx"] = idx
                else:
                    branch_pack["context_tokens"] = context_tokens
                    branch_pack["context_tokens_idx"] = torch.arange(Tt, device=context_tokens.device)

            if Z_event_all is not None and torch.is_tensor(Z_event_all) and Z_event_all.numel() > 0:
                branch_pack["Z_event_all"] = Z_event_all

            self._append_hg_store(
                A_np=A_np,
                tower_time=tower_time,
                content_feat_time=content_feat_time,
                branch_pack=branch_pack,
                time_hg_pack=time_hg_pack,
                tower_token_pack=tower_token_pack,
            )

        return y, reg


