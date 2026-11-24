# -*- coding: utf-8 -*-
"""
models/sthyper/model/st_hyper_model.py
Version: v0.6.7  (2025-12-31)

【changelog】
- v0.6.7 [Add][HG-Store-Export]
  1) 新增 Model._append_hg_store()（仅 eval/test 生效）作为唯一导出入口：
     - ProtoMemory：A_np 的 TopK + incidence(mask/weight)；
     - EventTower：tower_base（base 仅存一次）+ tower（batch 每次追加）+ tower_repr（pooled repr）；
     - 新增 tower_meta：收集 mu_span/mu_hf，以及（若存在）layers_export 中 diffusion(alpha,K) 与 stats；
     - Content：保存 content_feat_time（span/hf）；
     - Branches：保存三分支输出与关键中间张量（用于 CKA/Probe/复现）；
     - Time_hg：保存每尺度 HGConv pre/post token 的 pool + subsample tokens；
     - Tower_tokens：保存事件塔各层 token（raw/proj）及 pooled mean。
  2) 增强健壮性：兼容旧 hg_store key 缺失/长度不一致；所有导出 Tensor 均 detach+cpu+clone。
  3) 新增 get_hg_store/reset_hg_store 两个轻量接口，便于测试脚本落盘保存与复现。

- v0.6.6 [TimeHG-Sanitize]
  1) 新增静态方法 _sanitize_time_scale，用于集中清洗 AHL 输出的单尺度超图 (M,E,mask)：
     • 严格检查节点/超边索引是否越界；
     • 按 min_edge_card (默认 2) 过滤节点数过少的超边；
     • 对保留超边做连续重编号，统一输出合法 (M_i, E_i, mask_clean)。
  2) 在 forward 时间分支 2.3 处统一调用 _sanitize_time_scale，替换原本分散的 mask 检查逻辑，
     保证后续 HypergraphConv / _edge_centers_batched / LocalRefine 始终使用“干净”超图。
  3) 新增配置参数 t_min_edge_card、t_use_ring_fallback，控制时间分支超边最小基数与是否使用 ring 兜底构图。

- v0.6.5 [EventContentReg-Wiring]
  1) 为时间事件塔 SemanticEventTowerV2 完成 span / hf “内容层级约束”的主干接线：
     • 在 ProtoMemoryHypergraph 之后，基于“大时间图”节点特征 Xn_time_union 与节点→原型注意力 A_np，
       构造每个原型事件的 content_feat[b,k,:] = [span, hf_ratio]。
  2) 新增全局权重 w_event_content，用于控制 tower_content loss 在总 loss 中的占比。
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

# 事件塔：ProtoMemory + SemanticEventTowerV2
from .e2e.e2e_prop import ProtoMemoryHypergraph, SemanticEventTowerV2


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

        # 保证至少 min_card 成员
        while len(nodes) < min_card:
            nodes.append((nodes[-1] + 1) % M if nodes else (e % M))

        # 去重并限制不要超过 M
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
            （当前版本的事件塔路径已改为 ProtoMemory + SemanticTower，
             默认不再使用该项，可保留兼容。）
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
        mask : [2, P]  (node_idx, edge_idx)

        return: edge_centers [B, E, C]
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

        # CPU 索引检查，避免 CUDA assert
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

        edge_centers = X.new_zeros(B, E, C)  # [B, E, C]
        for b in range(B):
            x_b = X[b]  # [M, C]
            x_sel = x_b.index_select(0, n_idx)  # [P, C]

            agg = x_b.new_zeros(E, C)
            agg.index_add_(0, e_idx, x_sel)

            cnt = torch.bincount(e_idx, minlength=E).clamp(min=1)
            cnt = cnt.to(agg.dtype).unsqueeze(-1)

            edge_centers[b] = agg / cnt

        return edge_centers

    # ===== 单尺度时间超图清洗统一逻辑 =====
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
                    f"[TimeBranch][Scale {scale_idx}] AHL 输出的 M={M_raw} 与 特征 M_feat={M_i} 不一致。"
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
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回的 mask 为空或非法：M_feat={M_i}, E_conf={E_conf}。")

        mask = mask.to(device=device, dtype=torch.long)
        if mask.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回空超图：M_feat={M_i}, E_conf={E_conf}。")

        if mask.dim() != 2 or mask.size(0) != 2:
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] AHL 返回的 mask 形状非法：期望 [2,P]，实际为 {tuple(mask.shape)}。")

        node_idx = mask[0]
        edge_idx = mask[1]

        if edge_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long,
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
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            n_min = int(node_idx.min().item()); n_max = int(node_idx.max().item())
            e_min = int(edge_idx.min().item()); e_max = int(edge_idx.max().item())
            raise RuntimeError(
                f"[TimeBranch][Scale {scale_idx}] AHL 输出的所有 membership 均越界："
                f"M_feat={M_i}, E_conf={E_conf}, E_i(after align)={E_i}, "
                f"node_idx范围=[{n_min},{n_max}], edge_idx范围=[{e_min},{e_max}]。"
            )

        node_idx = node_idx[valid]
        edge_idx = edge_idx[valid]
        if node_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=max(1, int(min_edge_card)),
                    device=device, dtype=torch.long,
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
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 所有超边的节点数都小于 min_edge_card={min_edge_card}。")

        keep_flags = keep_edges[edge_idx]
        node_idx = node_idx[keep_flags]
        edge_idx = edge_idx[keep_flags]

        if node_idx.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 按 min_edge_card 过滤后 membership 为空。")

        uniq_edges, new_eidx = torch.unique(edge_idx, return_inverse=True)
        E_eff = int(uniq_edges.numel())
        if E_eff <= 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long,
                )
                return {'M': M_i, 'E': int(E_i_conf), 'mask': mask_ring.to(device), 'prob_val': prob_val}
            raise RuntimeError(f"[TimeBranch][Scale {scale_idx}] 清洗后得到 E_eff={E_eff}。")

        mask_clean = torch.stack(
            [node_idx, new_eidx.to(device=device, dtype=torch.long)],
            dim=0,
        ).contiguous()

        if mask_clean.numel() == 0:
            if use_ring_fallback and E_i_conf > 0:
                mask_ring = _build_segment_ring_mask(
                    M=M_i, E=E_i_conf, min_card=min_edge_card,
                    device=device, dtype=torch.long,
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
        device = Xi.device
        B, M_feat, C = Xi.shape
        if M_feat != M_i:
            raise RuntimeError(f"[LocalHG][Scale {scale_idx}] M_feat={M_feat} 与 M_i={M_i} 不一致。")

        if base_mask is None or base_mask.numel() == 0:
            return Xi, Xi.new_zeros(())

        base_mask = base_mask.to(device=device, dtype=torch.long)

        edge_centers = Model._edge_centers_batched(Xi, base_mask, M_i, E_i)

        if self.edge_refiners is not None and scale_idx < len(self.edge_refiners):
            refiner = self.edge_refiners[scale_idx]
            gate_raw = refiner(edge_centers)
            gate = torch.sigmoid(gate_raw).squeeze(-1)
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
            diff = (Xn_time_union[:, 1:, :] - Xn_time_union[:, :-1, :]).pow(2).mean(dim=-1)
            pad = diff.new_zeros(B, 1)
            diff = torch.cat([diff, pad], dim=1)
        else:
            diff = Xn_time_union.new_zeros(B, M_union)

        A_np32 = A_np.to(dtype)
        denom = A_np32.sum(dim=1, keepdim=True).clamp_min(1e-12)
        w = A_np32 / denom

        E_t = torch.einsum("bmk,m->bk", w, pos)
        E_t2 = torch.einsum("bmk,m->bk", w, pos2)
        var_t = (E_t2 - E_t.pow(2)).clamp_min(0.0)
        span = torch.sqrt(var_t + 1e-12)

        hf_raw = torch.einsum("bmk,bm->bk", w, diff)
        mean_hf = hf_raw.mean(dim=1, keepdim=True).clamp_min(1e-6)
        hf = hf_raw / mean_hf

        return torch.stack([span, hf], dim=-1)

    def _viz_reset(self):
        self._viz_H0 = None
        self._viz_H_layers = []
        self._viz_enabled_this_forward = False
        self._viz_time_stats = []
        self._viz_spatial_stats = []
        self._last_edge_gates = []

    def __init__(self, configs):
        super().__init__()

        # ================== 通用基础参数 ==================
        self.seq_len = _as_int(getattr(configs, 'seq_len', 96), 96)
        self.pred_len = _as_int(getattr(configs, 'pred_len', 96), 96)
        self.channels = _as_int(getattr(configs, 'enc_in', 7), 7)
        self.individual = bool(getattr(configs, 'individual', True))

        self.dropout = float(getattr(configs, 'dropout', 0.1))

        # ========= cycle_index 位置编码（可选） =========
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

        # ========== GTS & 正则项 ==========
        self.global_tri_stage = bool(getattr(configs, 'global_tri_stage', False))

        def _get_float(name, default):
            raw_val = getattr(configs, name, default)
            try:
                return float(raw_val)
            except (TypeError, ValueError):
                return default

        self.w_e2e = _get_float('w_e2e', 1e-1)
        self.w_gp = _get_float('w_gp', 5e-1)
        self.w_e2e_sp = _get_float('w_e2e_sp', self.w_e2e)
        self.w_gp_sp = _get_float('w_gp_sp', self.w_gp)
        self.w_sp = _get_float('w_sp', 1e-1)
        self.w_sp_sp = _get_float('w_sp_sp', self.w_sp)
        self.w_ee_smooth = _get_float('w_ee_smooth', 1e-1)
        self.w_ee_smooth_sp = _get_float('w_ee_smooth_sp', 1e-1)
        self.w_gts_smooth = _get_float('w_gts_smooth', 2e-1)
        self.w_event_content = _get_float('w_event_content', 0.3)

        raw_w_fusion_ent = getattr(configs, 'w_fusion_entropy', getattr(configs, 'w_aid_entropy', 1e-1))
        self.w_fusion_entropy = float(raw_w_fusion_ent)

        # ======================= 时间分支 =======================

        ctor_name = getattr(configs, 't_cscm', getattr(configs, 'CSCM', 'Bottleneck_Construct'))
        ctor = CONSTRUCTORS.get(ctor_name)
        if ctor is None:
            raise NameError(f'Unknown CSCM={ctor_name}. Choose from {list(CONSTRUCTORS.keys())}')

        raw_window_size = getattr(configs, 't_window_size', None)
        if isinstance(raw_window_size, int):
            raw_window_size = [int(raw_window_size)]
        else:
            raw_window_size = [int(x) for x in raw_window_size]
        self.window_size = raw_window_size

        all_size = get_mask(self.seq_len, self.window_size)
        if isinstance(all_size, (list, tuple)):
            self.all_size = [int(m) for m in all_size]
        else:
            self.all_size = [int(all_size)]
        if len(self.all_size) == 0:
            raise ValueError(f'[STHyperModel][TimeBranch] get_mask 返回空列表：seq_len={self.seq_len}, window_size={self.window_size}')

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
                f'请检查 seq_len={self.seq_len}, t_window_size={self.window_size}, t_hyper_num={raw_hyper_num}。'
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

        _raw_gamma_time = getattr(configs, 't_gamma_dist', 0.2)
        _raw_lambda_nc_time = getattr(configs, 'lambda_nc_time', getattr(configs, 'lambda_nc', 0.5))
        _raw_heads_time = getattr(configs, 'hyper_heads_time', 4)
        _raw_neg_slope_time = getattr(configs, 'hyper_negative_slope_time', 0.2)
        _raw_dropout_time = getattr(configs, 'hyper_dropout_time', 0.3)

        heads_time = int(_raw_heads_time)
        neg_slope_time = float(_raw_neg_slope_time)
        dropout_time = float(_raw_dropout_time)
        gamma_time = float(_raw_gamma_time)
        lambda_nc_time = float(_raw_lambda_nc_time)
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

        # ================== ProtoMemory + 语义事件塔 + QueryDecoder ==================

        D_event = getattr(configs, "event_dim", self.channels)
        self.event_dim = int(D_event)

        self.proto_time = ProtoMemoryHypergraph(
            D_in=self.t_d_model,
            D_mem=self.event_dim,
            num_protos=int(getattr(configs, "num_event_protos", 32)),
            dropout=float(getattr(configs, "event_dropout", self.dropout)),
            topk_per_node=int(getattr(configs, "proto_topk", 8)),
            w_proto_div=float(getattr(configs, "w_proto_div", 0.3)),
            w_proto_usage=float(getattr(configs, "w_proto_usage", 0.3)),
        )

        self.event_tower_time = SemanticEventTowerV2(
            configs=configs,
            D=self.event_dim,
            num_layers=int(getattr(configs, "event_layers", 2)),
            topk=int(getattr(configs, "event_topk", 16)),
            dropout=float(getattr(configs, "event_dropout", self.dropout)),
            expand=float(getattr(configs, "event_expand", 2.0)),
            ffn_ratio=float(getattr(configs, "event_ffn_ratio", 2.0)),
            cheby_order=int(getattr(configs, "cheby_order", 2)),
            num_heads=int(getattr(configs, "event_heads", 4)),
            mem_slots=int(getattr(configs, "event_mem_slots", 16)),
            w_div=float(getattr(configs, "w_event_div", 0.1)),
            w_node=float(getattr(configs, "w_event_node", 0.3)),
            meta_dim=0,
            meta_hidden=32,
            gate_strength=float(getattr(configs, "event_gate_strength", 1.0)),
            w_hier=float(getattr(configs, "w_event_hier", 1.0)),
            use_event_propagation=bool(getattr(configs, "use_event_propagation", True)),
            use_content_reg=bool(getattr(configs, "use_content_reg", True)),
            content_feat_dim=int(getattr(configs, "content_feat_dim", 2)),
            content_w_span=float(getattr(configs, "content_w_span", 1.0)),
            content_w_hf=float(getattr(configs, "content_w_hf", 1.0)),
            content_margin_span=float(getattr(configs, "content_margin_span", 0.3)),
            content_margin_hf=float(getattr(configs, "content_margin_hf", 0.3)),
        )

        if self.event_dim != self.t_d_model:
            self.event_proj = nn.Linear(self.event_dim, self.t_d_model)
        else:
            self.event_proj = None

        self.token_type_embed = nn.Embedding(2, self.t_d_model)

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

        # ================== 基线 + 节点预测头 ==================
        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.node_out_proj = nn.Linear(self.t_d_model, self.channels)
        self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        self._last_branch_stats = {}
        self.collect_node_stats = bool(getattr(configs, "collect_node_stats", True))
        self._last_node_stats = {}

        self.use_event = bool(getattr(configs, 'use_event', True))
        self.use_node = bool(getattr(configs, 'use_node', True))
        self.use_linear = bool(getattr(configs, 'use_linear', True))

        # ===== [NEW][v0.6.7] 导出缓存容器（eval/test）=====
        self._hg_store = None

    # ================== HG store 接口 ==================
    @torch.no_grad()
    def reset_hg_store(self):
        self._hg_store = None

    @torch.no_grad()
    def get_hg_store(self, clear: bool = False):
        out = self._hg_store
        if clear:
            self._hg_store = None
        return out

    # ================== 导出入口（仅 eval/test） ==================
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
        Version: v0.8.2  (2025-12-29)

        约束：只用这个函数做所有导出（eval/test）。
        """
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

        # 1) ProtoMemory (A_np)
        if A_np is not None and torch.is_tensor(A_np) and A_np.numel() > 0:
            A = A_np
            Bp, M, K = A.shape
            k_node = int(getattr(self.proto_time, "topk_per_node", 8))
            k_node = max(1, min(k_node, K))

            topv, topi = torch.topk(A, k=k_node, dim=-1)

            b_idx = torch.arange(Bp, device=A.device).view(Bp, 1, 1).expand(Bp, M, k_node).reshape(-1)
            n_idx = torch.arange(M, device=A.device).view(1, M, 1).expand(Bp, M, k_node).reshape(-1)
            e_idx = topi.reshape(-1)
            w = topv.reshape(-1)

            keep = w > 0
            mask = torch.stack([b_idx[keep], n_idx[keep], e_idx[keep]], dim=0).long()
            weight = w[keep]

            self._hg_store["proto"].append({
                "shape": (int(Bp), int(M), int(K)),
                "topi": _cpu(topi, torch.int32),
                "topv": _cpu(topv, torch.float32),
                "incidence_mask": _cpu(mask, torch.int32),
                "incidence_weight": _cpu(weight, torch.float32),
            })
        else:
            self._hg_store["proto"].append(None)

        # 2) EventTower：结构 + repr + meta
        A_base_list = None
        A_batch_list = None

        repr_layers_mean = None
        repr_layers_max = None
        repr_mean_mask = None
        repr_max_mask = None
        layer_ids = None

        mu_span = None
        mu_hf = None
        layers_export = None

        if isinstance(tower_time, dict):
            A_base_list = tower_time.get("A_base_list", None)
            A_batch_list = tower_time.get("A_batch_list", None)

            mu_span = tower_time.get("mu_span", None)
            mu_hf = tower_time.get("mu_hf", None)

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

        # 2.1 tower_base/tower（TopK incidence）
        if isinstance(A_base_list, (list, tuple)) and len(A_base_list) > 0:
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

                # 2.2 layers_export: diffusion/stats（若存在）
                if isinstance(layers_export, (list, tuple)) and l < len(layers_export):
                    le = layers_export[l]
                    if isinstance(le, dict):
                        export_pack = {}

                        diff = le.get("diffusion", None)
                        if isinstance(diff, dict):
                            export_pack["diffusion"] = {
                                "K": int(diff.get("K", -1)) if diff.get("K", None) is not None else None,
                                "alpha": _cpu(diff.get("alpha", None), torch.float32) if torch.is_tensor(diff.get("alpha", None)) else None,
                            }

                        stats = le.get("stats", None)
                        if isinstance(stats, dict):
                            stats_pack = {}
                            for kk, vv in stats.items():
                                if torch.is_tensor(vv):
                                    stats_pack[kk] = _cpu(vv, torch.float32)
                                else:
                                    stats_pack[kk] = vv
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

        # 2.3 repr（pooled）
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

        # 2.4 tower_meta：mu_span/mu_hf
        meta_pack = None
        if mu_span is not None or mu_hf is not None:
            meta_pack = {}
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

        self._hg_store["tower_meta"].append(meta_pack)

        # 3) Content feat (span/hf)
        if content_feat_time is not None and torch.is_tensor(content_feat_time) and content_feat_time.numel() > 0:
            self._hg_store["content"].append({
                "shape": tuple(content_feat_time.shape),
                "content_feat": _cpu(content_feat_time, torch.float32),
            })
        else:
            self._hg_store["content"].append(None)

        # 4) Branch pack
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

        # 5) Time HG pack：pre/post HGConv 的节点 token（多尺度）
        if isinstance(time_hg_pack, dict) and len(time_hg_pack) > 0:
            max_tokens = int(getattr(self, "viz_max_pred", 256) or 256)
            pack = {"scales": []}
            scales = time_hg_pack.get("scales", None)
            if isinstance(scales, (list, tuple)):
                for item in scales:
                    if not isinstance(item, dict):
                        continue
                    Xi_pre = item.get("pre", None)
                    Xi_post = item.get("post", None)
                    scale_i = item.get("scale", None)

                    entry = {"scale": int(scale_i) if scale_i is not None else None}

                    if torch.is_tensor(Xi_pre) and Xi_pre.numel() > 0:
                        entry["pre_shape"] = tuple(Xi_pre.shape)
                        entry["pre_pool_mean"] = _cpu(Xi_pre.mean(dim=1), torch.float32)
                        entry["pre_pool_max"] = _cpu(Xi_pre.max(dim=1).values, torch.float32)
                        Xs, idx = _subsample_tokens(Xi_pre, max_tokens=max_tokens)
                        entry["pre_tokens"] = _cpu(Xs, torch.float32) if Xs is not None else None
                        entry["pre_tokens_idx"] = _cpu(idx, torch.int32) if idx is not None else None
                    else:
                        entry.update({"pre_shape": None, "pre_pool_mean": None, "pre_pool_max": None,
                                      "pre_tokens": None, "pre_tokens_idx": None})

                    if torch.is_tensor(Xi_post) and Xi_post.numel() > 0:
                        entry["post_shape"] = tuple(Xi_post.shape)
                        entry["post_pool_mean"] = _cpu(Xi_post.mean(dim=1), torch.float32)
                        entry["post_pool_max"] = _cpu(Xi_post.max(dim=1).values, torch.float32)
                        Xs, idx = _subsample_tokens(Xi_post, max_tokens=max_tokens)
                        entry["post_tokens"] = _cpu(Xs, torch.float32) if Xs is not None else None
                        entry["post_tokens_idx"] = _cpu(idx, torch.int32) if idx is not None else None
                    else:
                        entry.update({"post_shape": None, "post_pool_mean": None, "post_pool_max": None,
                                      "post_tokens": None, "post_tokens_idx": None})

                    pack["scales"].append(entry)

            self._hg_store["time_hg"].append(pack)
        else:
            self._hg_store["time_hg"].append(None)

        # 6) Tower tokens：事件塔各层 token（raw/proj）
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

    # ================== Debug/Stats 接口（保持不变） ==================
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

        # ===== 1. 标准化 =====
        mean_enc = x.mean(1, keepdim=True).detach()
        x_norm = x - mean_enc
        std_enc = torch.sqrt(torch.var(x_norm, dim=1, keepdim=False).mean(dim=1, keepdim=True) + 1e-5).detach()
        std_enc = std_enc.unsqueeze(1)
        x_norm = x_norm / std_enc

        B, L, Cref = x_norm.shape
        device = x_norm.device
        zero = x_norm.new_zeros(())

        # ===== 1.x cycle_index 位置注入 =====
        if self.use_cycle_index and (cycle_index is not None) and (self.cycle_proj is not None):
            if not torch.is_tensor(cycle_index):
                cycle_index = torch.as_tensor(cycle_index, device=device)
            else:
                cycle_index = cycle_index.to(device)

            while cycle_index.dim() > 2 and cycle_index.size(-1) == 1:
                cycle_index = cycle_index.squeeze(-1)

            if cycle_index.dim() == 1:
                if cycle_index.size(0) != B:
                    raise RuntimeError(f"[STHyperModel] cycle_index 形状 {tuple(cycle_index.shape)} 与 B={B} 不匹配")
                cycle_index = cycle_index.view(B, 1).expand(B, L)
            elif cycle_index.dim() == 2:
                if cycle_index.size(0) != B:
                    raise RuntimeError(f"[STHyperModel] cycle_index 形状 {tuple(cycle_index.shape)} 与 B={B} 不匹配")
                if cycle_index.size(1) == 1:
                    cycle_index = cycle_index.expand(B, L)
                elif cycle_index.size(1) != L:
                    raise RuntimeError(f"[STHyperModel] cycle_index 期望 [B,L]=({B},{L})，当前为 {tuple(cycle_index.shape)}")
            else:
                raise RuntimeError(f"[STHyperModel] cycle_index 维度 {cycle_index.dim()} 非法，期望 1D/2D")

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

            cycle_embed = self.cycle_proj(feat)
            x_norm = x_norm + cycle_embed

        use_linear = bool(getattr(self, "use_linear", True))
        use_node = bool(getattr(self, "use_node", True))
        use_event = bool(getattr(self, "use_event", True))
        if (not use_linear) and (not use_node) and (not use_event):
            use_linear = True

        # ===== Linear 基线 =====
        if self.individual:
            y_lin = x_norm.new_zeros(B, self.pred_len, self.channels)
            for i in range(self.channels):
                y_lin[:, :, i] = self.Linear[i](x_norm[:, :, i])
        else:
            tmp = self.Linear(x_norm.permute(0, 2, 1))
            y_lin = tmp.permute(0, 2, 1)

        if (not use_node) and (not use_event):
            y_time = y_lin if use_linear else x_norm.new_zeros(B, self.pred_len, self.channels)
            with torch.no_grad():
                self._last_branch_stats = {
                    "y_lin_mean_abs": y_lin.detach().abs().mean().item(),
                    "y_nodes_mean_abs": 0.0,
                    "y_event_mean_abs": 0.0,
                }
            y_time_T = y_time.permute(0, 2, 1)
            y_time_T = self.Linear_Tran(y_time_T)
            y_time = y_time_T.permute(0, 2, 1)
            y = y_time * std_enc + mean_enc
            reg = {
                'e2e': zero, 'gp': zero, 'hyper': zero, 'ee_smooth': zero,
                'const_node_time': zero, 'const_time': zero,
                'event_div': zero, 'event_node': zero, 'tower_hier': zero,
                'proto_div': zero, 'proto_usage': zero, 'tower_content': zero,
            }
            return y, reg

        # ===== 2. 时间分路：多尺度特征 + 外部构图(AHL) =====
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

        # 2.3 单尺度清洗
        adj_info = []
        min_edge_card = int(getattr(self, "t_min_edge_card", 2))
        use_ring_fallback = bool(getattr(self, "t_use_ring_fallback", False))

        for i in range(n_expected):
            Xi = seq_list[i]
            M_feat = Xi.size(1)
            E_conf = int(self.hyper_num1[i])
            info_i = Model._sanitize_time_scale(
                M_feat=M_feat,
                E_conf=E_conf,
                raw_info=adj_info_ext[i],
                min_edge_card=min_edge_card,
                use_ring_fallback=use_ring_fallback,
                scale_idx=i,
                device=device,
            )
            adj_info.append(info_i)

        # ===== 节点分支 =====
        X_time_ms_list = []
        loss_const_time = x_norm.new_zeros(())
        time_hg_scales = []  # [NEW] 用于 time_hg_pack

        if use_node:
            for i in range(n_expected):
                Xi_pre = seq_list[i]
                info_i = adj_info[i]
                M_i, E_i = int(info_i['M']), int(info_i['E'])
                m = info_i['mask']
                if M_i <= 0 or E_i <= 0 or m is None or m.numel() == 0:
                    raise RuntimeError(f"[TimeBranch][Scale {i}] 清洗后超图非法：M={M_i}, E={E_i}")
                m = m.to(device)

                if self.use_local_hg_refine:
                    Xi_post, loss_i = self._hyperconv_with_local_refine(
                        self.time_hgconvs[i], Xi_pre, m, M_i, E_i, scale_idx=i
                    )
                else:
                    Xi_post, loss_i = self.time_hgconvs[i](Xi_pre, m)

                X_time_ms_list.append(Xi_post)
                loss_const_time = loss_const_time + loss_i

                time_hg_scales.append({"scale": i, "pre": Xi_pre, "post": Xi_post})

            # 节点统计（保持你的原逻辑）
            if self.collect_node_stats and len(X_time_ms_list) > 0:
                per_node_energy_ms, per_node_var_ms, per_node_meanabs_ms, per_node_diff_ms = [], [], [], []
                scale_mean_abs, scale_std, scale_max_abs, scale_l2, scale_diff_energy = [], [], [], [], []

                for i, Xi_conv in enumerate(X_time_ms_list):
                    B_i, M_i, C_i = Xi_conv.shape
                    energy_i = Xi_conv.pow(2).mean(dim=-1, keepdim=True)
                    var_i = Xi_conv.var(dim=-1, unbiased=False, keepdim=True)
                    meanabs_i = Xi_conv.abs().mean(dim=-1, keepdim=True)
                    if M_i > 1:
                        diff_i = (Xi_conv[:, 1:, :] - Xi_conv[:, :-1, :]).pow(2).mean(dim=-1, keepdim=True)
                        pad = Xi_conv.new_zeros(B_i, 1, 1)
                        diff_i = torch.cat([diff_i, pad], dim=1)
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
                    diff_energy_i = (Xi_conv[:, 1:, :] - Xi_conv[:, :-1, :]).pow(2).mean().item() if M_i > 1 else 0.0
                    scale_diff_energy.append(diff_energy_i)

                self._last_node_stats = {
                    "per_node": {
                        "energy": per_node_energy_ms,
                        "var": per_node_var_ms,
                        "mean_abs": per_node_meanabs_ms,
                        "diff_energy": per_node_diff_ms,
                    },
                    "per_scale": {
                        "mean_abs": scale_mean_abs,
                        "std": scale_std,
                        "max_abs": scale_max_abs,
                        "l2": scale_l2,
                        "diff_energy": scale_diff_energy,
                    },
                }
            else:
                self._last_node_stats = {}

            # 多尺度节点拼接 -> Z_nodes
            X_ms_nodes_concat = torch.cat(X_time_ms_list, dim=1) if len(X_time_ms_list) > 0 else x_norm[:, :0, :]
            if X_ms_nodes_concat.size(1) > 0:
                nodes_concat_T = X_ms_nodes_concat.permute(0, 2, 1)
                if nodes_concat_T.size(-1) != self.Ms_length:
                    if nodes_concat_T.size(-1) < self.Ms_length:
                        nodes_concat_T = F.pad(nodes_concat_T, (0, self.Ms_length - nodes_concat_T.size(-1)))
                    else:
                        nodes_concat_T = F.adaptive_avg_pool1d(nodes_concat_T, self.Ms_length)
                x_out_time_pred_ms = self.out_tran(nodes_concat_T)
                Z_nodes = x_out_time_pred_ms.permute(0, 2, 1)
            else:
                Z_nodes = x_norm.new_zeros(B, self.pred_len, self.t_d_model)

            y_nodes = self.node_out_proj(Z_nodes)
            loss_node_const_time = loss_const_time
        else:
            self._last_node_stats = {}
            Z_nodes = x_norm.new_zeros(B, self.pred_len, self.t_d_model)
            y_nodes = x_norm.new_zeros(B, self.pred_len, self.channels)
            loss_const_time = zero
            loss_node_const_time = zero

        # ===== union 节点（事件分支）=====
        Xn_time_union_list, mask_parts = [], []
        node_base, edge_offset, E_time_init = 0, 0, 0
        for i in range(n_expected):
            Xi = seq_list[i]
            info = adj_info[i]
            M_i, E_i = info['M'], info['E']
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
        M_time_union = int(sum(adj_info[i]['M'] for i in range(n_expected))) if n_expected > 0 else 0

        # ===== 事件分支 =====
        A_np = None
        content_feat_time = None
        tower_time = None
        decoded = None
        context_tokens = None
        Z_event_all = None
        context_tokens_idx = None

        if use_event and (M_time_union > 0):
            Z_proto, reg_proto_time, extra_proto_time = self.proto_time(Xn_time_union, mask_valid=None, return_sparse=False)
            A_np = extra_proto_time.get("A_np", None)

            if (
                    getattr(self.event_tower_time, "use_content_reg", False)
                    and Z_proto.size(1) > 0
                    and A_np is not None
                    and A_np.numel() > 0
            ):
                content_feat_time = self._build_event_content_feat(Xn_time_union, A_np)

            if Z_proto.size(1) > 0:
                Z_event_top, reg_tower_time, tower_time = self.event_tower_time(
                    Z_proto, meta=None, node_feat=None, mask=None, M=None, content_feat=content_feat_time
                )
            else:
                Z_event_top = Z_proto
                reg_tower_time = {
                    "e2e": zero, "ee_smooth": zero, "event_div": zero, "event_node": zero,
                    "tower_hier": zero, "tower_content": zero,
                }
                tower_time = {"Z_layers": [], "A_base_list": [], "A_batch_list": []}

            # 事件层拼接 -> Z_event_all，并同时准备导出 raw/proj tokens
            Z_layers_raw = tower_time.get("Z_layers", None) if isinstance(tower_time, dict) else None
            Z_layers_raw = list(Z_layers_raw) if isinstance(Z_layers_raw, (list, tuple)) else []
            Z_layers_proj = []

            if Z_proto.size(1) > 0:
                if len(Z_layers_raw) > 0:
                    proj_layers = []
                    for Z_l in Z_layers_raw:
                        Zp = Z_l
                        if Zp.size(-1) != self.t_d_model:
                            Zp = self.event_proj(Zp) if self.event_proj is not None else Zp[..., :self.t_d_model]
                        proj_layers.append(Zp)
                    Z_layers_proj = proj_layers
                    Z_event_all = torch.cat(proj_layers, dim=1)
                else:
                    Zp = Z_event_top
                    if Zp.size(-1) != self.t_d_model:
                        Zp = self.event_proj(Zp) if self.event_proj is not None else Zp[..., :self.t_d_model]
                    Z_event_all = Zp
            else:
                Z_event_all = x_norm.new_zeros(B, 0, self.t_d_model)

            # decoder memory：仅事件塔 token
            if Z_event_all.size(1) == 0:
                y_event = x_norm.new_zeros(B, self.pred_len, self.channels)
            else:
                type_event = torch.ones(1, dtype=torch.long, device=device)
                event_type_emb = self.token_type_embed(type_event)[0]

                context_tokens = Z_event_all + event_type_emb.view(1, 1, -1)

                query_tokens = self.query_embed.unsqueeze(0).expand(B, -1, -1)
                tgt = query_tokens.transpose(0, 1)
                memory = context_tokens.transpose(0, 1)

                decoded = self.query_decoder(tgt=tgt, memory=memory)
                decoded = decoded.transpose(0, 1)
                y_event = self.dec_out_proj(decoded)

            # ===== [NEW][v0.6.7] 事件塔 token 导出包 =====
            tower_token_pack = {
                "Z_layers_raw": Z_layers_raw if len(Z_layers_raw) > 0 else None,
                "Z_layers_proj": Z_layers_proj if len(Z_layers_proj) > 0 else None,
                "Z_top_raw": Z_event_top,
            }
        else:
            y_event = x_norm.new_zeros(B, self.pred_len, self.channels)
            reg_proto_time = {"proto_div": zero, "proto_usage": zero}
            reg_tower_time = {
                "e2e": zero, "ee_smooth": zero, "event_div": zero, "event_node": zero,
                "tower_hier": zero, "tower_content": zero,
            }
            tower_token_pack = None

        # ===== 融合 =====
        y_time = (y_lin if use_linear else 0.0) + (y_nodes if use_node else 0.0) + (y_event if use_event else 0.0)

        with torch.no_grad():
            self._last_branch_stats = {
                "y_lin_mean_abs": y_lin.detach().abs().mean().item(),
                "y_nodes_mean_abs": y_nodes.detach().abs().mean().item() if use_node else 0.0,
                "y_event_mean_abs": y_event.detach().abs().mean().item() if use_event else 0.0,
            }

        y_time_T = y_time.permute(0, 2, 1)
        y_time_T = self.Linear_Tran(y_time_T)
        y_time = y_time_T.permute(0, 2, 1)

        # ===== 反标准化 =====
        y = y_time * std_enc + mean_enc

        # ===== reg =====
        loss_gp = zero
        loss_hgc = zero

        loss_e2e_time = reg_tower_time.get("e2e", zero)
        loss_e2e_smooth_time = reg_tower_time.get("ee_smooth", zero)
        loss_event_div = reg_tower_time.get("event_div", zero)
        loss_event_node = reg_tower_time.get("event_node", zero)
        loss_tower_hier = reg_tower_time.get("tower_hier", zero)
        loss_tower_content = reg_tower_time.get("tower_content", zero)

        loss_proto_div = reg_proto_time.get("proto_div", zero)
        loss_proto_usage = reg_proto_time.get("proto_usage", zero)

        reg = {
            'e2e': self.w_e2e * loss_e2e_time,
            'gp': self.w_gp * loss_gp,
            'hyper': loss_hgc,
            'ee_smooth': self.w_ee_smooth * loss_e2e_smooth_time,

            'const_node_time': loss_node_const_time if use_node else zero,
            'const_time': loss_const_time if use_node else zero,

            'event_div': 0.5 * loss_event_div,
            'event_node': loss_event_node,
            'tower_hier': loss_tower_hier,
            'proto_div': loss_proto_div,
            'proto_usage': loss_proto_usage,

            # 你原代码这里是 0 * self.w_event_content * loss_tower_content（保持不动）
            'tower_content': 0 * self.w_event_content * loss_tower_content,
        }

        # ===== [NEW][v0.6.7] eval/test 导出（不新增任何 flag，沿用 viz_auto_eval）=====
        if (not self.training) and bool(getattr(self, "viz_auto_eval", True)):
            # time_hg_pack
            time_hg_pack = {"scales": time_hg_scales} if (use_node and len(time_hg_scales) > 0) else None

            # branch_pack：保证可复现/可做 CKA
            branch_pack = {
                "y_lin": y_lin,
                "y_nodes": y_nodes if use_node else None,
                "y_event": y_event if use_event else None,
                "Z_nodes": Z_nodes if use_node else None,
                "decoded": decoded if (decoded is not None and torch.is_tensor(decoded)) else None,
                "context_tokens": context_tokens if (context_tokens is not None and torch.is_tensor(context_tokens)) else None,
                "Z_event_all": Z_event_all if (Z_event_all is not None and torch.is_tensor(Z_event_all)) else None,
                "context_tokens_idx": context_tokens_idx,
            }

            self._append_hg_store(
                A_np=A_np,
                tower_time=tower_time,
                content_feat_time=content_feat_time,
                branch_pack=branch_pack,
                time_hg_pack=time_hg_pack,
                tower_token_pack=tower_token_pack,
            )

        return y, reg
