# -*- coding: utf-8 -*-
"""
models/sthyper/core/utils.py  v0.1.0  (2025-11-03)

【版本号与修改记录（changelog）】
- [Add] 从单文件迁移基础工具：_as_int/_as_float/get_mask/_resize_feat_last_dim/_block_diag/_smoothness_loss。
- [Keep] 行为与原实现一致；仅整理到独立工具模块。
"""
import math
import torch
import torch.nn.functional as F

def _as_int(v, default=None):
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else default
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            v = v.reshape(-1)[0] if v.size > 0 else default
    except Exception:
        pass
    if isinstance(v, str):
        try: return int(float(v.strip()))
        except Exception: return int(default) if default is not None else 0
    try: return int(v)
    except Exception:
        try: return int(float(v))
        except Exception: return int(default) if default is not None else 0

def _as_float(v, default=None):
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else default
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            v = v.reshape(-1)[0] if v.size > 0 else default
    except Exception:
        pass
    if isinstance(v, str):
        try: return float(v.strip())
        except Exception: return float(default) if default is not None else 0.0
    try: return float(v)
    except Exception: return float(default) if default is not None else 0.0

def get_mask(input_size, window_size):
    all_size = [input_size]
    for w in window_size:
        all_size.append(math.floor(all_size[-1] / w))
    return all_size

def _resize_feat_last_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    if t.size(-1) == target or t.dim() != 3:
        return t
    B, ME, C = t.size()
    tmp = t.reshape(B * ME, 1, C)
    tmp = F.interpolate(tmp, size=target, mode='linear', align_corners=False)
    return tmp.reshape(B, ME, target).to(t.dtype)

def _block_diag(mats, device=None, dtype=None):
    mats = [m for m in mats if m is not None and m.numel() > 0]
    if len(mats) == 0:
        return None
    sizes = [m.shape[0] for m in mats]
    E_total = int(sum(sizes))
    device = device or mats[0].device
    dtype  = dtype  or mats[0].dtype
    out = torch.zeros(E_total, E_total, device=device, dtype=dtype)
    offset = 0
    for Mi in mats:
        e = Mi.shape[0]
        out[offset:offset+e, offset:offset+e] = Mi
        offset += e
    out = out / (out.sum(-1, keepdim=True).clamp_min(1e-12))
    return out

def _smoothness_loss(feat: torch.Tensor, Arow: torch.Tensor):
    if feat.numel() == 0 or Arow.numel() == 0:
        return feat.new_zeros(())
    A = 0.5 * (Arow + Arow.transpose(-1, -2))
    L = torch.eye(A.size(-1), device=A.device, dtype=A.dtype) - A  # I - A
    diff = torch.matmul(L, feat)
    num = (diff * diff).sum(dim=(-2, -1))
    den = (feat * feat).sum(dim=(-2, -1)).clamp_min(1e-6)
    return (num / den).mean()
