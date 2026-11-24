# -*- coding: utf-8 -*-
"""
exp/exp_main.py
Version: v1.8.16  (2026-03-18)

【版本号与修改记录（changelog）】
- v1.8.16 [Test][SavePredTrueCurvesAndBestWindow]
  1) 在 test() 中新增保存全部窗口真实/预测曲线：
     - ./results/{setting}/pred_true_all_windows.npz
  2) 在 test() 中新增“最佳窗口”自动选择逻辑：
     - 默认按每窗口 MSE 最小选择，可通过 args.best_window_metric = 'mse'|'mae'|'rmse' 控制
  3) 新增最佳窗口曲线保存：
     - ./results/{setting}/best_window_curve.png
     - ./results/{setting}/best_window_curve.csv
     - ./results/{setting}/best_window_curve.npz
  4) 多变量场景下默认绘制 feature_idx=0，可通过 args.plot_feature_idx 指定展示变量
  5) 其余训练/验证逻辑保持不变
"""

import os
import time
import warnings
import inspect
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import EventMSHyper, EventMSHyperV1
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop
from utils.metrics import metric

# —— NNI（可选，当前文件中不再直接调用）——
try:
    import nni  # noqa: F401
    _NNI = True
except Exception:
    _NNI = False

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """
    主实验入口：构建模型、训练、验证与测试。
    """

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.best_val = None
        self.best_metric_val = None
        self.best_metric_epoch = -1

        self.tb_writer = None

        self.best_pred_val = None
        self.best_total_val = None
        self.best_struct_raw_val = None
        self.best_struct_scaled_val = None
        self.best_mse_val = None
        self.best_mae_val = None
        self.best_pred_epoch = -1
        self.best_total_epoch = -1
        self.best_struct_raw_epoch = -1
        self.best_struct_scaled_epoch = -1
        self.best_mse_epoch = -1
        self.best_mae_epoch = -1

        self.best_test_mse = None
        self.best_test_mae = None
        self.best_test_epoch = -1
        self.last_test_mse = None
        self.last_test_mae = None
        self.last_test_epoch = -1

        # test 阶段只保存一次 pipeline（每次 test() 调用周期内）
        self._saved_hg_pipeline_test = False
        self._saved_hg_store_test = False

    # ---------- 小工具：仅在需要时迁移到 device ----------
    @staticmethod
    def _to_device_if_needed(obj, device):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            t = obj.float()
            if t.device == device:
                return t
            return t.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: Exp_Main._to_device_if_needed(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            tlist = [Exp_Main._to_device_if_needed(v, device) for v in obj]
            return type(obj)(tlist)
        return obj

    # ---------- 小工具：兼容 DataParallel 获取模型属性 ----------
    def _get_model_attr(self, name, default=None):
        m = getattr(self, 'model', None)
        if m is None:
            return default
        if hasattr(m, name):
            return getattr(m, name)
        if hasattr(m, 'module') and hasattr(m.module, name):
            return getattr(m.module, name)
        return default

    # ---------- 前向：统一返回 (outputs, loss2_raw) ----------
    def _forward_with_loss2(self,
                            batch_x,
                            batch_x_mark,
                            dec_inp=None,
                            batch_y_mark=None,
                            batch_cycle=None):
        device = batch_x.device
        zero = torch.tensor(0.0, device=device)

        use_ci = bool(self._get_model_attr('use_cycle_index', True))
        if use_ci and (batch_cycle is not None):
            out = self.model(batch_x, batch_x_mark, cycle_index=batch_cycle)
        else:
            out = self.model(batch_x, batch_x_mark)

        if isinstance(out, tuple) and len(out) == 2:
            outputs, loss2 = out
        else:
            outputs, loss2 = out, zero

        return outputs, loss2

    # ---------- 结构项权重表 ----------
    def _reg_weight_for_key(self, key: str):
        k = (key or '').lower()
        defaults = {
            'e2e': 1.0,
            'ee_smooth': 0.1,
            'link': 1.0,
            'recon': None,
            'ib': 0.1,
            'div': 0.1,
            'const_time': 1.0,
            'const_node_time': 1.0,
        }
        arg_map = {
            'e2e': 'w_reg_e2e',
            'ee_smooth': 'w_reg_smooth',
            'link': 'w_reg_link',
            'recon': 'w_reg_recon',
            'ib': 'w_reg_ib',
            'div': 'w_reg_div',
            'const_time': 'const_time',
            'const_node_time': 'const_node_time',
        }
        w = getattr(self.args, arg_map.get(k, ''), None)
        if w is None:
            w = defaults.get(k, 1.0)
        if k == 'recon' and (w is None):
            w = getattr(self.args, 'w_reg_link', defaults['link'])
        try:
            return float(w)
        except Exception:
            return 1.0

    # ---------- 把 loss2 规约成标量（逐项加权 + 数值防护） ----------
    def _reduce_aux_loss(self, loss2, device):
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if loss2 is None:
            return zero

        if torch.is_tensor(loss2):
            v = torch.nan_to_num(loss2, nan=0.0, posinf=0.0, neginf=0.0)
            if not v.requires_grad:
                v = v.detach().to(device).requires_grad_()
            return v

        if isinstance(loss2, (int, float)):
            return torch.tensor(float(loss2), device=device, requires_grad=True)

        if isinstance(loss2, dict):
            monitor_keys = {'gts_entropy', 'gts_sparsity'}
            total = None
            for k, v in loss2.items():
                if k in monitor_keys or v is None:
                    continue
                if not torch.is_tensor(v):
                    try:
                        v = torch.as_tensor(v, dtype=torch.float32, device=device)
                    except Exception:
                        continue
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                w = self._reg_weight_for_key(k)
                term = w * v
                total = term if total is None else (total + term)

            if total is None:
                total = zero

            clip_val = float(getattr(self.args, 'struct_clip_value', 1e4))
            total = torch.clamp(total, min=-clip_val, max=clip_val)
            if not total.requires_grad:
                total = total.detach().to(device).requires_grad_()
            return total

        return zero

    @staticmethod
    def _has_grad(t: torch.Tensor) -> bool:
        try:
            return t.requires_grad and (t.grad_fn is not None)
        except Exception:
            return False

    # ---------- 工具：按“是否衰减”拆分参数组（用于 AdamW） ----------
    @staticmethod
    def _split_decay_groups(named_params):
        no_decay_keys = ["bias", "LayerNorm.weight", "layer_norm.weight", "bn.weight", "bn.bias"]
        decay, no_decay = [], []
        for n, p in named_params:
            if (not p.requires_grad):
                continue
            lname = n.lower()
            if any(k.lower() in lname for k in no_decay_keys):
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    # 结构损失缩放（w_struct_train * 预热因子）
    def _struct_scale(self, epoch_idx_zero_based: int):
        base = float(getattr(self.args, 'w_struct_train', 0.88))
        ramp_ep = int(getattr(self.args, 'struct_rampup_epochs', 0))
        if ramp_ep > 0:
            warm = min(1.0, float(epoch_idx_zero_based + 1) / float(ramp_ep))
        else:
            warm = 1.0
        return base * warm

    # ---------- 构建模型并拆分参数 ----------
    def _build_model(self):
        model_dict = {
            'EventMSHyper': EventMSHyper,
            'EventMSHyperV1': EventMSHyperV1,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.model = model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters:', num_params)

        if self.args.use_multi_gpu and self.args.use_gpu and len(getattr(self.args, 'device_ids', [])) > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            self.model = model

        main_named, hg_named = [], []
        hg_keys = (
            'hyper', 'hg', 'graph', 'structure', 'laplacian',
            'multiadphyper', 'e2e', 'pyramid', 'hyconv', 'hyperedge',
            'global_fusion', 'gts', 'tri_stage', 'spatial', 'masked_en',
            'edge_gat', 'edge_mem'
        )
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if any(k in lname for k in hg_keys):
                hg_named.append((name, p))
            else:
                main_named.append((name, p))

        if (not main_named) and hg_named:
            main_named = list(self.model.named_parameters())
            hg_named = []

        if not hg_named:
            print('[Warn] hg_params is empty -> struct optimizer will be skipped.')

        self.main_named = main_named
        self.hg_named = hg_named
        self.main_params = [p for _, p in main_named]
        self.hg_params = [p for _, p in hg_named]
        return self.model, self.hg_params

    def _get_data(self, flag):
        old_state = getattr(self.args, 'is_training', 1)
        self.args.is_training = 1
        try:
            data = data_provider(self.args, flag)
        finally:
            self.args.is_training = old_state
        return data

    # ---------- 两个 AdamW 优化器（主干 + 超图） ----------
    def _select_optimizer(self):
        lr_main = float(getattr(self.args, 'learning_rate', 1e-3))
        wd_main = float(getattr(self.args, 'weight_decay', 0.0))
        betas = tuple(getattr(self.args, 'betas', (0.9, 0.999)))
        eps = float(getattr(self.args, 'eps', 1e-8))

        lr_hg = getattr(self.args, 'learning_rate', None)
        wd_hg = getattr(self.args, 'weight_decay', None)
        if lr_hg is None and getattr(self.args, 'lr_ratio_hg', None) is not None:
            lr_hg = lr_main * float(self.args.lr_ratio_hg)
        lr_hg = float(lr_main if lr_hg is None else lr_hg)
        wd_hg = float(wd_main if wd_hg is None else wd_hg)

        main_decay, main_nodecay = self._split_decay_groups(
            self.main_named if hasattr(self, 'main_named') else list(self.model.named_parameters())
        )
        opt_main = optim.AdamW(
            [
                {"params": main_decay, "weight_decay": wd_main},
                {"params": main_nodecay, "weight_decay": 0.0},
            ],
            lr=lr_main, betas=betas, eps=eps
        )

        opt_hg = None
        if self.hg_params:
            hg_decay, hg_nodecay = self._split_decay_groups(self.hg_named)
            opt_hg = optim.AdamW(
                [
                    {"params": hg_decay, "weight_decay": wd_hg},
                    {"params": hg_nodecay, "weight_decay": 0.0},
                ],
                lr=lr_hg, betas=betas, eps=eps
            )

        return opt_main, opt_hg

    # ---------- 选择损失：mse|mae|huber ----------
    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'mse')).lower()
        if loss_name == 'mae':
            return nn.L1Loss()
        if loss_name in ('huber', 'smoothl1'):
            delta = float(getattr(self.args, 'huber_delta', 1.0))
            return nn.SmoothL1Loss(beta=delta)
        return nn.MSELoss()

    # ---------- 分项验证：返回 dict（无任何额外分析/抓取） ----------
    @torch.no_grad()
    def _vali_split(self, vali_loader, criterion):
        self.model.eval()
        preds, trues = [], []

        vals_pred = []
        vals_struct_raw = []

        for batch in vali_loader:
            if len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_cycle = None
            else:
                batch_x, batch_y = batch[0], batch[1]
                batch_x_mark = batch[2] if len(batch) > 2 else None
                batch_y_mark = batch[3] if len(batch) > 3 else None
                batch_cycle = batch[4] if len(batch) > 4 else None

            batch_x = self._to_device_if_needed(batch_x, self.device)
            batch_y = self._to_device_if_needed(batch_y, self.device)
            batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
            batch_y_mark = self._to_device_if_needed(batch_y_mark, self.device)
            batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None

            dec_inp = None
            if self.args.output_attention:
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = self._to_device_if_needed(dec_inp, self.device)

            out, loss2_raw = self._forward_with_loss2(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
            )

            f_dim = -1 if self.args.features == 'MS' else 0
            out = out[:, -self.args.pred_len:, f_dim:]
            tgt = batch_y[:, -self.args.pred_len:, f_dim:]

            loss_pred = criterion(out, tgt)
            vals_pred.append(loss_pred.detach().item())

            struct_raw = self._reduce_aux_loss(loss2_raw, batch_x.device)
            struct_raw = torch.nan_to_num(struct_raw, nan=0.0, posinf=0.0, neginf=0.0)
            vals_struct_raw.append(struct_raw.detach().item())

            preds.append(out.detach().cpu().numpy())
            trues.append(tgt.detach().cpu().numpy())

        if not vals_pred:
            return dict(
                val_pred=0.0, val_struct_raw=0.0,
                val_struct_scaled=0.0, val_total=0.0,
                mse=0.0, mae=0.0
            )

        val_pred = float(np.mean(vals_pred))
        val_struct_raw = float(np.mean(vals_struct_raw))
        w_struct = float(getattr(self.args, 'w_struct', 0.8))
        val_struct_scaled = w_struct * val_struct_raw
        val_total = val_pred + val_struct_scaled

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, *_ = metric(preds, trues)

        return dict(
            val_pred=val_pred,
            val_struct_raw=val_struct_raw,
            val_struct_scaled=val_struct_scaled,
            val_total=val_total,
            mse=float(mse),
            mae=float(mae),
        )

    def vali(self, vali_data, vali_loader, criterion):
        m = self._vali_split(vali_loader, criterion)
        return float(m['val_total'])

    # ---------- 计算 loader 的 MSE/MAE（无任何额外分析/抓取） ----------
    def _eval_mse_mae(self, data_loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_cycle = None
                else:
                    batch_x, batch_y = batch[0], batch[1]
                    batch_x_mark = batch[2] if len(batch) > 2 else None
                    batch_y_mark = batch[3] if len(batch) > 3 else None
                    batch_cycle = batch[4] if len(batch) > 4 else None

                batch_x = self._to_device_if_needed(batch_x, self.device)
                batch_y = self._to_device_if_needed(batch_y, self.device)
                batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
                batch_y_mark = self._to_device_if_needed(batch_y_mark, self.device)
                batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = None
                if self.args.output_attention:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                    dec_inp = self._to_device_if_needed(dec_inp, self.device)

                out, _ = self._forward_with_loss2(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                out = out[:, -self.args.pred_len:, f_dim:]
                tgt = batch_y[:, -self.args.pred_len:, f_dim:]
                preds.append(out.detach().cpu().numpy())
                trues.append(tgt.detach().cpu().numpy())

        if not preds:
            self.model.train()
            return 0.0, 0.0
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, *_ = metric(preds, trues)
        self.model.train()
        return float(mse), float(mae)

    # ---------- 完整预测指标（无任何额外分析/抓取） ----------
    @torch.no_grad()
    def _eval_full_metrics(self, data_loader):
        self.model.eval()
        preds, trues = [], []
        for batch in data_loader:
            if len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_cycle = None
            else:
                batch_x, batch_y = batch[0], batch[1]
                batch_x_mark = batch[2] if len(batch) > 2 else None
                batch_y_mark = batch[3] if len(batch) > 3 else None
                batch_cycle = batch[4] if len(batch) > 4 else None

            batch_x = self._to_device_if_needed(batch_x, self.device)
            batch_y = self._to_device_if_needed(batch_y, self.device)
            batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
            batch_y_mark = self._to_device_if_needed(batch_y_mark, self.device)
            batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None

            dec_inp = None
            if self.args.output_attention:
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                dec_inp = self._to_device_if_needed(dec_inp, self.device)

            out, _ = self._forward_with_loss2(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
            )
            f_dim = -1 if self.args.features == 'MS' else 0
            out = out[:, -self.args.pred_len:, f_dim:]
            tgt = batch_y[:, -self.args.pred_len:, f_dim:]
            preds.append(out.detach().cpu().numpy())
            trues.append(tgt.detach().cpu().numpy())

        if not preds:
            self.model.train()
            return {
                'mse': 0.0, 'mae': 0.0, 'rmse': 0.0,
                'mape': 0.0, 'mspe': 0.0, 'rse': 0.0, 'corr': 0.0,
            }

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        def _to_scalar(x):
            x = np.asarray(x)
            if x.size == 0:
                return 0.0
            if x.ndim == 0:
                return float(x)
            return float(x.mean())

        out_dict = {
            'mse': _to_scalar(mse),
            'mae': _to_scalar(mae),
            'rmse': _to_scalar(rmse),
            'mape': _to_scalar(mape),
            'mspe': _to_scalar(mspe),
            'rse': _to_scalar(rse),
            'corr': _to_scalar(corr),
        }
        self.model.train()
        return out_dict

    # ---------- TB writer（仅记录标量指标，不做额外抓取） ----------
    def _init_tb_writer(self, setting: str) -> SummaryWriter:
        logdir = os.path.join("runs", setting)
        os.makedirs(logdir, exist_ok=True)
        return SummaryWriter(log_dir=logdir)

    # =========================================================
    # Test 阶段 HG-PipelineDump（仅 test() 调用周期内触发）
    # =========================================================
    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _call_with_supported_kwargs(fn, **kwargs):
        if not callable(fn):
            return None
        try:
            sig = inspect.signature(fn)
            supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**supported)
        except Exception:
            try:
                return fn(**kwargs)
            except Exception:
                try:
                    return fn()
                except Exception:
                    return None

    @staticmethod
    def _to_numpy_recursive(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, dict):
            return {k: Exp_Main._to_numpy_recursive(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [Exp_Main._to_numpy_recursive(v) for v in x]
        return x

    def _collect_hg_pipeline_snapshot(self) -> Dict[str, Any]:
        """
        从模型侧缓存抓取（若存在则抓取）
        """
        m = self._unwrap_model(self.model)
        snap: Dict[str, Any] = {}

        def _try(name: str, key: str = None, **kwargs):
            fn = getattr(m, name, None)
            if callable(fn):
                snap[key or name.replace("get_", "")] = self._call_with_supported_kwargs(fn, **kwargs)

        _try("get_ms_node_stats", clear=True, to_cpu=True)
        _try("get_time_hypergraph_struct", clear=True, to_cpu=True)
        _try("get_h0_struct", clear=True, to_cpu=True)
        _try("get_proto_hypergraph_struct", clear=True, to_cpu=True)
        _try("get_event_hypergraph_struct", clear=True, to_cpu=True)
        _try("get_content_feat", key="content_feat", clear=True, to_cpu=True)
        _try("get_chain_hg_analysis", key="chain_hg_analysis", clear=True, to_cpu=True)
        _try("get_chain_hg_analysis_err", key="chain_hg_analysis_err", clear=True, to_cpu=True)
        _try("get_debug", key="debug", clear=True, to_cpu=True)

        _try("get_content_stats", clear=True, to_cpu=True)
        _try("get_proto_stats", use_clean=True, clear=True, to_cpu=True)
        _try("get_hg_analysis", clear=True, to_cpu=True)
        _try("get_link_analysis", clear=True, to_cpu=True)

        for name in ("get_explicit_hypergraph_struct", "get_hypergraph_struct"):
            _try(name, use_clean=True, clear=True, to_cpu=True)

        snap = self._to_numpy_recursive(snap)
        return snap

    def _save_hg_pipeline_test(self, setting: str, extra_meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        保存到 ./results/{setting}/hg_pipeline_test.npz
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        out_path = os.path.join(folder_path, 'hg_pipeline_test.npz')

        snap = self._collect_hg_pipeline_snapshot()

        meta = {
            "setting": setting,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        if isinstance(extra_meta, dict):
            meta.update(extra_meta)

        pack: Dict[str, Any] = {"meta": np.array(meta, dtype=object)}
        for k, v in snap.items():
            pack[k] = np.array(v, dtype=object)

        np.savez_compressed(out_path, **pack)
        print(f"[HG-PipelineDump] saved to: {out_path}")
        return out_path


    # =========================================================
    # 曲线导出 / 最佳窗口-最佳通道选择
    # =========================================================
    @staticmethod
    def _ensure_3d_curve_arrays(preds, trues):
        """
        统一为 [N, pred_len, C]
        """
        preds = np.asarray(preds)
        trues = np.asarray(trues)

        if preds.shape != trues.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, trues={trues.shape}")

        if preds.ndim == 1:
            preds = preds[None, :, None]
            trues = trues[None, :, None]
        elif preds.ndim == 2:
            preds = preds[:, :, None]
            trues = trues[:, :, None]
        elif preds.ndim != 3:
            raise ValueError(f"Unsupported preds ndim={preds.ndim}, expect 1/2/3")

        return preds, trues

    def _select_best_channel_and_window(self, preds: np.ndarray, trues: np.ndarray):
        """
        preds/trues: [N, T, C]

        选择逻辑：
        1) 先对每个 (window, channel) 计算误差
        2) 对每个 channel，在所有 window 上取平均，找最佳通道
        3) 在最佳通道内，再找最佳窗口

        可选指标：
            args.best_select_metric = 'mse' | 'mae' | 'rmse'
        若未设置，则回退到 args.best_window_metric，再回退到 'mse'
        """
        preds, trues = self._ensure_3d_curve_arrays(preds, trues)

        diff = preds - trues  # [N, T, C]
        mse_wc = np.mean(diff ** 2, axis=1)           # [N, C]
        mae_wc = np.mean(np.abs(diff), axis=1)        # [N, C]
        rmse_wc = np.sqrt(mse_wc)                     # [N, C]

        metric_name = str(
            getattr(
                self.args,
                'best_select_metric',
                getattr(self.args, 'best_window_metric', 'mse')
            )
        ).lower()

        if metric_name == 'mae':
            score_wc = mae_wc
        elif metric_name == 'rmse':
            score_wc = rmse_wc
        else:
            metric_name = 'mse'
            score_wc = mse_wc

        # 每个通道在所有窗口上的平均表现
        channel_mse = np.mean(mse_wc, axis=0)   # [C]
        channel_mae = np.mean(mae_wc, axis=0)   # [C]
        channel_rmse = np.mean(rmse_wc, axis=0) # [C]
        channel_score = np.mean(score_wc, axis=0)

        # 每个窗口在所有通道上的平均表现
        window_mse = np.mean(mse_wc, axis=1)    # [N]
        window_mae = np.mean(mae_wc, axis=1)    # [N]
        window_rmse = np.mean(rmse_wc, axis=1)  # [N]
        window_score = np.mean(score_wc, axis=1)

        best_channel_idx = int(np.argmin(channel_score))
        best_channel_score = float(channel_score[best_channel_idx])

        # 在最佳通道里找最佳窗口
        best_window_in_best_channel_idx = int(np.argmin(score_wc[:, best_channel_idx]))
        best_window_in_best_channel_score = float(score_wc[best_window_in_best_channel_idx, best_channel_idx])

        # 全局“最佳窗口”（窗口维度上平均所有通道）
        best_window_idx = int(np.argmin(window_score))
        best_window_score = float(window_score[best_window_idx])

        # 全局最佳 pair（某一个 window-channel 组合）
        best_pair_flat_idx = int(np.argmin(score_wc))
        best_pair_window_idx, best_pair_channel_idx = np.unravel_index(best_pair_flat_idx, score_wc.shape)
        best_pair_score = float(score_wc[best_pair_window_idx, best_pair_channel_idx])

        return {
            'metric_name': metric_name,

            'mse_wc': mse_wc,
            'mae_wc': mae_wc,
            'rmse_wc': rmse_wc,
            'score_wc': score_wc,

            'channel_mse': channel_mse,
            'channel_mae': channel_mae,
            'channel_rmse': channel_rmse,
            'channel_score': channel_score,

            'window_mse': window_mse,
            'window_mae': window_mae,
            'window_rmse': window_rmse,
            'window_score': window_score,

            'best_channel_idx': best_channel_idx,
            'best_channel_score': best_channel_score,

            'best_window_in_best_channel_idx': best_window_in_best_channel_idx,
            'best_window_in_best_channel_score': best_window_in_best_channel_score,

            'best_window_idx': best_window_idx,
            'best_window_score': best_window_score,

            'best_pair_window_idx': int(best_pair_window_idx),
            'best_pair_channel_idx': int(best_pair_channel_idx),
            'best_pair_score': best_pair_score,
        }

    def _save_complete_curves(self, setting: str, preds: np.ndarray, trues: np.ndarray, select_info: dict):
        """
        保存完整预测结果：
        1) 原始窗口结果 [N, T, C]
        2) 按窗口顺序拼接后的 full curve [N*T, C]
        3) 各类 ranking score
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = self._ensure_3d_curve_arrays(preds, trues)
        n_win, pred_len, n_ch = preds.shape

        out_path = os.path.join(folder_path, 'pred_true_complete_curves.npz')
        np.savez_compressed(
            out_path,
            preds=preds,                                   # [N, T, C]
            trues=trues,                                   # [N, T, C]
            preds_flat_by_window_order=preds.reshape(n_win * pred_len, n_ch),
            trues_flat_by_window_order=trues.reshape(n_win * pred_len, n_ch),

            mse_wc=select_info['mse_wc'],
            mae_wc=select_info['mae_wc'],
            rmse_wc=select_info['rmse_wc'],
            score_wc=select_info['score_wc'],

            channel_mse=select_info['channel_mse'],
            channel_mae=select_info['channel_mae'],
            channel_rmse=select_info['channel_rmse'],
            channel_score=select_info['channel_score'],

            window_mse=select_info['window_mse'],
            window_mae=select_info['window_mae'],
            window_rmse=select_info['window_rmse'],
            window_score=select_info['window_score'],

            metric_name=np.array(select_info['metric_name'], dtype=object),
            best_channel_idx=np.array(select_info['best_channel_idx']),
            best_window_idx=np.array(select_info['best_window_idx']),
            best_window_in_best_channel_idx=np.array(select_info['best_window_in_best_channel_idx']),
            best_pair_window_idx=np.array(select_info['best_pair_window_idx']),
            best_pair_channel_idx=np.array(select_info['best_pair_channel_idx']),
        )
        print(f"[CurveSave] complete curves saved to: {out_path}")
        return out_path

    def _save_rankings_csv(self, setting: str, select_info: dict):
        """
        保存通道排名 / 窗口排名
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        # channel ranking
        channel_table = np.stack([
            np.arange(len(select_info['channel_score'])),
            select_info['channel_mse'],
            select_info['channel_mae'],
            select_info['channel_rmse'],
            select_info['channel_score'],
        ], axis=1)
        channel_order = np.argsort(channel_table[:, -1])
        channel_csv = os.path.join(folder_path, 'channel_ranking.csv')
        np.savetxt(
            channel_csv,
            channel_table[channel_order],
            delimiter=',',
            header='channel_idx,avg_mse,avg_mae,avg_rmse,select_score',
            comments='',
            fmt='%.10f'
        )
        print(f"[CurveSave] channel ranking saved to: {channel_csv}")

        # window ranking
        window_table = np.stack([
            np.arange(len(select_info['window_score'])),
            select_info['window_mse'],
            select_info['window_mae'],
            select_info['window_rmse'],
            select_info['window_score'],
        ], axis=1)
        window_order = np.argsort(window_table[:, -1])
        window_csv = os.path.join(folder_path, 'window_ranking.csv')
        np.savetxt(
            window_csv,
            window_table[window_order],
            delimiter=',',
            header='window_idx,avg_mse,avg_mae,avg_rmse,select_score',
            comments='',
            fmt='%.10f'
        )
        print(f"[CurveSave] window ranking saved to: {window_csv}")

        return {
            'channel_csv': channel_csv,
            'window_csv': window_csv,
        }

    def _save_best_channel_full_curve(self, setting: str, preds: np.ndarray, trues: np.ndarray, select_info: dict):
        """
        保存“最佳通道”的完整曲线（按窗口顺序串接）
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = self._ensure_3d_curve_arrays(preds, trues)

        ch = int(select_info['best_channel_idx'])
        metric_name = str(select_info['metric_name'])
        metric_value = float(select_info['best_channel_score'])

        pred_curve = preds[:, :, ch].reshape(-1)
        true_curve = trues[:, :, ch].reshape(-1)

        npz_path = os.path.join(folder_path, 'best_channel_full_curve.npz')
        np.savez_compressed(
            npz_path,
            channel_idx=np.array(ch),
            metric_name=np.array(metric_name, dtype=object),
            metric_value=np.array(metric_value),
            pred_curve=pred_curve,
            true_curve=true_curve,
        )
        print(f"[CurveSave] best channel full curve npz saved to: {npz_path}")

        csv_path = os.path.join(folder_path, 'best_channel_full_curve.csv')
        table = np.stack([
            np.arange(len(true_curve)),
            true_curve,
            pred_curve
        ], axis=1)
        np.savetxt(
            csv_path,
            table,
            delimiter=',',
            header='t,true,pred',
            comments='',
            fmt='%.10f'
        )
        print(f"[CurveSave] best channel full curve csv saved to: {csv_path}")

        png_path = os.path.join(folder_path, 'best_channel_full_curve.png')
        plt.figure(figsize=(12, 4))
        plt.plot(true_curve, label='True')
        plt.plot(pred_curve, label='Pred')
        plt.xlabel('Flattened Time Step (by window order)')
        plt.ylabel('Value')
        plt.title(
            f'Best Channel #{ch} | {metric_name.upper()}={metric_value:.6f} | by window order'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"[CurveSave] best channel full curve png saved to: {png_path}")

        return {
            'best_channel_full_curve_npz': npz_path,
            'best_channel_full_curve_csv': csv_path,
            'best_channel_full_curve_png': png_path,
            'best_channel_idx': ch,
            'best_channel_metric_name': metric_name,
            'best_channel_metric_value': metric_value,
        }

    def _save_best_window_in_best_channel(self, setting: str, preds: np.ndarray, trues: np.ndarray, select_info: dict):
        """
        保存“最佳通道中的最佳窗口”曲线
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        preds, trues = self._ensure_3d_curve_arrays(preds, trues)

        ch = int(select_info['best_channel_idx'])
        win = int(select_info['best_window_in_best_channel_idx'])
        metric_name = str(select_info['metric_name'])
        metric_value = float(select_info['best_window_in_best_channel_score'])

        pred_curve = preds[win, :, ch]
        true_curve = trues[win, :, ch]

        npz_path = os.path.join(folder_path, 'best_window_in_best_channel_curve.npz')
        np.savez_compressed(
            npz_path,
            channel_idx=np.array(ch),
            window_idx=np.array(win),
            metric_name=np.array(metric_name, dtype=object),
            metric_value=np.array(metric_value),
            pred_curve=pred_curve,
            true_curve=true_curve,
        )
        print(f"[CurveSave] best window-in-best-channel npz saved to: {npz_path}")

        csv_path = os.path.join(folder_path, 'best_window_in_best_channel_curve.csv')
        table = np.stack([
            np.arange(len(true_curve)),
            true_curve,
            pred_curve
        ], axis=1)
        np.savetxt(
            csv_path,
            table,
            delimiter=',',
            header='t,true,pred',
            comments='',
            fmt='%.10f'
        )
        print(f"[CurveSave] best window-in-best-channel csv saved to: {csv_path}")

        png_path = os.path.join(folder_path, 'best_window_in_best_channel_curve.png')
        plt.figure(figsize=(10, 4))
        plt.plot(true_curve, label='True')
        plt.plot(pred_curve, label='Pred')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(
            f'Best Channel #{ch} | Best Window #{win} | {metric_name.upper()}={metric_value:.6f}'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"[CurveSave] best window-in-best-channel png saved to: {png_path}")

        return {
            'best_window_in_best_channel_npz': npz_path,
            'best_window_in_best_channel_csv': csv_path,
            'best_window_in_best_channel_png': png_path,
            'best_window_in_best_channel_idx': win,
            'best_window_in_best_channel_metric_name': metric_name,
            'best_window_in_best_channel_metric_value': metric_value,
        }
    # =========================================================
    # 曲线保存与最佳窗口选择
    # =========================================================
    def _select_best_window(self, preds: np.ndarray, trues: np.ndarray):
        """
        preds/trues: [N, pred_len, C] 或 [N, pred_len]
        默认按每个窗口整体 MSE 最小选最优窗口
        可通过 args.best_window_metric = 'mse'|'mae'|'rmse' 控制
        """
        if preds.shape != trues.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, trues={trues.shape}")

        diff = preds - trues
        if diff.ndim <= 1:
            mse_per_win = np.array([float(np.mean(diff ** 2))], dtype=np.float64)
            mae_per_win = np.array([float(np.mean(np.abs(diff)))], dtype=np.float64)
        else:
            reduce_axes = tuple(range(1, diff.ndim))
            mse_per_win = np.mean(diff ** 2, axis=reduce_axes)
            mae_per_win = np.mean(np.abs(diff), axis=reduce_axes)

        rmse_per_win = np.sqrt(mse_per_win)

        metric_name = str(getattr(self.args, 'best_window_metric', 'mae')).lower()
        if metric_name == 'mae':
            score = mae_per_win
        elif metric_name == 'rmse':
            score = rmse_per_win
        else:
            metric_name = 'mse'
            score = mse_per_win

        best_idx = int(np.argmin(score))

        return {
            'best_idx': best_idx,
            'metric_name': metric_name,
            'best_score': float(score[best_idx]),
            'mse_per_win': mse_per_win,
            'mae_per_win': mae_per_win,
            'rmse_per_win': rmse_per_win,
        }

    def _save_all_curves(self, setting: str, preds: np.ndarray, trues: np.ndarray, select_info: dict):
        """
        保存所有窗口的真实/预测曲线
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        out_path = os.path.join(folder_path, 'pred_true_all_windows.npz')
        np.savez_compressed(
            out_path,
            preds=preds,
            trues=trues,
            mse_per_win=select_info['mse_per_win'],
            mae_per_win=select_info['mae_per_win'],
            rmse_per_win=select_info['rmse_per_win'],
            best_idx=np.array(select_info['best_idx']),
            best_metric=np.array(select_info['metric_name'], dtype=object),
            best_score=np.array(select_info['best_score']),
        )
        print(f"[CurveSave] all windows saved to: {out_path}")
        return out_path

    def _save_best_window_plot(self, setting: str, preds: np.ndarray, trues: np.ndarray, select_info: dict):
        """
        保存最佳窗口图像、CSV、NPZ
        多变量时默认画 feature_idx=0，可通过 args.plot_feature_idx 指定
        """
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        best_idx = int(select_info['best_idx'])
        metric_name = str(select_info['metric_name'])
        best_score = float(select_info['best_score'])

        pred_win = preds[best_idx]
        true_win = trues[best_idx]

        if pred_win.ndim == 0:
            feature_idx = 0
            pred_curve = np.array([float(pred_win)], dtype=np.float64)
            true_curve = np.array([float(true_win)], dtype=np.float64)
        elif pred_win.ndim == 1:
            feature_idx = 0
            pred_curve = pred_win
            true_curve = true_win
        else:
            feature_idx = int(getattr(self.args, 'plot_feature_idx', 0))
            feature_idx = max(0, min(feature_idx, pred_win.shape[-1] - 1))
            pred_curve = pred_win[:, feature_idx]
            true_curve = true_win[:, feature_idx]

        pred_curve = np.asarray(pred_curve).reshape(-1)
        true_curve = np.asarray(true_curve).reshape(-1)

        curve_npz_path = os.path.join(folder_path, 'best_window_curve.npz')
        np.savez_compressed(
            curve_npz_path,
            best_idx=np.array(best_idx),
            feature_idx=np.array(feature_idx),
            metric_name=np.array(metric_name, dtype=object),
            metric_value=np.array(best_score),
            pred_curve=pred_curve,
            true_curve=true_curve,
        )
        print(f"[CurveSave] best window raw data saved to: {curve_npz_path}")

        curve_csv_path = os.path.join(folder_path, 'best_window_curve.csv')
        curve_table = np.stack([
            np.arange(len(true_curve)),
            true_curve,
            pred_curve
        ], axis=1)
        np.savetxt(
            curve_csv_path,
            curve_table,
            delimiter=',',
            header='t,true,pred',
            comments='',
            fmt='%.10f'
        )
        print(f"[CurveSave] best window csv saved to: {curve_csv_path}")

        fig_path = os.path.join(folder_path, 'best_window_curve.png')
        plt.figure(figsize=(10, 4))
        plt.plot(true_curve, label='True')
        plt.plot(pred_curve, label='Pred')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(
            f'Best Window #{best_idx} | feature={feature_idx} | '
            f'{metric_name.upper()}={best_score:.6f}'
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[CurveSave] best window figure saved to: {fig_path}")

        return {
            'best_window_npz': curve_npz_path,
            'best_window_csv': curve_csv_path,
            'best_window_png': fig_path,
            'best_idx': best_idx,
            'feature_idx': feature_idx,
            'metric_name': metric_name,
            'metric_value': best_score,
        }

    # ---------- 训练 ----------
    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data('val')
            test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        min_improve = float(getattr(self.args, 'min_improve', 0.001))

        try:
            early_stopping = EarlyStopping(
                patience=self.args.patience,
                verbose=True,
                delta=min_improve
            )
        except TypeError:
            early_stopping = EarlyStopping(
                patience=self.args.patience,
                verbose=True
            )

        opt_main, opt_hg = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        grad_clip = float(getattr(self.args, 'grad_clip', 1.0))
        struct_clip_value = float(getattr(self.args, 'struct_clip_value', 1e4))

        LOG_WINDOW = 100
        eval_interval = int(getattr(self.args, 'eval_interval', 0))

        mode = str(getattr(self.args, 'early_stop_on', 'mse')).lower()
        name_map = {
            'total': 'val_total',
            'pred': 'val_pred',
            'struct': 'val_struct_raw',
            'struct_raw': 'val_struct_raw',
            'struct_scaled': 'val_struct_scaled',
            'mse': 'mse',
            'mae': 'mae',
            'val_mse': 'mse',
            'val_mae': 'mae',
        }
        monitor_key = name_map.get(mode, 'val_total')

        save_dual = bool(getattr(self.args, 'save_dual_checkpoints', False))
        two_fwd = bool(getattr(self.args, 'two_forward_struct', True))
        print(f"[Info] two_forward_struct = {int(two_fwd)}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            logs_pred, logs_struct = [], []
            self.model.train()
            epoch_time = time.time()

            struct_scale_train = self._struct_scale(epoch)
            w_struct_base = float(getattr(self.args, 'w_struct', 0.4))
            eff_struct_coef = w_struct_base * struct_scale_train
            if struct_scale_train != 1.0 or w_struct_base != 1.0:
                print(
                    f"[Info] epoch {epoch+1}: effective_struct_coef={eff_struct_coef:.4f} "
                    f"(w_struct={w_struct_base}, "
                    f"w_struct_train={getattr(self.args, 'w_struct_train', 0.4)}, "
                    f"rampup={getattr(self.args, 'struct_rampup_epochs', 0)})"
                )

            for i, batch in enumerate(train_loader):
                iter_count += 1

                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_cycle = None
                else:
                    batch_x, batch_y = batch[0], batch[1]
                    batch_x_mark = batch[2] if len(batch) > 2 else None
                    batch_y_mark = batch[3] if len(batch) > 3 else None
                    batch_cycle = batch[4] if len(batch) > 4 else None

                opt_main.zero_grad(set_to_none=True)
                if opt_hg is not None:
                    opt_hg.zero_grad(set_to_none=True)

                batch_x = self._to_device_if_needed(batch_x, self.device)
                batch_y = self._to_device_if_needed(batch_y, self.device)
                batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
                batch_y_mark = self._to_device_if_needed(batch_y_mark, self.device)
                batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = None
                if self.args.output_attention:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                    dec_inp = self._to_device_if_needed(dec_inp, self.device)

                if two_fwd:
                    # ---------- Pass-A：预测 ----------
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outA, _loss2A = self._forward_with_loss2(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                            )
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outA = outA[:, -self.args.pred_len:, f_dim:]
                            tgtA = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss_pred = criterion(outA, tgtA)

                        from torch.nn.utils import clip_grad_norm_
                        scaler.scale(loss_pred).backward(
                            inputs=(self.main_params if self.main_params else list(self.model.parameters()))
                        )
                        if grad_clip > 0:
                            scaler.unscale_(opt_main)
                            clip_grad_norm_(
                                self.main_params if self.main_params else self.model.parameters(),
                                grad_clip
                            )
                        scaler.step(opt_main)
                    else:
                        outA, _loss2A = self._forward_with_loss2(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                        )
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outA = outA[:, -self.args.pred_len:, f_dim:]
                        tgtA = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss_pred = criterion(outA, tgtA)
                        loss_pred.backward(
                            inputs=(self.main_params if self.main_params else list(self.model.parameters()))
                        )
                        if grad_clip > 0:
                            from torch.nn.utils import clip_grad_norm_
                            clip_grad_norm_(
                                self.main_params if self.main_params else self.model.parameters(),
                                grad_clip
                            )
                        opt_main.step()

                    # ---------- Pass-B：结构 ----------
                    if opt_hg is not None:
                        opt_hg.zero_grad(set_to_none=True)

                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                outB, loss2_rawB = self._forward_with_loss2(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                                )
                                loss_struct_raw = 0.2 * self._reduce_aux_loss(loss2_rawB, batch_x.device)
                                loss_struct = torch.clamp(
                                    loss_struct_raw * eff_struct_coef,
                                    -struct_clip_value,
                                    struct_clip_value
                                )
                            do_struct = self._has_grad(loss_struct) and (loss_struct.detach().item() != 0.0)
                            if do_struct:
                                from torch.nn.utils import clip_grad_norm_
                                scaler.scale(loss_struct).backward(inputs=self.hg_params)
                                if grad_clip > 0:
                                    scaler.unscale_(opt_hg)
                                    clip_grad_norm_(self.hg_params, grad_clip)
                                scaler.step(opt_hg)
                            if scaler is not None:
                                scaler.update()
                        else:
                            outB, loss2_rawB = self._forward_with_loss2(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                            )
                            loss_struct_raw = 0.5 * self._reduce_aux_loss(loss2_rawB, batch_x.device)
                            loss_struct = torch.clamp(
                                loss_struct_raw * eff_struct_coef,
                                -struct_clip_value,
                                struct_clip_value
                            )
                            do_struct = self._has_grad(loss_struct) and (loss_struct.detach().item() != 0.0)
                            if do_struct:
                                loss_struct.backward(inputs=self.hg_params)
                                if grad_clip > 0:
                                    from torch.nn.utils import clip_grad_norm_
                                    clip_grad_norm_(self.hg_params, grad_clip)
                                opt_hg.step()
                    else:
                        loss_struct = torch.tensor(0.0, device=batch_x.device)

                else:
                    # 单次 forward 同时更新主干+结构
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            out, loss2_raw = self._forward_with_loss2(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                            )
                            f_dim = -1 if self.args.features == 'MS' else 0
                            out = out[:, -self.args.pred_len:, f_dim:]
                            tgt = batch_y[:, -self.args.pred_len:, f_dim:]
                            loss_pred = criterion(out, tgt)
                            loss_struct_raw = self._reduce_aux_loss(loss2_raw, batch_x.device)
                            loss_struct = torch.clamp(
                                loss_struct_raw * eff_struct_coef,
                                -struct_clip_value,
                                struct_clip_value
                            )

                        from torch.nn.utils import clip_grad_norm_
                        scaler.scale(loss_pred).backward(retain_graph=True)

                        do_struct = (opt_hg is not None) and self._has_grad(loss_struct) and (loss_struct.detach().item() != 0.0)
                        if do_struct:
                            scaler.scale(loss_struct).backward(inputs=self.hg_params)

                        if grad_clip > 0:
                            scaler.unscale_(opt_main)
                            clip_grad_norm_(
                                self.main_params if self.main_params else self.model.parameters(),
                                grad_clip
                            )
                            if do_struct:
                                scaler.unscale_(opt_hg)
                                clip_grad_norm_(self.hg_params, grad_clip)

                        scaler.step(opt_main)
                        if do_struct:
                            scaler.step(opt_hg)
                        scaler.update()
                    else:
                        out, loss2_raw = self._forward_with_loss2(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                        )
                        f_dim = -1 if self.args.features == 'MS' else 0
                        out = out[:, -self.args.pred_len:, f_dim:]
                        tgt = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss_pred = criterion(out, tgt)
                        loss_struct_raw = self._reduce_aux_loss(loss2_raw, batch_x.device)
                        loss_struct = torch.clamp(
                            loss_struct_raw * eff_struct_coef,
                            -struct_clip_value,
                            struct_clip_value
                        )

                        loss_pred.backward(retain_graph=True)
                        do_struct = (opt_hg is not None) and self._has_grad(loss_struct) and (loss_struct.detach().item() != 0.0)
                        if do_struct:
                            loss_struct.backward(inputs=self.hg_params)

                        if grad_clip > 0:
                            from torch.nn.utils import clip_grad_norm_
                            clip_grad_norm_(
                                self.main_params if self.main_params else self.model.parameters(),
                                grad_clip
                            )
                            if do_struct:
                                clip_grad_norm_(self.hg_params, grad_clip)

                        opt_main.step()
                        if do_struct:
                            opt_hg.step()

                logs_pred.append(float(loss_pred.item()))
                logs_struct.append(float(loss_struct.item()) if torch.is_tensor(loss_struct) else float(loss_struct))

                if (i + 1) % LOG_WINDOW == 0:
                    wnd = min(LOG_WINDOW, len(logs_pred))
                    avg_pred = float(np.mean(logs_pred[-wnd:]))
                    avg_struct = float(np.mean(logs_struct[-wnd:]))
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | "
                        f"avg_pred({wnd}): {avg_pred:.7f} | avg_struct({wnd}): {avg_struct:.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            ep_avg_pred = float(np.mean(logs_pred)) if logs_pred else 0.0
            ep_avg_struct = float(np.mean(logs_struct)) if logs_struct else 0.0
            print(
                f"Epoch: {epoch+1} cost: {time.time() - epoch_time:.2f}s | "
                f"epoch_avg_pred: {ep_avg_pred:.7f} | epoch_avg_struct: {ep_avg_struct:.7f}"
            )

            # 可选：间隔评估
            if eval_interval > 0 and ((epoch + 1) % eval_interval == 0):
                train_mse, train_mae = self._eval_mse_mae(train_loader)
                if not self.args.train_only:
                    val_tmp = self._vali_split(vali_loader, criterion)
                    print(
                        f"[Eval@epoch{epoch+1}] "
                        f"Train MSE: {train_mse:.6f}, MAE: {train_mae:.6f} | "
                        f"Val MSE: {val_tmp['mse']:.6f}, MAE: {val_tmp['mae']:.6f}"
                    )

            if not self.args.train_only:
                m = self._vali_split(vali_loader, criterion)
                print(
                    f"Epoch {epoch+1} | ValPred: {m['val_pred']:.7f} | "
                    f"ValStructRaw: {m['val_struct_raw']:.7f} | "
                    f"ValStructScaled: {m['val_struct_scaled']:.7f} | "
                    f"ValTotal: {m['val_total']:.7f} | "
                    f"(Val MSE: {m['mse']:.6f}, MAE: {m['mae']:.6f})"
                )

                test_metrics = self._eval_full_metrics(test_loader)
                print(
                    f"[Test@epoch{epoch+1}] "
                    f"MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}, "
                    f"RMSE: {test_metrics['rmse']:.6f}, MAPE: {test_metrics['mape']:.6f}, "
                    f"MSPE: {test_metrics['mspe']:.6f}, RSE: {test_metrics['rse']:.6f}, "
                    f"Corr: {test_metrics['corr']:.6f}"
                )

                try:
                    cur_mse = float(test_metrics.get('mse', 0.0))
                    cur_mae = float(test_metrics.get('mae', 0.0))
                except Exception:
                    cur_mse, cur_mae = 0.0, 0.0
                self.last_test_mse = cur_mse
                self.last_test_mae = cur_mae
                self.last_test_epoch = epoch + 1
                if (self.best_test_mse is None) or (cur_mse < self.best_test_mse):
                    self.best_test_mse = cur_mse
                    self.best_test_mae = cur_mae
                    self.best_test_epoch = epoch + 1

                # 早停与 checkpoint
                def _update_best(name, val, epoch_idx, filename):
                    cur_best = getattr(self, f'best_{name}_val')
                    if cur_best is None:
                        is_better = True
                    elif min_improve > 0.0:
                        is_better = (cur_best - val) > min_improve
                    else:
                        is_better = val < cur_best

                    if is_better:
                        setattr(self, f'best_{name}_val', float(val))
                        setattr(self, f'best_{name}_epoch', int(epoch_idx))
                        if save_dual:
                            torch.save(self.model.state_dict(), os.path.join(path, filename))

                _update_best('pred', m['val_pred'], epoch + 1, 'checkpoint_pred.pth')
                _update_best('total', m['val_total'], epoch + 1, 'checkpoint_total.pth')
                _update_best('struct_raw', m['val_struct_raw'], epoch + 1, 'checkpoint_struct_raw.pth')
                _update_best('struct_scaled', m['val_struct_scaled'], epoch + 1, 'checkpoint_struct_scaled.pth')
                _update_best('mse', m['mse'], epoch + 1, 'checkpoint_mse.pth')
                _update_best('mae', m['mae'], epoch + 1, 'checkpoint_mae.pth')

                monitor_val = float(m[monitor_key])
                early_stopping(monitor_val, self.model, path)

                if self.best_metric_val is None:
                    self.best_metric_val = float(monitor_val)
                    self.best_metric_epoch = epoch + 1
                else:
                    if min_improve > 0.0:
                        if (self.best_metric_val - monitor_val) > min_improve:
                            self.best_metric_val = float(monitor_val)
                            self.best_metric_epoch = epoch + 1
                    else:
                        if monitor_val < self.best_metric_val:
                            self.best_metric_val = float(monitor_val)
                            self.best_metric_epoch = epoch + 1

            else:
                early_stopping(ep_avg_pred, self.model, path)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

            adjust_learning_rate(opt_main, epoch + 1, self.args)
            if opt_hg is not None:
                adjust_learning_rate(opt_hg, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        label = {
            'val_total': 'ValTotal',
            'val_pred': 'ValPred',
            'val_struct_raw': 'ValStructRaw',
            'val_struct_scaled': 'ValStructScaled',
            'mse': 'ValMSE',
            'mae': 'ValMAE',
        }.get(monitor_key, 'ValTotal')

        print(
            f"[Info] Best epoch by {label}: {self.best_metric_epoch}, "
            f"best_value={self.best_metric_val}"
        )

        if save_dual:
            print(f"[Info] Best (pred)          @ epoch {self.best_pred_epoch}: {self.best_pred_val}")
            print(f"[Info] Best (total)         @ epoch {self.best_total_epoch}: {self.best_total_val}")
            print(f"[Info] Best (struct_raw)    @ epoch {self.best_struct_raw_epoch}: {self.best_struct_raw_val}")
            print(f"[Info] Best (struct_scaled) @ epoch {self.best_struct_scaled_epoch}: {self.best_struct_scaled_val}")
            print(f"[Info] Best (mse)           @ epoch {self.best_mse_epoch}: {self.best_mse_val}")
            print(f"[Info] Best (mae)           @ epoch {self.best_mae_epoch}: {self.best_mae_val}")

        if self.best_test_mse is not None:
            print(
                f"[Info] Best Test@epoch{self.best_test_epoch}: "
                f"MSE={self.best_test_mse:.6f}, MAE={self.best_test_mae:.6f}"
            )
        if self.last_test_mse is not None:
            print(
                f"[Info] Last Test@epoch{self.last_test_epoch}: "
                f"MSE={self.last_test_mse:.6f}, MAE={self.last_test_mae:.6f}"
            )

        return {
            'best_epoch': int(self.best_metric_epoch),
            'metric_name': label,
            'metric_value': float(self.best_metric_val if self.best_metric_val is not None else -1.0),
            'model': self.model,
            'best_test_mse': float(self.best_test_mse) if self.best_test_mse is not None else None,
            'best_test_mae': float(self.best_test_mae) if self.best_test_mae is not None else None,
            'best_test_epoch': int(self.best_test_epoch),
            'last_test_mse': float(self.last_test_mse) if self.last_test_mse is not None else None,
            'last_test_mae': float(self.last_test_mae) if self.last_test_mae is not None else None,
            'last_test_epoch': int(self.last_test_epoch),
        }

    # ---------- 测试 ----------
    # ---------- 测试 ----------
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')

        if test:
            print('loading model')
            ckpt_path = None

            cand = getattr(self.args, 'checkpoint_file', None)
            if isinstance(cand, str) and os.path.isfile(cand):
                ckpt_path = cand

            if ckpt_path is None:
                cand = getattr(self.args, 'checkpoints', None)
                if isinstance(cand, str) and os.path.isfile(cand):
                    ckpt_path = cand

            if ckpt_path is None:
                ckpt_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')

            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and ('state_dict' in state) and isinstance(state['state_dict'], dict):
                    state = state['state_dict']

                try:
                    self.model.load_state_dict(state, strict=True)
                except Exception:
                    if isinstance(state, dict):
                        new_state = {}
                        model_is_dp = hasattr(self.model, 'module')
                        for k, v in state.items():
                            if model_is_dp and (not k.startswith('module.')):
                                new_state['module.' + k] = v
                            elif (not model_is_dp) and k.startswith('module.'):
                                new_state[k[len('module.'):]] = v
                            else:
                                new_state[k] = v
                        self.model.load_state_dict(new_state, strict=False)

                print(f"[Info] Loaded checkpoint from '{ckpt_path}'")
            else:
                print(f"[Warn] checkpoint not found at '{ckpt_path}', use random-initialized model.", flush=True)

        preds, trues = [], []
        self.model.eval()

        # === [HG-STORE] 本次 test 周期启用收集 ===
        m = self._unwrap_model(self.model)
        m._hg_store = {
            "proto": [],
            "tower": [],
            "tower_base": None,
            "tower_repr": [],
            "content": [],
        }

        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_cycle = None
                else:
                    batch_x, batch_y = batch[0], batch[1]
                    batch_x_mark = batch[2] if len(batch) > 2 else None
                    batch_y_mark = batch[3] if len(batch) > 3 else None
                    batch_cycle = batch[4] if len(batch) > 4 else None

                batch_x = self._to_device_if_needed(batch_x, self.device)
                batch_y = self._to_device_if_needed(batch_y, self.device)
                batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
                batch_y_mark = self._to_device_if_needed(batch_y_mark, self.device)
                batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                dec_inp = None
                if self.args.output_attention:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
                    dec_inp = self._to_device_if_needed(dec_inp, self.device)

                out, _loss2_raw = self._forward_with_loss2(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle
                )

                f_dim = -1 if self.args.features == 'MS' else 0
                out_sel = out[:, -self.args.pred_len:, f_dim:]
                tgt_sel = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(out_sel.detach().cpu().numpy())
                trues.append(tgt_sel.detach().cpu().numpy())

        # === 保存 hg_store ===
        m = self._unwrap_model(self.model)
        hg_store = getattr(m, "_hg_store", None)

        if not isinstance(hg_store, dict):
            print("[HG-STORE] invalid _hg_store; skip.")
        else:
            print("[HG-STORE] keys:", list(hg_store.keys()))
            for k in ("proto", "tower", "tower_repr", "content"):
                v = hg_store.get(k, None)
                if isinstance(v, list):
                    none_cnt = sum(x is None for x in v)
                    print(f"[HG-STORE] {k}: len={len(v)} none={none_cnt}")

            folder_path = os.path.join('./results', setting)
            os.makedirs(folder_path, exist_ok=True)
            out_path = os.path.join(folder_path, 'hg_store_test.pt')
            torch.save(hg_store, out_path)
            print(f"[HG-STORE] saved to: {out_path}")

        m._hg_store = None

        # ===== 汇总预测 =====
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds, trues = self._ensure_3d_curve_arrays(preds, trues)

        # ===== 新增：搜索最佳通道 + 最佳窗口 =====
        select_info = self._select_best_channel_and_window(preds, trues)

        # ===== 新增：完整曲线保存 =====
        complete_curve_path = self._save_complete_curves(setting, preds, trues, select_info)
        ranking_paths = self._save_rankings_csv(setting, select_info)

        # ===== 新增：最佳通道的完整曲线 =====
        best_channel_info = self._save_best_channel_full_curve(setting, preds, trues, select_info)

        # ===== 新增：最佳通道中的最佳窗口 =====
        best_window_channel_info = self._save_best_window_in_best_channel(setting, preds, trues, select_info)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        print(
            f"[BestChannel] idx={select_info['best_channel_idx']} | "
            f"{select_info['metric_name']}={select_info['best_channel_score']:.6f}"
        )
        print(
            f"[BestWindowInBestChannel] window={select_info['best_window_in_best_channel_idx']} | "
            f"channel={select_info['best_channel_idx']} | "
            f"{select_info['metric_name']}={select_info['best_window_in_best_channel_score']:.6f}"
        )
        print(
            f"[BestGlobalPair] window={select_info['best_pair_window_idx']} | "
            f"channel={select_info['best_pair_channel_idx']} | "
            f"{select_info['metric_name']}={select_info['best_pair_score']:.6f}"
        )

        # TB 标量
        self.tb_writer = self._init_tb_writer(setting)

        def _as_scalar(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.size == 0:
                return 0.0
            if x.ndim == 0:
                return float(x)
            return float(x.mean())

        try:
            self.tb_writer.add_scalar('test/mse', _as_scalar(mse), 0)
            self.tb_writer.add_scalar('test/mae', _as_scalar(mae), 0)
            self.tb_writer.add_scalar('test/rmse', _as_scalar(rmse), 0)
            self.tb_writer.add_scalar('test/mape', _as_scalar(mape), 0)
            self.tb_writer.add_scalar('test/mspe', _as_scalar(mspe), 0)
            self.tb_writer.add_scalar('test/rse', _as_scalar(rse), 0)
            self.tb_writer.add_scalar('test/corr', _as_scalar(corr), 0)

            self.tb_writer.add_scalar('best/channel_idx', float(select_info['best_channel_idx']), 0)
            self.tb_writer.add_scalar('best/channel_score', float(select_info['best_channel_score']), 0)
            self.tb_writer.add_scalar('best/window_in_best_channel_idx',
                                      float(select_info['best_window_in_best_channel_idx']), 0)
            self.tb_writer.add_scalar('best/window_in_best_channel_score',
                                      float(select_info['best_window_in_best_channel_score']), 0)
            self.tb_writer.add_scalar('best/global_pair_window_idx', float(select_info['best_pair_window_idx']), 0)
            self.tb_writer.add_scalar('best/global_pair_channel_idx', float(select_info['best_pair_channel_idx']), 0)
            self.tb_writer.add_scalar('best/global_pair_score', float(select_info['best_pair_score']), 0)
        finally:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None

        return {
            'test/mse': _as_scalar(mse),
            'test/mae': _as_scalar(mae),
            'test/rmse': _as_scalar(rmse),
            'test/mape': _as_scalar(mape),
            'test/mspe': _as_scalar(mspe),
            'test/rse': _as_scalar(rse),
            'test/corr': _as_scalar(corr),

            'select_metric': select_info['metric_name'],

            'best_channel_idx': int(select_info['best_channel_idx']),
            'best_channel_score': float(select_info['best_channel_score']),

            'best_window_idx': int(select_info['best_window_idx']),
            'best_window_score': float(select_info['best_window_score']),

            'best_window_in_best_channel_idx': int(select_info['best_window_in_best_channel_idx']),
            'best_window_in_best_channel_score': float(select_info['best_window_in_best_channel_score']),

            'best_pair_window_idx': int(select_info['best_pair_window_idx']),
            'best_pair_channel_idx': int(select_info['best_pair_channel_idx']),
            'best_pair_score': float(select_info['best_pair_score']),

            'complete_curve_file': complete_curve_path,
            'channel_ranking_csv': ranking_paths['channel_csv'],
            'window_ranking_csv': ranking_paths['window_csv'],

            'best_channel_full_curve_png': best_channel_info['best_channel_full_curve_png'],
            'best_channel_full_curve_csv': best_channel_info['best_channel_full_curve_csv'],
            'best_channel_full_curve_npz': best_channel_info['best_channel_full_curve_npz'],

            'best_window_in_best_channel_png': best_window_channel_info['best_window_in_best_channel_png'],
            'best_window_in_best_channel_csv': best_window_channel_info['best_window_in_best_channel_csv'],
            'best_window_in_best_channel_npz': best_window_channel_info['best_window_in_best_channel_npz'],
        }

    # （可选）predict()：保留更稳
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data('pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if os.path.isfile(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in pred_loader:
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_cycle = None
                else:
                    batch_x, batch_y = batch[0], batch[1]
                    batch_x_mark = batch[2] if len(batch) > 2 else None
                    batch_y_mark = batch[3] if len(batch) > 3 else None
                    batch_cycle = batch[4] if len(batch) > 4 else None

                batch_x = self._to_device_if_needed(batch_x, self.device)
                batch_x_mark = self._to_device_if_needed(batch_x_mark, self.device)
                batch_cycle = self._to_device_if_needed(batch_cycle, self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None

                out, _ = self._forward_with_loss2(batch_x, batch_x_mark, None, None, batch_cycle)

                f_dim = -1 if self.args.features == 'MS' else 0
                out = out[:, -self.args.pred_len:, f_dim:]
                preds.append(out.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        print(f"[Predict] saved to: {os.path.join(folder_path, 'real_prediction.npy')}")

        return preds