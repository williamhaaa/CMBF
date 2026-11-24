# -*- coding: utf-8 -*-
"""
run_STHyper1.py
Version: v1.9.5  (2025-12-06)

【版本号与修改记录（changelog）】
- v1.9.5 [Fix][TestSettingFromCkptFile]
  1) 改进 --checkpoint / --checkpoints 解析方式：
     • --checkpoints: 作为训练阶段使用的“根目录”，默认值 './checkpoints/'；
     • --checkpoint: 仅作为测试阶段的别名输入，解析后赋值给 args.checkpoints。
  2) 测试模式 (is_training=0) 下，_load_args_from_json_for_test：
     • 使用“命令行提供的 checkpoint 路径”推导 ckpt_dir，并记录
       _test_setting_from_ckpt = os.path.basename(ckpt_dir)；
     • 从 ckpt_dir/args.json 读取训练时参数并生成 json_args；
     • 将 _test_setting_from_ckpt 写回 json_args，供后续使用。
  3) 测试分支不再尝试从 args.checkpoints 推导 setting，而是优先使用
     args._test_setting_from_ckpt；若不存在该字段再回退到 format(...) 方式。
  4) 训练模式 (is_training=1) 的行为保持与 v1.9.2 完全一致。

- v1.9.4 [TestSettingFromCkptDir]
  （已由 v1.9.5 覆盖，但核心思路保留）

- v1.9.3 [TestFromJSON + ArgsPostprocess]
  1) 新增 _load_args_from_json_for_test：
     • 当 is_training=0 且 --checkpoints 指向某个 checkpoint.pth 或其目录时，
       自动在同目录查找 args.json，并将其中的参数作为“基准配置”；
     • 仅用命令行覆写少量运行相关字段（is_training / train_only / do_predict /
       use_gpu / use_multi_gpu / gpu / devices / test_flop），其余全部沿用训练时配置。
  2) 抽取 _postprocess_args(args) 统一处理：LR 映射、列表解析、缺省派生、
     GPU/AMP 选择、t_d_model/event_layers 映射以及 hyper_s*/win_s* 同步。
"""

import argparse
import os
import re
import sys
import json
import random
import numpy as np
import torch

from exp.exp_main import Exp_Main

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ---------------- NNI（可选） ----------------
try:
    import nni
except Exception:
    nni = None

# ---------------- Seed ----------------
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------- Utils ----------------
def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ['1', 'true', 'yes', 'y', 't']
    return False


def str_to_int_list(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    if isinstance(s, str):
        nums = re.findall(r'-?\d+', s)
        return [int(x) for x in nums] if nums else []
    try:
        return [int(s)]
    except Exception:
        raise TypeError(f'Expect str/list/tuple/int/None, got {type(s)}')


def str_to_float_list(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    if isinstance(s, str):
        nums = re.findall(r'-?\d+(?:\.\d+)?', s)
        return [float(x) for x in nums] if nums else []
    try:
        return [float(s)]
    except Exception:
        raise TypeError(f'Expect str/list/tuple/float/None, got {type(s)}')


def _print_merged_args(cli_args, nni_params, final_args):
    print("PRINT CLI ARGS:")
    print(cli_args)
    if nni_params:
        print("PRINT NNI PARAMS:")
        print(json.dumps(nni_params, ensure_ascii=False, sort_keys=True, indent=2))
    print("PRINT MERGED ARGS (EFFECTIVE):")
    print(final_args)


def _save_args_json(path, args_namespace):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vars(args_namespace), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'[Warn] save args.json failed: {e}', file=sys.stderr)


# === AMP: 自动选择精度 ===
def _auto_pick_amp_dtype(req: str) -> str:
    if req == 'off':
        return 'off'
    if not torch.cuda.is_available():
        return 'off'
    if req in ('bf16', 'fp16'):
        return req
    # auto
    try:
        if torch.cuda.is_bf16_supported():
            return 'bf16'
    except Exception:
        pass
    return 'fp16'


def _merge_time_d_model(args):
    """将 t_d_model 映射到 d_model（AHL 与部分模块仍访问 d_model）。"""
    td = getattr(args, 't_d_model', 128)
    if td is not None and td > 0:
        args.d_model = td


def _sync_event_layers(args):
    """将 event_layers 映射到 t_ee_layers，兼容旧的 E2E 塔深度字段。"""
    ev = getattr(args, 'event_layers', None)
    if ev is not None and ev > 0:
        args.t_ee_layers = ev


def _apply_time_hyperparams_from_search(args):
    """
    使用 win_s1/win_s2 和 hyper_s1/hyper_s2/hyper_s3
    统一回填时间端与兼容字段：
      - t_window_size, t_hyper_num
      - window_size, hyper_num
    """
    # ----- 窗口大小 -----
    win_vals = []
    for name in ['win_s1', 'win_s2']:
        if hasattr(args, name):
            v = getattr(args, name)
            if v is None:
                continue
            try:
                v_int = int(v)
            except Exception:
                continue
            if v_int > 0:
                win_vals.append(v_int)

    if win_vals:
        args.t_window_size = list(win_vals)
        args.window_size = list(win_vals)

    # ----- 超边个数 -----
    hyper_vals = []
    for name in ['hyper_s1', 'hyper_s2', 'hyper_s3']:
        if hasattr(args, name):
            v = getattr(args, name)
            if v is None:
                continue
            try:
                v_int = int(v)
            except Exception:
                continue
            if v_int > 0:
                hyper_vals.append(v_int)

    if hyper_vals:
        args.t_hyper_num = list(hyper_vals)
        args.hyper_num = list(hyper_vals)

    return args


def _postprocess_args(args):
    """
    统一对“最终 args”做后处理：LR 映射、列表解析、缺省派生、GPU/AMP 配置、
    t_d_model/event_layers 映射以及时间多尺度 hyper_s*/win_s* 同步。
    训练和测试（含 JSON 合并后）共用。
    """
    # ---------- LR 兼容映射 ----------
    if hasattr(args, 'lr_sched') and getattr(args, 'lr_sched', None) not in (None, '', 'none'):
        args.lradj = args.lr_sched
    if not hasattr(args, 'lradj') or args.lradj is None:
        args.lradj = 'type1'
    if not hasattr(args, 'cosine_tmax'):
        args.cosine_tmax = 10
    if not hasattr(args, 'eta_min'):
        args.eta_min = 1e-6

    # ---------- 列表解析 ----------
    t_ws = getattr(args, 't_window_size', None)
    args.t_window_size = str_to_int_list(t_ws) if t_ws not in (None, '') else []

    t_hn = getattr(args, 't_hyper_num', None)
    args.t_hyper_num = str_to_int_list(t_hn) if t_hn not in (None, '') else []

    s_hn = getattr(args, 's_hyper_num', None)
    args.s_hyper_num = str_to_int_list(s_hn) if s_hn not in (None, '') else []

    if hasattr(args, 'window_size'):
        ws = getattr(args, 'window_size', None)
        args.window_size = str_to_int_list(ws) if ws not in (None, '') else []
    if hasattr(args, 'hyper_num'):
        hn = getattr(args, 'hyper_num', None)
        args.hyper_num = str_to_int_list(hn) if hn not in (None, '') else []

    if isinstance(getattr(args, 'spatial_hyper_num', []), str) and getattr(args, 'spatial_hyper_num', '').strip():
        args.spatial_hyper_num = str_to_int_list(args.spatial_hyper_num)
    elif isinstance(getattr(args, 'spatial_hyper_num', []), (list, tuple)):
        args.spatial_hyper_num = [int(x) for x in args.spatial_hyper_num]
    else:
        args.spatial_hyper_num = []

    # ---------- 构图器缺省 ----------
    if not getattr(args, 't_cscm', None):
        args.t_cscm = getattr(args, 'CSCM', 'Bottleneck_Construct')

    # ---------- 空间超边 ----------
    if (not args.s_hyper_num) and getattr(args, 'spatial_hyper_num', []):
        args.s_hyper_num = args.spatial_hyper_num

    # ---------- 时间/空间构图参数缺省 ----------
    if getattr(args, 't_alpha', None) is None:
        args.t_alpha = getattr(args, 'alpha', 2.0)
    if getattr(args, 't_kprime', None) is None:
        args.t_kprime = getattr(args, 'kprime', 16)
    if getattr(args, 't_lambda_temp', None) is None:
        args.t_lambda_temp = getattr(args, 'lambda_temp', 1.0)
    if getattr(args, 't_beta_thr', None) is None:
        args.t_beta_thr = getattr(args, 'beta', 0.0)
    if getattr(args, 's_beta_thr', None) is None:
        args.s_beta_thr = getattr(args, 's_beta', 0.01)

    # ---------- GPU ----------
    args.use_gpu = True if torch.cuda.is_available() and getattr(args, 'use_gpu', True) else False
    if args.use_gpu and (not getattr(args, 'use_multi_gpu', False)):
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(getattr(args, 'gpu', 0))
            torch.cuda.set_device(getattr(args, 'gpu', 0))
            print(f'Use GPU: cuda:{getattr(args, "gpu", 0)}')
        except Exception:
            print('Warn: set_device failed, fallback to default CUDA device', file=sys.stderr)
    if args.use_gpu and getattr(args, 'use_multi_gpu', False):
        args.devices = getattr(args, 'devices', '0').replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids if id_.isdigit()]
        if len(args.device_ids) == 0:
            args.device_ids = [0]
        args.gpu = args.device_ids[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.device_ids))
        print(f'Use Multi-GPU: {args.device_ids}')

    # ---------- AMP ----------
    args.amp_dtype_eff = _auto_pick_amp_dtype(getattr(args, 'amp_dtype', 'auto'))
    if not getattr(args, 'use_amp', False):
        args.amp_dtype_eff = 'off'
    if args.amp_dtype_eff != 'fp16':
        args.use_grad_scaler = False

    # ---------- d_model / 事件层数 ----------
    _merge_time_d_model(args)
    _sync_event_layers(args)

    # ---------- 多尺度超图：以 hyper_s*/win_s* 为主 ----------
    _apply_time_hyperparams_from_search(args)

    return args


def _load_args_from_json_for_test(args):
    """
    测试模式下（is_training=0），尝试从 checkpoint 所在目录的 args.json 读取训练时参数：
      1) 仅在 is_training=0 时生效；
      2) 要求 args.checkpoints 指向 checkpoint.pth 或其所在目录；
      3) 读取到 JSON 后，用 JSON 构造新的 Namespace；
      4) 用命令行覆写少数“运行相关”的字段（含 cycle / checkpoints），
         其余全部沿用训练时配置；
      5) 如果最终 cycle<=0，再做一次安全兜底，避免 forward 报错；
      6) 从 checkpoint 所在目录反推出 setting，写入 _test_setting_from_ckpt。
    """
    # 训练模式：完全不从 JSON 读，保证训练行为不变
    if getattr(args, 'is_training', 1) != 0:
        return args

    ckpt_root = getattr(args, 'checkpoints', None)
    if not ckpt_root:
        return args

    # 1) 判定 ckpt_root 是文件还是目录
    if os.path.isfile(ckpt_root):
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_root))
    elif os.path.isdir(ckpt_root):
        ckpt_dir = os.path.abspath(ckpt_root)
    else:
        print(f"[Info] checkpoints path '{ckpt_root}' not found, skip loading args.json.", file=sys.stderr)
        return args

    json_path = os.path.join(ckpt_dir, 'args.json')
    if not os.path.isfile(json_path):
        print(f"[Info] args.json not found at '{json_path}', skip loading.", file=sys.stderr)
        return args

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[Warn] load args.json failed: {e}", file=sys.stderr)
        return args

    print(f"[Info] Loaded args.json from '{json_path}' for test.")

    # 2) 以 JSON 为基础构造新的 Namespace（训练时完整配置）
    json_args = argparse.Namespace(**cfg)

    # === 关键：从 checkpoint 目录反推 setting 名 ===
    # 比如 ckpt_dir = .../checkpoints/ELC_EventMSHyper_sl96_pl720_te_EventMSHyperV1_custom_.../
    # 那么 setting_from_ckpt = 'ELC_EventMSHyper_sl96_pl720_te_EventMSHyperV1_custom_...'
    setting_from_ckpt = os.path.basename(os.path.normpath(ckpt_dir))
    setattr(json_args, '_test_setting_from_ckpt', setting_from_ckpt)
    # 也顺手记一下真正的 ckpt 目录（可选）
    setattr(json_args, '_test_ckpt_dir', ckpt_dir)

    # 3) 用命令行参数覆写少量“运行相关字段”
    override_keys = [
        'is_training',
        'train_only',
        'do_predict',
        'use_gpu',
        'use_multi_gpu',
        'gpu',
        'devices',
        'test_flop',
        'checkpoints',   # 允许你指定具体哪个 .pth
        'cycle',         # 允许你用 CLI 改周期（例如 168）
    ]
    for k in override_keys:
        if hasattr(args, k):
            setattr(json_args, k, getattr(args, k))

    # 4) 安全兜底：如果 cycle 仍然 <= 0，再做一次合理修复
    if getattr(json_args, 'cycle', 0) <= 0:
        cli_cycle = getattr(args, 'cycle', 0)
        if cli_cycle and cli_cycle > 0:
            json_args.cycle = int(cli_cycle)
        else:
            data_name = str(getattr(json_args, 'data', '')).lower()
            freq = str(getattr(json_args, 'freq', 'h')).lower()

            if data_name in ['custom', 'electricity'] and freq == 'h':
                json_args.cycle = 168
            elif freq == 'h':
                json_args.cycle = 24
            else:
                json_args.cycle = max(int(getattr(json_args, 'seq_len', 24)), 1)

        print(f"[Info] cycle<=0 in args.json, reset cycle={json_args.cycle}")

    return json_args


# ---------------- Argparse ----------------
parser = argparse.ArgumentParser(description='ASTHyper / STHyper Forecasting (v1.9.5)')

# ===== basic config =====
parser.add_argument('--is_training', type=int, required=True, default=1)
parser.add_argument('--train_only', type=_to_bool, required=False, default=False)
parser.add_argument('--model_id', type=str, required=False, default='test')
parser.add_argument('--model', type=str, required=False, default='EventMSHyperV1')

# ===== data loader =====
parser.add_argument('--data', type=str, required=False, default='ETTm1')
parser.add_argument('--root_path', type=str, default='./data/ETT/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='training root dir for checkpoints (train) or ckpt path (test)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='alias of --checkpoints, usually full path to checkpoint.pth in test mode')

# ===== forecasting task =====
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--cycle', type=int, default=24)

# ===== channel spec =====
parser.add_argument('--individual', type=_to_bool, default=True)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)

# ===== optimization =====
parser.add_argument('--num_workers', type=int, default=7, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2)
parser.add_argument('--train_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='mae')

# ==== LR schedule（兼容老的 lradj & 新的 lr_sched）====
parser.add_argument(
    '--lradj',
    type=str,
    default='type1',
    choices=['none', 'type1', 'type2', 'cosine', 'plateau', 'step']
)
parser.add_argument(
    '--lr_sched',
    type=str,
    default='none',
    choices=['none', 'type1', 'type2', 'cosine', 'plateau', 'step']
)
parser.add_argument('--cosine_tmax', type=int, default=10)
parser.add_argument('--eta_min', type=float, default=1e-6)

# ===== AMP =====
parser.add_argument('--use_amp', type=_to_bool, default=False)
parser.add_argument('--amp_dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'off'])
parser.add_argument('--use_grad_scaler', type=_to_bool, default=True)
parser.add_argument('--amp_keep_bn_fp32', type=_to_bool, default=True)

# ===== GPU =====
parser.add_argument('--use_gpu', type=_to_bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', type=_to_bool, default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')
parser.add_argument('--test_flop', type=_to_bool, default=False)

# ===== Hyper-graph / time branch =====
parser.add_argument('--CSCM', type=str, default='Bottleneck_Construct')
parser.add_argument('--t_cscm', type=str, default='')
parser.add_argument('--t_window_size', type=str, default='')
parser.add_argument('--t_hyper_num', type=str, default='')

# 概率式构图（时间）
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--kprime', type=int, default=16)
parser.add_argument('--t_kprime', type=int, default=None)
parser.add_argument('--lambda_temp', type=float, default=1.0)
parser.add_argument('--t_lambda_temp', type=float, default=None)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--t_beta_thr', type=float, default=0.1)
parser.add_argument('--t_gamma_dist', type=float, default=0.0)
parser.add_argument('--t_distance_metric', type=str, default='l2', choices=['l2', 'cos'])
parser.add_argument('--t_alpha', type=float, default=2.0)

# 时间端 HGNN 的节点约束权重
parser.add_argument('--lambda_nc_time', type=float, default=0.3)

# 与搜索空间对接的多尺度配置
parser.add_argument('--hyper_s1', type=int, default=20, help='scale-1 hyperedge num (search)')
parser.add_argument('--hyper_s2', type=int, default=20, help='scale-2 hyperedge num (search)')
parser.add_argument('--hyper_s3', type=int, default=12, help='scale-3 hyperedge num (search)')
parser.add_argument('--win_s1', type=int, default=2, help='scale-1 window size (search)')
parser.add_argument('--win_s2', type=int, default=4, help='scale-2 window size (search)')

# 时间端 d_model
parser.add_argument('--t_d_model', type=int, default=8, help='[search] time-branch d_model')

# 事件塔（时间端，容量）
parser.add_argument('--t_ee_layers', type=int, default=3)
parser.add_argument('--t_ee_dropout', type=float, default=0.15)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--event_d_model', type=int, default=128)
parser.add_argument('--event_ffn_dim', type=int, default=128)

# 新事件塔相关
parser.add_argument('--event_layers', type=int, default=4)   # 若给定，将覆盖 t_ee_layers
parser.add_argument('--num_event_protos', type=int, default=16)
parser.add_argument('--proto_topk', type=int, default=8)
parser.add_argument('--event_topk', type=int, default=16)
parser.add_argument('--cheby_order', type=int, default=1)
parser.add_argument('--use_event_propagation', type=_to_bool, default=True)

# 解码器参数
parser.add_argument('--dec_heads', type=int, default=4)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--dec_ffn_dim', type=int, default=256)

# 旧数据管线兼容
parser.add_argument('--embed', type=str, default='timeF', choices=['timeF', 'fixed', 'none'])
parser.add_argument('--output_attention', type=_to_bool, default=False)
parser.add_argument('--do_predict', type=_to_bool, default=False)

# ===== 空间端 =====
parser.add_argument('--use_spatial_branch', type=_to_bool, default=True)
parser.add_argument('--spatial_independent_norm', type=_to_bool, default=True)
parser.add_argument('--s_hyper_num', type=str, default='')
parser.add_argument('--spatial_hyper_num', type=str, default='')
parser.add_argument('--s_alpha', type=float, default=3.0)
parser.add_argument('--s_kprime', type=float, default=8)
parser.add_argument('--s_beta', type=float, default=0.01)
parser.add_argument('--s_lambda_temp', type=float, default=None)
parser.add_argument('--s_beta_thr', type=float, default=None)
parser.add_argument('--s_gamma_dist', type=float, default=0.0)
parser.add_argument('--s_distance_metric', type=str, default='l2', choices=['l2', 'cos'])
parser.add_argument('--s_ee_layers', type=int, default=2)
parser.add_argument('--s_ee_dropout', type=float, default=0.15)

parser.add_argument('--global_tri_stage', type=_to_bool, default=True)
parser.add_argument('--gts_layers', type=int, default=2)
parser.add_argument('--gts_heads', type=int, default=4)
parser.add_argument('--gts_topk', type=int, default=12)
parser.add_argument('--gts_mem_slots', type=int, default=16)
parser.add_argument('--gts_len', type=int, default=None)
parser.add_argument('--gts_mix', type=float, default=0.5)
parser.add_argument('--gts_eta', type=float, default=-1.0)
parser.add_argument('--gts_tau', type=float, default=0.01)
parser.add_argument('--gts_dropout', type=float, default=0.15)
parser.add_argument('--gts_take_top_time', type=int, default=1)
parser.add_argument('--gts_take_top_spatial', type=int, default=1)

# ===== 正则权重 =====
parser.add_argument('--w_e2e', type=float, default=1e-1)
parser.add_argument('--w_gp', type=float, default=5e-1)
parser.add_argument('--w_sp', type=float, default=1e-1)
parser.add_argument('--w_e2e_sp', type=float, default=1e-1)
parser.add_argument('--w_gp_sp', type=float, default=5e-1)
parser.add_argument('--w_sp_sp', type=float, default=1e-1)
parser.add_argument('--w_ee_smooth', type=float, default=1e-1)
parser.add_argument('--w_ee_smooth_sp', type=float, default=1e-1)
parser.add_argument('--w_gts_smooth', type=float, default=2e-1)
parser.add_argument('--w_fusion_entropy', type=float, default=1e-1)



# ======= 兼容占位参数 =======
parser.add_argument('--inner_size', type=int, default=5, help='[compat] may be used by CSCM/AHL')
parser.add_argument('--k', type=int, default=3, help='[compat]')
parser.add_argument('--d_model', type=int, default=128, help='[compat] time-branch d_model')
parser.add_argument('--d_bottleneck', type=int, default=256, help='[compat]')
parser.add_argument('--d_ff', type=int, default=256, help='[compat]')
parser.add_argument('--dropout', type=float, default=0.15, help='[compat]')


parser.add_argument('--use_event', type=_to_bool, default=True)
parser.add_argument('--use_node', type=_to_bool, default=True)
parser.add_argument('--use_linear', type=_to_bool, default=True)
parser.add_argument('--slot_num_base', type=int, default=56)
parser.add_argument('--slot_num_list', type=int, default=None)
parser.add_argument('--slot_ratio', type=float, default=0.7)
parser.add_argument('--event_topk_ratio', type=float, default=0.5)
parser.add_argument('--slot_hidden_dim', type=int, default=128)

args = parser.parse_args()

# ---- 处理 --checkpoint 别名（如果提供则覆盖 checkpoints）----
if getattr(args, 'checkpoint', None):
    args.checkpoints = args.checkpoint

# ---------------- NNI 覆盖（训练场景） ----------------
raw_cli_args_repr = repr(args)
nni_params = None
if nni is not None:
    try:
        nni_params = nni.get_next_parameter()
    except Exception:
        nni_params = None

if isinstance(nni_params, dict):
    for k, v in nni_params.items():
        setattr(args, k, v)

    # 标准化布尔
    for bk in [
        'train_only', 'individual',
        'use_spatial_branch', 'spatial_independent_norm',
        'global_tri_stage',
        'use_amp', 'use_grad_scaler', 'amp_keep_bn_fp32',
        'use_gpu', 'use_multi_gpu',
        'output_attention', 'do_predict',
        'use_event_propagation',
    ]:
        if hasattr(args, bk):
            setattr(args, bk, _to_bool(getattr(args, bk)))

    # NNI 下强制单次
    args.itr = 1

# ---------------- 测试模式：优先从 args.json 读取训练时参数 ----------------
args = _load_args_from_json_for_test(args)

# ---------------- 统一后处理：LR / 列表 / GPU / AMP / 多尺度超图 等 ----------------
args = _postprocess_args(args)

# ---------------- 打印最终生效参数 ----------------
_print_merged_args(raw_cli_args_repr, nni_params, args)

# DEBUG：输出最终真正生效的多尺度参数
print(f"[DEBUG] final t_window_size = {getattr(args, 't_window_size', None)}")
print(f"[DEBUG] final t_hyper_num   = {getattr(args, 't_hyper_num', None)}")
print(f"[DEBUG] final window_size   = {getattr(args, 'window_size', None)}")
print(f"[DEBUG] final hyper_num     = {getattr(args, 'hyper_num', None)}")

# 组装 setting 标签
ws_tag = f"ws[{','.join(map(str, getattr(args, 't_window_size', [])))}]" if getattr(args, 't_window_size', []) else "ws[auto]"
hn_tag = f"hn[{','.join(map(str, getattr(args, 't_hyper_num', [])))}]" if getattr(args, 't_hyper_num', []) else "hn[auto]"
sp_tag = f"sp{int(bool(args.use_spatial_branch))}"
sp_hn_tag = f"sp_hn[{','.join(map(str, args.s_hyper_num))}]" if args.s_hyper_num else "sp_hn[auto]"
e2e_tag = f"tEE-L{getattr(args, 't_ee_layers', 0)}"
amp_tag = f"amp[{args.amp_dtype_eff}]" if args.use_amp and args.amp_dtype_eff != 'off' else "amp[off]"

# 事件塔与解码器标记
ev_tag = f"evL{getattr(args, 'event_layers', getattr(args, 't_ee_layers', 0))}_K{getattr(args, 'num_event_protos', 0)}"
dec_tag = f"decH{getattr(args, 'dec_heads', 0)}L{getattr(args, 'dec_layers', 0)}"

# ---------------- Run ----------------
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            getattr(args, 'd_model', 128), getattr(args, 't_d_model', 128),
            getattr(args, 'n_heads', 8),
            getattr(args, 'e_layers', 2), getattr(args, 'd_layers', 1),
            getattr(args, 'd_ff', 256), getattr(args, 'factor', 1),
            getattr(args, 'embed', 'timeF'),
            ws_tag, hn_tag, e2e_tag, ev_tag, dec_tag, sp_tag, sp_hn_tag, amp_tag
        )

        ckpt_dir = os.path.join(args.checkpoints, setting)
        os.makedirs(ckpt_dir, exist_ok=True)
        _save_args_json(os.path.join(ckpt_dir, 'args.json'), args)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        train_ret = exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_ret = exp.test(setting)

            # ===== NNI 上报 =====
            if nni is not None:
                try:
                    payload = {}
                    if isinstance(train_ret, dict):
                        for k in [
                            'best_test_mse', 'last_test_mse',
                            'best_test_mae', 'last_test_mae'
                        ]:
                            if k in train_ret and train_ret[k] is not None:
                                payload[k] = float(train_ret[k])

                    if 'best_test_mse' in payload:
                        payload['default'] = payload['best_test_mse']
                    elif 'last_test_mse' in payload:
                        payload['default'] = payload['last_test_mse']
                    else:
                        if isinstance(test_ret, dict) and 'test/mse' in test_ret:
                            payload['default'] = float(test_ret['test/mse'])
                        else:
                            payload['default'] = 0.0

                    print(f"[NNI] report_final_result: {payload}")
                    nni.report_final_result(payload)
                except Exception as e:
                    print(f"[Warn] NNI report_final_result failed: {e}", file=sys.stderr)

        torch.cuda.empty_cache()
else:
    ii = 0

    # 测试模式：优先使用 _test_setting_from_ckpt（由命令行 checkpoint 推导）
    setting = getattr(args, '_test_setting_from_ckpt', None)

    # 若无法从 checkpoint 推导，就回退到 format(...) 方式
    if not setting:
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            getattr(args, 'd_model', 128), getattr(args, 't_d_model', 128),
            getattr(args, 'n_heads', 8),
            getattr(args, 'e_layers', 2), getattr(args, 'd_layers', 1),
            getattr(args, 'd_ff', 256), getattr(args, 'factor', 1),
            getattr(args, 'embed', 'timeF'),
            ws_tag, hn_tag, e2e_tag, ev_tag, dec_tag, sp_tag, sp_hn_tag, amp_tag
        )

    exp = Exp(args)
    if getattr(args, 'do_predict', False):
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_ret = exp.test(setting, test=1)
        if nni is not None and isinstance(test_ret, dict):
            try:
                payload = {}
                if 'test/mse' in test_ret:
                    payload['default'] = float(test_ret['test/mse'])
                else:
                    payload['default'] = 0.0
                payload.update(test_ret)
                print(f"[NNI] report_final_result (test only): {payload}")
                nni.report_final_result(payload)
            except Exception as e:
                print(f"[Warn] NNI report_final_result failed: {e}", file=sys.stderr)

    torch.cuda.empty_cache()
