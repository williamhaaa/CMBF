# -*- coding: utf-8 -*-
"""
utils/tools.py
Version: v1.2.0 (2025-11-13)

[Changelog]
- v1.2.0
  1) adjust_learning_rate 彻底重写：兼容 none/type1/type2/step/cosine/plateau/数值倍率，
     且以“相对初始 lr 比例”缩放每个 param_group 的 lr，避免覆盖结构优化器等自定义初值。
  2) test_params_flop 支持宽松签名：如果只传 x_shape，不会报错（给出提示后返回）。
  3) visual 确保保存目录存在；EarlyStopping/StandardScaler 保持原逻辑。
"""

import os
import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


# ====================== LR Scheduler ======================

def adjust_learning_rate(optimizer, epoch, args):
    """
    兼容策略：
      - 'none'/'off'：不调整
      - 'type1'：按 epoch=2,4,6,8... 等比例衰减（与旧代码近似）
      - 'type2'：按固定里程碑字典衰减（保持兼容）
      - 'step' ：等间隔阶梯衰减（lr_step_size, lr_gamma）
      - 'cosine'：余弦退火（cosine_tmax, eta_min）
      - 'plateau'：近似 plateau（plateau_patience, plateau_gamma），无度量版本
      - 直接数值：若 lradj 可转为 float，则视作倍率（如 0.1 表示降到 0.1×）
    说明：
      以每个 param_group 的“初始 lr”为基准，按比例缩放，避免覆盖结构优化器等的自定初值。
    """
    # 允许从 --lradj 或 --lr_sched 获取策略名
    sched = str(getattr(args, 'lradj', getattr(args, 'lr_sched', 'type1'))).lower()
    base_lr = float(getattr(args, 'learning_rate', 1e-3))

    # 记录并按比例应用到每个 param_group
    def _apply(new_lr):
        ratio = float(new_lr) / max(1e-12, base_lr)
        for pg in optimizer.param_groups:
            if '_lr0' not in pg:
                # 记住每个 group 自己的初始 lr
                pg['_lr0'] = float(pg.get('lr', base_lr))
            pg['lr'] = pg['_lr0'] * ratio
        print(f'[LR] epoch={epoch} sched={sched} -> lr={new_lr:.6g} (×{ratio:.4g})')

    # none/off
    if sched in ('none', 'off', 'no'):
        return

    # 旧式策略：type1
    if sched == 'type1':
        # 你原来的写法相当于每个 epoch 都有一个值；这里保持“每过 1 个 epoch 衰减 0.5”
        k = max(0, epoch - 1)  # 从第1个 epoch 开始衰减更平滑
        new_lr = base_lr * (0.5 ** k)
        _apply(new_lr)
        return

    # 旧式策略：type2（按固定里程碑）
    if sched == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust:
            _apply(lr_adjust[epoch])
        return

    # 新式：step
    if sched == 'step':
        step_size = int(getattr(args, 'lr_step_size', 10))
        gamma = float(getattr(args, 'lr_gamma', 0.5))
        k = (max(1, epoch) - 1) // max(1, step_size)
        new_lr = base_lr * (gamma ** k)
        _apply(new_lr)
        return

    # 新式：cosine
    if sched == 'cosine':
        T = int(getattr(args, 'cosine_tmax', 10))
        if T <= 0:
            T = int(getattr(args, 'train_epochs', max(1, epoch)))
        eta_min = float(getattr(args, 'eta_min', 1e-6))
        tt = min(epoch, T)
        new_lr = eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * tt / T)) / 2.0
        _apply(new_lr)
        return

    # 新式：plateau（无度量版，用耐心窗口近似）
    if sched == 'plateau':
        patience = int(getattr(args, 'plateau_patience', 5))
        gamma = float(getattr(args, 'plateau_gamma', 0.5))
        k = (max(1, epoch) - 1) // max(1, patience)
        new_lr = base_lr * (gamma ** k)
        _apply(new_lr)
        return

    # 直接把 lradj 当作倍率
    try:
        factor = float(sched)
        _apply(base_lr * factor)
    except Exception:
        # 未知策略：静默跳过，避免报错
        print(f'[LR] Unknown schedule "{sched}", skip.')
        return


# ====================== Early Stopping ======================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = float(delta)

    def __call__(self, val_loss, model, path):
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} -> {float(val_loss):.6f}). Saving model...')
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = float(val_loss)


# ====================== Misc ======================

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__  = dict.get
    __setattr__  = dict.__setitem__
    __delattr__  = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean) if np.isscalar(mean) else mean
        self.std  = float(std)  if np.isscalar(std)  else std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-12)

    def inverse_transform(self, data):
        return data * (self.std + 1e-12) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    曲线可视化：保存到 name 路径。
    """
    out_dir = os.path.dirname(name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_params_flop(*args):
    """
    用法：
      1) test_params_flop(model, x_shape)
      2) test_params_flop(x_shape)  # 兼容旧误用：只传 x_shape 时给出提示并返回，不报错
    说明：
      x_shape 是 ptflops 需要的输入尺寸（例如 1D 模型可用 (T, C) 或 (C, H, W) 等）。
    """
    if len(args) == 1:
        # 只有 x_shape：不做统计，避免报错
        print('[Warn] test_params_flop: 需要传入 model 实例 (model, x_shape)。已跳过 FLOPs 统计。')
        return
    if len(args) != 2:
        print('[Warn] test_params_flop: 参数个数不对，期望 (model, x_shape)。已跳过。')
        return

    model, x_shape = args[0], args[1]

    # 统计可训练参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1e6))

    # ptflops 统计（可选）
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        print('[Warn] 未安装 ptflops，跳过 FLOPs 统计。可 pip install ptflops')
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, x_shape,
            as_strings=True,
            print_per_layer_stat=True
        )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
