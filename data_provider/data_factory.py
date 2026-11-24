# -*- coding: utf-8 -*-
"""
data_provider/data_provider.py
Version: v0.2.1  (2026-08-20)

【版本号与修改记录（changelog）】
- v0.2.1 [Validation Partial Batch]
  1) 验证集不打乱并保留最后一个不足 batch_size 的批次。

- v0.2.0 [Simplify + Cycle]
  1) 删除 CUDAPrefetchLoader 和 RAMCacheDataset 等加速/缓存逻辑，
     保留一个最简洁的 data_provider 接口；
  2) 在构造各类 Dataset 时透传 cycle 参数，用于内部生成 cycle_index；
  3) 保持与原 baseline 相同的 flag / batch_size / shuffle / drop_last 行为。

- v0.1.0
  1) 初始版本：兼容 ETTh/ETTm/Solar/PEMS/custom/Pred 等数据集。
"""

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Solar,
    Dataset_PEMS,
    Dataset_Pred,
)

# ========= 数据集类型映射 =========
data_dict = {
    'ETTh1':  Dataset_ETT_hour,
    'ETTh2':  Dataset_ETT_hour,
    'ETTm1':  Dataset_ETT_minute,
    'ETTm2':  Dataset_ETT_minute,
    'Solar':  Dataset_Solar,
    'PEMS':   Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    """
    根据 args 和 flag 返回 (Dataset 实例, DataLoader)。

    约定：
      - Dataset.__getitem__ 返回 5 个量：
          seq_x, seq_y, x_mark, y_mark, cycle_x
        其中 cycle_x: [L] 或 [L,1]，由 Dataset 内部根据 cycle 生成。

      - args 中与本函数相关的字段：
          args.data         : 数据集名称（ETTh1/ETTm1/PEMS/Solar/custom）
          args.root_path    : 数据根路径
          args.data_path    : 具体文件名
          args.seq_len      : encoder 序列长度
          args.label_len    : decoder label 部分长度
          args.pred_len     : 预测长度
          args.features     : 'M' / 'S' 等
          args.target       : 目标列名
          args.embed        : 'timeF' / 'fixed' 等（决定 timeenc）
          args.freq         : 频率字符串（'h'/'t15'...）
          args.batch_size   : batch 大小
          args.num_workers  : DataLoader worker 数
          args.cycle        : 周期长度（整数，>0 时启用 cycle_index）
    """
    # 选择 Dataset 类
    Data = data_dict[args.data]

    # 是否使用 time features encoder
    timeenc = 0 if args.embed != 'timeF' else 1

    # ====== 根据 flag 设置 DataLoader 行为 ======
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size      # 测试阶段可用较大 batch
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred               # 预测专用 Dataset
    else:
        # 'train' / 'val'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # ====== 构造 Dataset 实例 ======
    # 关键：透传 cycle=args.cycle，供 Dataset 内部生成 cycle_index
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=getattr(args, 'cycle', 0),   # 若 args 没有 cycle，默认 0
    )

    print(flag, len(data_set))

    # ====== 构造 DataLoader ======
    num_workers = getattr(args, 'num_workers', 0)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader
