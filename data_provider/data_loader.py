# -*- coding: utf-8 -*-
"""
data_loader.py
Version: v1.1.0  (2025-11-24)

【版本号与修改记录（changelog）】
- v1.1.0  [CycleIndex + Backward-Compat]
  1) 为 ETT_hour / ETT_minute / Custom / Solar / PEMS 增加可选 cycle_index：
     • 在 __read_data__ 中按“全局时间索引”构造 self.cycle_index；
     • 当 cycle=None 时，self.cycle_index = None。
  2) 在上述五个数据集的 __getitem__ 中，根据是否启用 cycle_index 自适应返回 4/5 个元素：
     • 未启用：返回 (seq_x, seq_y, seq_x_mark, seq_y_mark)，与原版完全一致；
     • 启用：额外返回 cycle_index（LongTensor），用于周期条件建模。
  3) Dataset_Pred 行为保持不变，仅返回 4 个张量，兼容 RAMCacheDataset / ExpMain 旧逻辑。
"""

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


# =========================== ETT-hour ===========================

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, cycle=None):
        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # split type
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 官方 ETT 划分
        border1s = [0,
                    12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time features
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day'] = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # cycle_index（可选）
        if self.cycle is None:
            self.cycle_index = None
        else:
            full_idx = np.arange(len(data)) % int(self.cycle)
            self.cycle_index = full_idx[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 兼容旧逻辑：无 cycle 时只返回 4 个
        if self.cycle_index is None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            cycle_index = torch.tensor(self.cycle_index[s_end], dtype=torch.long)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =========================== ETT-minute ===========================

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 train_only=False, cycle=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0,
                    12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day'] = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda r: r.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda r: r.minute)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.cycle is None:
            self.cycle_index = None
        else:
            full_idx = np.arange(len(data)) % int(self.cycle)
            self.cycle_index = full_idx[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.cycle_index is None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            cycle_index = torch.tensor(self.cycle_index[s_end], dtype=torch.long)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =========================== Custom CSV ===========================

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, cycle=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1.0))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0,
                    num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train,
                    num_train + num_vali,
                    len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features: {self.features}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day'] = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.cycle is None:
            self.cycle_index = None
        else:
            full_idx = np.arange(len(data)) % int(self.cycle)
            self.cycle_index = full_idx[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.cycle_index is None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            cycle_index = torch.tensor(self.cycle_index[s_end], dtype=torch.long)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =========================== Prediction Dataset ===========================

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False,
                 timeenc=0, freq='15min', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ['M', 'MS']:
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported features: {self.features}")

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2].copy()
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda r: r.month)
            df_stamp['day'] = df_stamp.date.apply(lambda r: r.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda r: r.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda r: r.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda r: r.minute)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp['date'].values),
                freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported timeenc: {self.timeenc}")

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 与旧版一致，仅返回 4 个
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =========================== Solar ===========================

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, cycle=None, train_only=False):
        # Solar 数据集原本就是纯数值，没有 date 列
        if size is None:
            raise ValueError("Dataset_Solar requires explicit size=[seq_len, label_len, pred_len]")
        self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target  # 这里一般用不上
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns
        self.cycle = cycle
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0,
                    num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train,
                    num_train + num_vali,
                    len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.cycle is None:
            self.cycle_index = None
        else:
            full_idx = np.arange(len(data)) % int(self.cycle)
            self.cycle_index = full_idx[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # Solar 没有显式时间戳，这里用 0 占位
        seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
        seq_y_mark = torch.zeros((seq_y.shape[0], 1), dtype=torch.float32)

        if self.cycle_index is None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            cycle_index = torch.tensor(self.cycle_index[s_end], dtype=torch.long)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =========================== PEMS ===========================

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, cycle=None):
        if size is None:
            raise ValueError("Dataset_PEMS requires explicit size=[seq_len, label_len, pred_len]")
        self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target  # traffic 数据一般为单变量通道
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        # [T, N, 1] -> [T, N]
        data = data['data'][:, :, 0]

        T = len(data)
        num_train = int(T * 0.6)
        num_valid = int(T * 0.2)
        num_test = T - num_train - num_valid

        border1s = [0,
                    num_train,
                    num_train + num_valid]
        border2s = [num_train,
                    num_train + num_valid,
                    T]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df[border1:border2]
        self.data_y = df[border1:border2]

        if self.cycle is None:
            self.cycle_index = None
        else:
            full_idx = np.arange(T) % int(self.cycle)
            self.cycle_index = full_idx[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
        seq_y_mark = torch.zeros((seq_y.shape[0], 1), dtype=torch.float32)

        if self.cycle_index is None:
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            cycle_index = torch.tensor(self.cycle_index[s_end], dtype=torch.long)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
