# CMBF — Original Basic Version

Original basic implementation of the **Complementary Multi-Branch Framework (CMBF)** for long-term multivariate time-series forecasting. The model combines a linear branch, a multi-scale node/hypergraph branch, and a semantic event branch.

本仓库整理了 CMBF 的基础原始版本：保留线性、节点超图、语义事件三个分支，以及原始训练和数据处理代码。

## Version and contents

The documented model is `models.EventMSHyperV1.Model`, with the original source header **v0.6.7 (2025-12-31)**. All original model, data-loader and training files are unchanged; their SHA-256 hashes are recorded in `SOURCE_PROVENANCE.json`.

The original `EventMSHyper` module and its event tower are included because the unchanged training module imports them. The commands below explicitly select `EventMSHyperV1`.

Later sample-prototype routing, ridge/stable-basis extensions, and experimental fast paths are excluded. Datasets, checkpoints and experiment outputs are not included.

## Installation

Use Python 3.10 and install a [PyTorch build appropriate for your CPU or CUDA](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The package was verified with Python 3.10, PyTorch 2.7.0+cu128 and PyTorch Geometric 2.6.1. The smoke test runs on CPU and checks all three branches, auxiliary losses, gradients, and evaluation output. NNI is optional for hyperparameter search; normal training does not require it.

## Data

Place the Weather benchmark CSV at `data/weather/weather.csv`. The expected CSV has a `date` column followed by 21 numeric variables. For custom CSV data, supply the number of variables through `--enc_in`, `--dec_in` and `--c_out`. This original data loader uses chronological 70%/10%/20% train/validation/test splits for `custom` data, with normalization fitted on the training split.

The [Ada-MSHyper repository](https://github.com/shangzongjiang/Ada-MSHyper) describes how to access the benchmark datasets.

## Train and evaluate

From the repository root:

```bash
bash scripts/train_weather.sh
```

Equivalent command:

```bash
python -u run_STHyper1.py \
  --is_training 1 --model_id CMBF_weather_96_96 --model EventMSHyperV1 \
  --data custom --root_path ./data/weather/ --data_path weather.csv \
  --features M --target OT --freq t \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --win_s1 2 --win_s2 2 --hyper_s1 20 --hyper_s2 12 --hyper_s3 6 \
  --t_d_model 32 --t_kprime 3 --t_beta_thr 0.0 \
  --event_layers 2 --num_event_protos 16 --proto_topk 4 --event_topk 8 \
  --dec_heads 4 --dec_layers 1 --dec_ffn_dim 64 --cycle 144 \
  --batch_size 32 --train_epochs 25 --patience 5 --learning_rate 0.001 \
  --num_workers 0 --itr 1 --use_gpu true
```

These are a runnable example configuration, not a claim to reproduce the manuscript's tuned scores. Change `--pred_len` for other prediction horizons. The original runner derives multi-scale settings from `--win_s1`, `--win_s2`, and `--hyper_s1/2/3`; use these options when changing scale sizes. Its `--event_d_model` option is legacy; the documented original model derives event dimensionality internally.

Training automatically evaluates the selected checkpoint and writes checkpoints, metrics, prediction curves and optional event exports to `checkpoints/`, `results/` and `runs/`. These generated directories are ignored by Git.

For CPU-only execution, hide CUDA devices and set `--use_gpu false`:

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/train_weather.sh --use_gpu false
```

Extra arguments to `scripts/train_weather.sh` are forwarded to the original runner. For example:

```bash
bash scripts/train_weather.sh --pred_len 192 --batch_size 16
python run_STHyper1.py --help
```

## Source layout

- `models/EventMSHyperV1.py`: original three-branch model.
- `models/structure`, `models/ops`: adaptive hypergraph construction and convolution.
- `models/e2e`: prototype memory and semantic event tower.
- `data_provider`: original dataset preparation and loaders.
- `exp`: original training, validation and evaluation implementation.
- `run_STHyper1.py`: original command-line entry point.
- `scripts/smoke_test.py`: small synthetic forward/backward verification.
