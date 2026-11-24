#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
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
  --num_workers 0 --itr 1 --use_gpu true "$@"
