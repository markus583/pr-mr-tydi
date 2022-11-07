#!/usr/bin/env python
import sys

script = "tevatron.driver.train \
  --output_dir model_mdpr_3 \
  --model_name_or_path bert-base-multilingual-cased \
  --save_steps 1000 \
  --dataset_name Tevatron/wikipedia-nq \
  --per_device_train_batch_size 128 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 30 \
  --logging_steps 100 \
  --grad_cache \
  --overwrite_output_dir \
  --fp16 \
  --do_train \
  --untie_encoder"

import subprocess
subprocess.run([sys.executable] + ["-m"] + script.split())
