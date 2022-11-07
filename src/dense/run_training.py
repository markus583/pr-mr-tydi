#!/usr/bin/env python
import sys

script = "pr-dense.training.train \
  --output_dir /home/markus_frohmann/models/model_mdpr_TEST \
  --model_name_or_path bert-base-multilingual-cased \
  --save_steps 2 \
  --train_dir  /home/markus_frohmann/nq-train/bm25.mbert.json \
  --per_device_train_batch_size 128 \
  --positive_passage_no_shuffle \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 30 \
  --logging_steps 100 \
  --overwrite_output_dir \
  --fp16 \
  --do_train \
  --gc_q_chunk_size 16 \
  --gc_p_chunk_size 64 \
  --untie_encoder"

import subprocess
subprocess.run([sys.executable] + ["-m"] + script.split())
