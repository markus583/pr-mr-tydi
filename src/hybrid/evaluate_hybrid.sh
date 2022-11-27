#!/bin/bash

export PYTHONPATH=/home/markus/anaconda3/bin/python
# activate conda base environment
source /home/markus/anaconda3/bin/activate

lang=arabic     # one of {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
# shellcheck disable=SC2209
set_name=test   # one of {'training', 'dev', 'test'}


bm25_runfile=../../runs/sparse/run.bm25.mrtydi-v1.1-${lang}.${set_name}.txt
dense_runfile=../../runs/dense/run.mdpr.mrtydi-v1.1-${lang}.${set_name}.txt
output_runfile=../../runs/hybrid/run.hybrid-default.mrtydi-v1.1-${lang}.${set_name}.txt
qrels=../../data/mrtydi-v1.1-${lang}/qrels.${set_name}.txt

lang=arabic     # one of {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
python hybrid.py    --lang ${lang} \
                            --sparse ${bm25_runfile} \
                            --dense ${dense_runfile} \
                            --output ${output_runfile} \
                            --weight-on-dense \
                            --normalization

python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${output_runfile}