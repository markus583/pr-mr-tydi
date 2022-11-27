#!/bin/bash

export PYTHONPATH=/home/markus/anaconda3/bin/python
# activate conda base environment
source /home/markus/anaconda3/bin/activate

# shellcheck disable=SC2209
set_name=test   # one of {'training', 'dev', 'test'}

out_file=../../runs/hybrid_results.txt
touch $out_file
for lang in arabic bengali english finnish indonesian japanese korean russian swahili telugu thai; do
  bm25_runfile=../../runs/sparse/run.bm25.mrtydi-v1.1-${lang}.${set_name}.txt
  dense_runfile=../../runs/dense/run.mdpr.mrtydi-v1.1-${lang}.${set_name}.txt
  output_runfile=../../runs/hybrid/run.hybrid-default.mrtydi-v1.1-${lang}.${set_name}.txt
  qrels=../../data/mrtydi-v1.1-${lang}/qrels.${set_name}.txt

  python hybrid.py    --lang ${lang} \
                              --sparse ${bm25_runfile} \
                              --dense ${dense_runfile} \
                              --output ${output_runfile} \
                              --weight-on-dense \
                              --normalization

  python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${output_runfile}
  # write language to file
    echo $lang >> $out_file
  # write ---------------- to file
    echo "----------------" >> $out_file
  # fetch last 2 lines from next line's output and write to file
    python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${output_runfile} | tail -n 2 >> $out_file
done