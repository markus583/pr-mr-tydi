#!/bin/bash

languages=('arabic' 'bengali' 'english' 'finnish' 'indonesian' 'japanese' 'korean' 'russian' 'swahili' 'telugu' 'thai')
languages2=('ar' 'bn' 'en' 'fi' 'id' 'ja' 'ko' 'ru' 'sw' 'te' 'th')
set_name=test   # one of {'training', 'dev', 'test'}

touch runs/results_all.txt
truncate -s 0 runs/results_all.txt

# all languages and their abbreviations
for lang in "${!languages[@]}"; do
  runfile=runs/run.bm25.mrtydi-v1.1-${languages[$lang]}.${set_name}.txt
  qrels=../../data/mrtydi-v1.1-${languages[$lang]}/qrels.${set_name}.txt
  echo "############# ${languages[$lang]} #############" >> runs/results_all.txt

  python -m pyserini.search.lucene \
    --index indexes/${languages[$lang]} \
    --topics ../../data/mrtydi-v1.1-${languages[$lang]}/topic.${set_name}.tsv \
    --output ${runfile} \
    --bm25 \
    --k1 0.9 \
    --b 0.4 \
    --language ${languages2[$lang]} \

  python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${runfile} | tail -n 3 >> runs/results_all.txt
  done