#!/bin/bash

lang=finnish     # one of {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
lang_abbr=fi    # one of {'ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th'}
set_name=test   # one of {'training', 'dev', 'test'}
runfile=runs/run.bm25.mrtydi-v1.1-${lang}.${set_name}.txt
qrels=../../data/mrtydi-v1.1-${lang}/qrels.${set_name}.txt

python -m pyserini.search.lucene \
  --index indices/${lang} \
  --topics ../../data/mrtydi-v1.1-${lang}/topic.${set_name}.tsv \
  --output ${runfile} \
  --bm25 \
  --language ${lang_abbr}
  
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${runfile} 