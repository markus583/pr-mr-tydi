#!/bin/bash
set_name=test   # one of {'train', 'dev', 'test'}
# list: {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
lang=english


# loop over languages
export PYTHONPATH=/opt/conda/bin/python3
# activate conda base environment
source /opt/conda/bin/activate

runs=../../runs/dense_results.txt
touch $runs

encoder=../../models/model_untied-30-plen-256
index_dir=indices/$lang
corpus=../../data/mrtydi-v1.1-$lang/collection/docs.jsonl

nshards=8
for shard in $(seq 0 $((nshards-1))); do
  python -m pyserini.encode input   --corpus $corpus \
                                  --fields title text \
                                  --delimiter "\n\n" \
                                  --shard-id $shard \
                                  --shard-num $nshards \
                          output  --embeddings  $index_dir \
                                  --to-faiss \
                          encoder --encoder $encoder/passage_model/ \
                                  --fields title text \
                                  --batch 78 \
                                  --max-length 256 \
                                  --fp16

  runfile=../../runs/dense/run.mdpr.mrtydi-v1.1-$lang.${set_name}.txt-$shard
  python -m pyserini.search.faiss     --topics ../../data/mrtydi-v1.1-$lang/topic.$set_name.tsv     --index $index_dir --encoder=$encoder/query_model     --batch-size 128     --threads 12 --output $runfile
done
python merge_shards.py --files ../../runs/dense/run.mdpr.mrtydi-v1.1-$lang.${set_name}.txt-* --output ../../runs/dense/run.mdpr.mrtydi-v1.1-$lang.${set_name}.txt


qrels=../../data/mrtydi-v1.1-$lang/qrels.$set_name.txt
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${runfile} | tail -n 2 >> $runs