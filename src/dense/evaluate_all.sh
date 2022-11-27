#!/bin/bash


# list: {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
# loop over languages
export PYTHONPATH=/opt/conda/bin/python3
# activate conda base environment
source /opt/conda/bin/activate

runs=../../runs/dense_results.txt
touch $runs
for lang in arabic bengali english finnish indonesian japanese korean russian swahili telugu thai; do
  tarfn=mrtydi-v1.1-$lang.tar.gz
  corpus=../../data/mrtydi-v1.1-$lang/collection/docs.jsonl
  index_dir=indices/$lang
  encoder=../../models/model_untied-30-plen-256


  python -m pyserini.encode input   --corpus $corpus \
                                    --fields title text \
                                    --delimiter "\n\n" \
                            output  --embeddings  $index_dir \
                                    --to-faiss \
                            encoder --encoder $encoder/passage_model/ \
                                    --fields title text \
                                    --batch 78 \
						    --max-length 256 \
                                    --fp16

  set_name=test   # one of {'train', 'dev', 'test'}
  runfile=../../runs/dense/run.mdpr.mrtydi-v1.1-$lang.${set_name}.txt
  python -m pyserini.search.faiss     --topics ../../mrtydi-v1.1-$lang/topic.$set_name.tsv     --index $index_dir     --encoder=$encoder/query_model     --batch-size 128     --threads 12 --output $runfile


  qrels=../../data/mrtydi-v1.1-$lang/qrels.$set_name.txt
  # write language to file
    echo $lang >> $runs
  # write ---------------- to file
    echo "----------------" >> $runs
  # fetch last 2 lines from next line's output and write to file
    python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 ${qrels} ${runfile} | tail -n 2 >> $runs
done
