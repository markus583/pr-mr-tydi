#!/bin/bash
# download and extract Mr. TiDy for all languages

for lang in arabic bengali english finnish indonesian japanese korean russian swahili telugu thai; do
    wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-$lang.tar.gz
    tar -xf mrtydi-v1.1-$lang.tar.gz
    gzip -d mrtydi-v1.1-$lang/collection/docs.jsonl.gz
    echo "Downloaded and extracted $lang"
done