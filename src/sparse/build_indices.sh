#!/bin/bash

lang=bengali     # one of {'arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai'}
lang_abbr=bn    # one of {'ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th'}
collection_dir=../../data/mrtydi-v1.1-${lang}/collection
index_dir=indices/${lang}


python -m pyserini.index.lucene  \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input ${collection_dir} \
    -index ${index_dir} \
    -storePositions -storeDocvectors -storeRaw \
    -optimize \
    -language ${lang_abbr}