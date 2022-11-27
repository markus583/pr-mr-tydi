#!/bin/bash
# create an index for each language in Mr. TyDi

languages=('arabic' 'bengali' 'english' 'finnish' 'indonesian' 'japanese' 'korean' 'russian' 'swahili' 'telugu' 'thai')
languages2=('ar' 'bn' 'en' 'fi' 'id' 'ja' 'ko' 'ru' 'sw' 'te' 'th')

# all languages and their abbreviations
for lang in "${!languages[@]}"; do
  collection_dir=../../data/mrtydi-v1.1-${languages[$lang]}/collection
  index_dir=indices/${languages2[$lang]}
  mkdir -p "${index_dir}"
  python -m pyserini.index.lucene  \
    -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 8 \
    -input "${collection_dir}" \
    -index "${index_dir}" \
    -storePositions -storeDocvectors -storeRaw \
    -optimize \
    -language ${languages2[$lang]}
    echo "Built index for ${languages[$lang]}"
  done