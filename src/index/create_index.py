from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('mrtydi-v1.1-arabic')
hits = searcher.search('what is a lobster roll?')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')