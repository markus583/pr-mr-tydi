# Simple script for tuning BM25 parameters (k1 and b) for Mr. TyDi

import argparse
import os
import re
import subprocess
import numpy as np

parser = argparse.ArgumentParser(description='Tunes BM25 parameters for Mr. TyDi.')
parser.add_argument('--set_name', type=str, default='dev', help='Set name (dev, training, or test)')
parser.add_argument('--language', required=True, help='language of the queries')
parser.add_argument('--base_directory', help='Base directory of the index')


args = parser.parse_args()

set_name = args.set_name
language = args.language
if language == "bengali":
    language_abbr = "bn"
elif language == "indonesian":
    language_abbr = "id"
else:
    language_abbr = language[:2]
    
index = f"indices/{language}"
qrels = f"../../data/mrtydi-v1.1-{language}/qrels.{set_name}.txt"
topics = f"../../data/mrtydi-v1.1-{language}/topic.{set_name}.tsv"
base_directory = args.base_directory

if not os.path.exists(base_directory):
    os.makedirs(base_directory)

print('# Settings')
print(f'index: {index}')
print(f'qrels: {qrels}')
print(f'topics: {topics}')


for k1 in np.arange(0.1, 1.7, 0.1):
    for b in np.arange(0.1, 1.1, 0.1):
        k1 = str(k1)[:3]
        b = str(b)[:3]
        print(f'Trying... k1 = {k1}, b = {b}')
        filename = f'{base_directory}/run.bm25.{language}.k1_{k1}.b_{b}.{set_name}.txt'
        print(filename)
        if os.path.isfile(f'{filename}'):
            print('Run already exists , skipping!')
        else:
            subprocess.call(f'python -m pyserini.search.lucene --index indexes/{language} --topics {topics}'
                            f' --output {filename} --bm25 --k1 {k1} --b {b} --language {language_abbr}', shell=True)

print('\n\nStarting evaluation...')

# maximize recall
max_score = 0
max_file = None

for filename in sorted(os.listdir(base_directory)):
    print(filename)
    # TREC output run file, perhaps left over from a previous tuning run: skip.
    if filename.endswith('trec') or filename.startswith(".") or filename.startswith("best_"):
        print("Skip.")
        continue
    if not filename.endswith(f"{set_name}.txt"):
        print("Skip - wrong ending.")
        continue

    results = subprocess.check_output([
        'python',
        '-m',
        'pyserini.eval.trec_eval',
        '-c',
        '-mrecip_rank',
        qrels,
        f'{base_directory}/{filename}'
    ])
    mrr = results.decode('utf-8').split("0.")[-1].strip()
    mrr = float(f"0.{mrr}")
    print("mrr:", mrr)

    if mrr > max_score:
        max_score = mrr
        max_file = filename

print(f'\n\nBest parameters: {max_file}: mrr = {max_score}')
# save to file
with open(f'{base_directory}/best_parameters-mrr.txt', 'w') as f:
    f.write(f'{max_file}: mrr@100 = {max_score}')
    # get k1 and b
    k1 = re.search("k1_(.*?).b", max_file)[1]
    b = re.search("b_(.*?).txt", max_file)[1]
    if "." in b:
        b = b.split(".")[0] + "." + b.split(".")[1]
    print(f'k1: {k1}, b: {b}')
    # check results on test set
    qrels = f"../../data/mrtydi-v1.1-{language}/qrels.test.txt"
    topics = f"../../data/mrtydi-v1.1-{language}/topic.test.tsv"
    filename = f'{base_directory}/run.bm25-{language}.k1_{k1}.b_{b}.test.txt'
    print(filename)
    if os.path.isfile(f'{base_directory}/{filename}'):
        print('Run already exists, skipping!')
    else:
        subprocess.call(f'python -m pyserini.search.lucene --index indexes/{language} --topics {topics}'
                        f' --output {filename} --bm25 --k1 {k1} --b {b} --language {language_abbr}', shell=True)

    # Evaluate with official scoring script
    results = subprocess.check_output([
        'python',
        '-m',
        'pyserini.eval.trec_eval',
        '-c',
        '-mrecall.100',
        '-mrecip_rank',
        qrels,
        f'{filename}'
    ])
    # write only last 3 lines of results to file
    for line in results.decode('utf-8').splitlines()[-3:]:
        f.write(f'{line}')

