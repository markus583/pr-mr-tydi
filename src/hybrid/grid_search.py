# Simple script for tuning BM25 parameters (k1 and b) for Mr. TyDi

# python grid_search.py --language arabic --dense_runfile ../../runs/run.mdpr.mrtydi-v1.1-arabic.test.txt --bm25_runfile ../../runs/run.bm25.mrtydi-v1.1-arabic.test.txt --output_directory ../../runs/hybrid/arabic

import argparse
import os
import subprocess
import numpy as np

parser = argparse.ArgumentParser(
    description="Tunes alpha weight for dense and sparse retrieval for Mr. TyDi."
)
parser.add_argument(
    "--set_name", type=str, default="dev", help="Set name (dev, training, or test)"
)
parser.add_argument("--language", required=True, help="language of the queries")
parser.add_argument("--dense_runfile", help="dense runfile", required=True)
parser.add_argument("--bm25_runfile", help="sparse runfile", required=True)
parser.add_argument("--output_directory", help="output directory", required=True)

args = parser.parse_args()

set_name = args.set_name
language = args.language
dense_runfile = args.dense_runfile
bm25_runfile = args.bm25_runfile
output_directory = args.output_directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

qrels = f"../../data/mrtydi-v1.1-{language}/qrels.{set_name}.txt"


print("# Settings")
print(f"dense_runfile: {dense_runfile}")
print(f"bm25_runfile: {bm25_runfile}")
print(f"qrels: {qrels}")


for alpha in np.arange(0.700, 1.01, 0.01):
    alpha = str(alpha)[:5]
    if alpha[-1] == 0:
        alpha = alpha[:-1]
    # if exists, skip
    filename = f"{output_directory}/run.alpha_{alpha}.{set_name}.txt"
    if os.path.isfile(f"{filename}"):
        print("Run already exists, skipping!")
        continue
    print(f"Trying... alpha = {alpha}")
    filename = f"{output_directory}/run.alpha_{alpha}.{set_name}.txt"
    print(filename)
    subprocess.call(
        f"python hybrid.py --sparse {bm25_runfile} --dense {dense_runfile} --alpha {alpha}"
        f" --output {filename} --weight-on-dense --normalization",
        shell=True,
    )


print("\n\nStarting evaluation...")

# maximize recall
max_score = 0
max_file = None

for filename in sorted(os.listdir(output_directory)):
    print(filename)
    # TREC output run file, perhaps left over from a previous tuning run: skip.
    if (
        filename.endswith("trec")
        or filename.startswith(".")
        or filename.startswith("best_")
    ):
        print("Skip.")
        continue
    if not filename.endswith(f"{set_name}.txt"):
        print("Skip - wrong ending.")
        continue

    results = subprocess.check_output(
        [
            "python",
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            "-mrecip_rank",
            qrels,
            f"{output_directory}/{filename}",
        ]
    )

    mrr = results.decode("utf-8").split("0.")[-1].strip()
    mrr = float(f"0.{mrr}")
    print("mrr:", mrr)

    if mrr > max_score:
        max_score = mrr
        max_file = filename


print(f"\n\nBest parameters: {max_file}: mrr = {max_score}")

# run with best alpha on test set


# save to file
with open(f"{output_directory}/best_parameters-mrr.txt", "w") as f:
    f.write(f"{max_file}: mrr@100 = {max_score}\n")
    # get k1 and b
    alpha = max_file.split("_")[1].split(f".{set_name}")[0]
    print(alpha)

    test_dense_runfile = f"../../runs/run.mdpr.mrtydi-v1.1-{language}.test.txt"
    test_bm25_runfile = f"../../runs/run.bm25.mrtydi-v1.1-{language}.test.txt"
    test_output_directory = f"../../runs/hybrid/{language}/run.alpha_{alpha}.test.txt"
    subprocess.call(
        f"python hybrid.py --lang {language} --sparse {test_bm25_runfile} --dense {test_dense_runfile} --alpha {alpha}"
        f" --output {test_output_directory} --weight-on-dense --normalization",
        shell=True,
    )

    # check results on test set
    qrels = f"../../data/mrtydi-v1.1-{language}/qrels.test.txt"

    print("\n\nStarting evaluation on test set...")

    # Evaluate with official scoring script
    results = subprocess.check_output(
        [
            "python",
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            "-mrecall.100",
            "-mrecip_rank",
            qrels,
            f"{test_output_directory}",
        ]
    )
    print(results.decode("utf-8"))
    f.write(f"alpha: {alpha}\n")
    f.write(f"{language} - TEST SET ")
    # write only last 3 lines of results to file
    for line in results.decode("utf-8").splitlines()[-3:]:
        f.write(f"{line}")
        f.write("\n")
