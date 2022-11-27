import argparse
import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm


args = {
    "input": "../../data/nq-train/biencoder-nq-train.json",
    "output": "../../data/nq-train/bm25.mbert_8.json",
    "tokenizer": "bert-base-multilingual-cased",
    "minimum_negatives": 8,
}
args = argparse.Namespace(**args)

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer, use_fast=True
)

data = json.load(open(args.input))

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(args.output, "w") as f:
    print("Writing to", args.output)
    for idx, item in enumerate(tqdm(data)):
        if (
            len(item["hard_negative_ctxs"]) < args.minimum_negatives
            or len(item["positive_ctxs"]) < 1
        ):
            continue

        positives = [
            pos["title"] + tokenizer.sep_token + pos["text"]
            for pos in item["positive_ctxs"]
        ]
        negatives = [
            neg["title"] + tokenizer.sep_token + neg["text"]
            for neg in item["hard_negative_ctxs"]
        ]

        query = tokenizer.encode(
            item["question"], add_special_tokens=False, max_length=256, truncation=True
        )
        positives = tokenizer(
            positives, add_special_tokens=False, max_length=256, truncation=True
        )["input_ids"]
        negatives = tokenizer(
            negatives, add_special_tokens=False, max_length=256, truncation=True
        )["input_ids"]

        group = {"query": query, "positives": positives, "negatives": negatives}
        f.write(json.dumps(group) + "\n")
