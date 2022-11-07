from dataclasses import dataclass

from datasets import load_dataset


import random
from typing import List, Tuple, Dict

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from ..arguments import DataArguments

import logging

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """
    Used for the data collator.
    """

    def __init__(
        self,
        data_args: DataArguments,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizer,
        trainer=None,
    ):
        """

        :param data_args: Data arguments
        :param dataset: Dataset to use, should be a dataset from the datasets library
        :param tokenizer: Tokenizer to use, should be a tokenizer from the transformers library
        :param trainer: Trainer to use, should be a trainer from the transformers library
        """
        self.train_data = dataset
        self.tokenizer = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.length = len(self.train_data)

    def return_single_sample(self, text_encoding: List[int], is_query=False):
        return self.tokenizer.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len
            if is_query
            else self.data_args.p_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        Returns a tuple of the query and a list of positive and negative passages.
        :param item: Index of the sample to return
        :return: Tuple of the query and a list of positive and negative passages
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        encoded_query = self.return_single_sample(group["query"], is_query=True)

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group["positives"][0]
        else:
            pos_psg = group["positives"][
                (_hashed_seed + epoch) % len(group["positives"])
            ]
        encoded_passages = [self.return_single_sample(pos_psg)]
        neg_size = self.data_args.train_n_passages - 1

        # If there are not enough negatives, take the same negatives again
        if len(group["negatives"]) < neg_size:
            negatives = random.choices(group["negatives"], k=neg_size)
        # If no negatives are wanted, return the query and the positive passage
        elif self.data_args.train_n_passages == 1:
            negatives = []
        # If there are enough negatives, return the query and the positive passage, negatives without shuffling
        elif self.data_args.negative_passage_no_shuffle:
            negatives = group["negatives"][:neg_size]
        # If there are enough negatives, return the query and the positive passage, negatives with shuffling
        else:
            _offset = epoch * neg_size % len(group["negatives"])
            negatives = list(group["negatives"])
            random.Random(_hashed_seed).shuffle(negatives)
            negatives *= 2
            negatives = negatives[_offset : _offset + neg_size]

        encoded_passages.extend(
            self.return_single_sample(neg_psg) for neg_psg in negatives
        )
        return encoded_query, encoded_passages


class HFTrainDataset:
    def __init__(
        self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str
    ):
        """
        :param tokenizer: tokenizer to use, must be a HF tokenizer
        :param data_args: data arguments
        :param cache_dir: cache directory
        """
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_language,
            data_files=data_files,
            cache_dir=cache_dir,
        )[data_args.dataset_split]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(
            self.tokenizer,
            data_args.passage_field_separator,
            data_args.passage_field_separator,
        )

    def preprocess(self, shard_num=1, shard_idx=0):
        """
        Only implemented for already processed datasets.
        :param shard_num: number of shards to split the dataset into
        :param shard_idx: index of the shard to process
        :return: processed dataset
        """
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        return self.dataset


@dataclass
class QueryPassageCollator(DataCollatorWithPadding):
    """
    Helper to convert from List[Tuple[encode_qry, encode_psg]] --> List[qry], List[psg]
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(
        self, features: List[Tuple[BatchEncoding, List[BatchEncoding]]]
    ) -> Tuple[BatchEncoding, BatchEncoding]:
        """
        Perform the actual collation.
        :param features: List of tuples of query and list of passages
        :return: Tuple of query and passages, collated
        """
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]

        # map s.t. each list contains all passages for a query
        if isinstance(queries[0], list):
            # list of BatchEncodings
            queries = sum(queries, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        collated_queries, collated_passages = (
            self.tokenizer.pad(
                q,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            for q, max_len in zip([queries, passages], [self.max_q_len, self.max_p_len])
        )

        return collated_queries, collated_passages
