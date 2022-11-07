import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model OR model identifier from huggingface.co/models"
        }
    )
    target_model_path: str = field(
        default=None, metadata={"help": "Path to pretrained re-ranker target model"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path, if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from S3"},
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "Disable weight sharing between query and passage encoders"},
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to training directory"}
    )
    dataset_name: str = field(default=None, metadata={"help": "HF dataset name"})
    passage_field_separator: str = field(default=" ")
    dataset_proc_num: int = field(
        default=8, metadata={"help": "Number of processes during dataset preprocessing"}
    )
    train_n_passages: int = field(
        default=2,
        metadata={
            "help": "Number of passages per query. Negative passages are: train_n_passages - 1"
        },
    )
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "Always use 1st positive passage"}
    )
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "Always use 1st negative passages"}
    )

    encode_in_path: List[str] = field(
        default=None, metadata={"help": "Path to data to encode"}
    )
    encoded_save_path: str = field(
        default=None, metadata={"help": "Where to save the encode"}
    )
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the data downloaded from HF"
        },
    )

    # TODO
    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split("/")
            self.dataset_split = info[-1] if len(info) == 3 else "training"
            self.dataset_name = (
                "/".join(info[:-1]) if len(info) == 3 else "/".join(info)
            )
            self.dataset_language = "default"
            if ":" in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(":")
        else:
            self.dataset_name = "json"
            self.dataset_split = "training"
            self.dataset_language = "default"
        if self.train_dir is None:
            self.train_path = None

        elif os.path.isdir(self.train_dir):
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith("jsonl") or f.endswith("json")
            ]
        else:
            self.train_path = [self.train_dir]


@dataclass
class CustomTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    grad_cache: bool = field(
        default=True, metadata={"help": "Whether to use gradient cache updates"}
    )
    gc_q_chunk_size: int = field(default=4, metadata={"help": "Gradient cache chunk size for queries"})
    gc_p_chunk_size: int = field(default=32, metadata={"help": "Gradient cache chunk size for passages"})
