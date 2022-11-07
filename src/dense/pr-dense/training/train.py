import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)


from ..arguments import (
    ModelArguments,
    DataArguments,
    CustomTrainingArguments as TrainingArguments,
)
from ..models.encode import DenseModel
from ..data.datasets import HFTrainDataset, TrainingDataset, QueryPassageCollator
from ..trainer.grad_cache_trainer import GradCacheTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Directory ({training_args.output_dir}) already exists and is non-empty. Use --overwrite_output_dir to overcome."
        )

    setup_logging(model_args, training_args)

    set_seed(training_args.seed)

    tokenizer, model = setup_model_and_tokenizer(model_args)

    train_dataset = HFTrainDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )

    train_dataset = TrainingDataset(data_args, train_dataset.preprocess(), tokenizer)

    # TRAINING
    trainer = GradCacheTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QueryPassageCollator(
            tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


def setup_model_and_tokenizer(model_args: ModelArguments):
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = DenseModel.create_model(
        model_args, config=config, cache_dir=model_args.cache_dir
    )
    return tokenizer, model


def setup_logging(model_args, training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, n_gpu: %s, 16-bits training: %s",
        training_args.device,
        training_args.n_gpu,
        training_args.fp16,
    )
    logger.info("Training/Eval parameters %s", training_args)
    logger.info("Model parameters %s", model_args)


if __name__ == "__main__":
    main()
