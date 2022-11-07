import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader

from .loss import SimpleContrastiveLoss

import logging

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("GradCache not found!") from e


class CustomTrainer(Trainer):
    """
    Implementation of the Trainer using Gradient Caching.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Trainer.
        :param args: Arguments to pass to the Trainer.
        :param kwargs: Keyword arguments to pass to the Trainer.
        """
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, **kwargs) -> None:
        """
        Save the model to the output directory for checkpointing.
        :param output_dir: The output directory to save the model to.
        :param kwargs: Keyword arguments to pass to the save
        :return: None
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
        self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        """
        Prepare the inputs for the model.
        :param inputs: The inputs to prepare.
        :return: The prepared inputs.
        """
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared
    def get_train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.
        :return: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, **kwargs):
        """
        Compute the loss for the model.
        :param model: The model to compute the loss for.
        :param inputs: The inputs to compute the loss for.
        :param kwargs: Keyword arguments to pass to the loss
        :return: Loss value.
        """
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        """
        Perform a training step.
        :param args: Arguments to pass to the training step.
        :return: The loss value.
        """
        return super(CustomTrainer, self).training_step(*args)

    @staticmethod
    def _get_dense_rep(x):
        """
        Get the dense representation from the model output.
        If the model output is a tuple, the dense representation is the first element.
        :param x: The model output.
        :return: The dense representation.
        """
        return x.p_reps if x.q_reps is None else x.q_reps

    @staticmethod
    def _split_dense_inputs(model_input: dict, chunk_size: int) -> List[dict]:
        """
        Split the dense inputs into chunks of size chunk_size. Used for gradient caching.
        :param model_input: The model input.
        :param chunk_size: The chunk size for the split in gradient caching.
        :return: The split model input.
        """

        arg_key = list(model_input.keys())[0]
        arg_val = model_input[arg_key]

        keys = list(arg_val.keys())
        # split the dense inputs into chunks of size chunk_size
        chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
        chunked_arg_val = [
            dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))
        ]

        return [{arg_key: c} for c in chunked_arg_val]


class GradCacheTrainer(CustomTrainer):
    """
    Custom Trainer class for Gradient Caching.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Trainer.
        :param args: Arguments to pass to the Trainer.
        :param kwargs: Keyword arguments to pass to the Trainer.
        """
        logger.info("Initializing Gradient Cache Trainer Class.")
        super(GradCacheTrainer, self).__init__(*args, **kwargs)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=SimpleContrastiveLoss(),
            split_input_fn=self._split_dense_inputs,
            get_rep_fn=self._get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def training_step(self, model: torch.nn.Module, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]) -> float:
        """
        Perform a training step.
        :param model: The model to perform the training step for.
        :param inputs: The inputs to perform the training step for.
        :return: The loss value.
        """
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {"query": queries}, {"passage": passages}

        self.gc.models = [model, model]
        return self.gc(queries, passages)
