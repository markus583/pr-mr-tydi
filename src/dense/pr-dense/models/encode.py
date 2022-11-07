import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput

from ..arguments import ModelArguments, CustomTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    """
    Output type of :class:`~transform

    Attributes:
        q_reps (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        p_reps (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`, `optional`,
        returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
    """
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    CLS_FOR_HF_TRANSFORMER = AutoModel

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        untie_encoder: bool = False,
    ):
        """
        Universal encoder for query and passage.
        :param lm_q: query language model
        :param lm_p: passage language model
        :param untie_encoder: if True, use different encoder for query and passage
        """
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss()
        self.untie_encoder = untie_encoder

    @classmethod
    def create_model(
        cls,
        model_args: ModelArguments,
        **hf_kwargs,
    ):
        """
        Create a new EncoderModel instance.
        :param model_args: ModelArguments instance
        :param hf_kwargs: Keyword arguments to pass to the HF model
        :return: EncoderModel instance
        """
        lm_q = cls.CLS_FOR_HF_TRANSFORMER.from_pretrained(
            model_args.model_name_or_path, **hf_kwargs
        )
        lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        return cls(
            lm_q=lm_q,
            lm_p=lm_p,
            untie_encoder=model_args.untie_encoder,
        )

    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # INFERENCE
        if any(x is None for x in [q_reps, p_reps]):
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # TRAINING
        if self.training:
            # compute similarity and transform to scores
            scores = self.compute_similarity(q_reps, p_reps).view(q_reps.size(0), -1)

            # positive passage is the first one
            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
        # EVAL
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )



    def encode_passage(self, psg):
        raise NotImplementedError("EncoderModel is abstract class!")

    def encode_query(self, qry):
        raise NotImplementedError("EncoderModel is abstract class!")

    def compute_loss(self, scores: Tensor, target: Tensor) -> Tensor:
        """
        Compute loss for training
        :param scores: scores for each query-passage pair
        :param target: index of the correct passage for each query
        :return: loss
        """
        return self.cross_entropy(scores, target)

    @staticmethod
    def compute_similarity(q_reps : Tensor, p_reps : Tensor) -> Tensor:
        """
        Compute similarity between query and passage representations
        :param q_reps: query representations
        :param p_reps: passage representations
        :return: similarity scores
        """
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def save(self, output_dir: str) -> None:
        """
        Save model to output_dir
        :param output_dir: directory to save model
        :return: None
        """
        if self.untie_encoder:
            # save query and passage models separately
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            self.lm_q.save_pretrained(output_dir)

class DenseModel(EncoderModel):
    """
    Dense encoder model
    """
    def encode_passage(self, psg: Dict[str, Tensor] = None) -> Union[Tensor, None]:
        """
        Encode passage
        :param psg: passage input
        :return: passage hidden representations
        """
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        return p_hidden[:, 0]

    def encode_query(self, qry: Dict[str, Tensor] = None) -> Union[Tensor, None]:
        """
        Encode query
        :param qry: query input
        :return: query hidden representations
        """
        if qry is None:
            return None
        query_out = self.lm_q(**qry, return_dict=True)
        q_hidden = query_out.last_hidden_state
        return q_hidden[:, 0]
