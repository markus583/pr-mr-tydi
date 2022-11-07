import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist

# TODO
class SimpleContrastiveLoss:
    """
    In the original DPR paper, the loss is computed as follows:
    - for each query, the positive passage is the first passage in the batch
    - the negatives are all the other passages, both from the same query and from other queries
    --> ALL BM25 negatives + ALL in-batch negatives (as in the original DPR paper)
    """

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        """
        :param x: [batch_size, emb_dim (768)]; query embeddings
        :param y: [batch_size, emb_dim (768)]; passage embeddings
        :param target: [batch_size]; target labels
        :param reduction: 'mean' or 'sum'
        :return: loss
        """
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            # index of the positive passage for each query
            # e.g. if target_per_qry = 2, then target = [0, 2, 4, 6, 8, 10, 12, 14, ...]
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        # compute similarity
        # x: (n_query, 768), y: (n_query * target_per_qry, 768)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)