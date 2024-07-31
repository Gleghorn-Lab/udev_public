### imports
import torch
import torch.nn.functional as F
from functools import partial
from torchmetrics.functional import pairwise_cosine_similarity


cossim = partial(pairwise_cosine_similarity, zero_diagonal=True)


def diff_loss(batch1: torch.Tensor, batch2: torch.Tensor) -> float: # (B, d) (B, d)
    batch1_sim = pairwise_cosine_similarity(batch1) # (B, B)
    batch2_sim = pairwise_cosine_similarity(batch2) # (B, B)
    return F.mse_loss(batch1_sim, batch2_sim) # match intra-latent relationships


def space_loss(Z_1: torch.Tensor, Z_2: torch.Tensor, lambda_1: float = 1.0, lambda_2: float = 0.1) -> float: # (B, d) (B, d)
    C_1 = cossim(Z_1)
    C_2 = cossim(Z_2)
    diff = F.mse_loss(C_1, C_2) # match intra-latent relationships
    anti_trivial = (lambda_1 * C_1 + lambda_2 * C_2).mean() # prevent trivial solution
    return diff + anti_trivial


def MNR_loss(batch1: torch.Tensor, batch2: torch.Tensor) -> float:
    logits = pairwise_cosine_similarity(batch1, batch2) # (B, B)
    targets = torch.arange(len(batch1), device=batch1.device) # (B,)
    return F.cross_entropy(logits, targets)


def clip_loss(batch1: torch.Tensor, batch2: torch.Tensor, temp: float = 1.0) -> float:
    """
    batch1, batch2 - both torch.Tensor (batch_size, hidden_size)
    This function takes two batches of vectors and returns the clip loss.
    It uses dot product as the similarity function.
    The output of the similarity function can be divided by a learned temperature value.
    """
    logits = (batch1 @ batch2.T) / temp
    batch1_similarity = batch1 @ batch1.T
    batch2_similarity = batch2 @ batch2.T
    targets = F.softmax((batch1_similarity + batch2_similarity) / 2 * temp, dim=-1)
    batch1_loss = F.cross_entropy(logits, targets.argmax(dim=1))
    batch2_loss = F.cross_entropy(logits.T, targets.T.argmax(dim=1))
    loss =  (batch1_loss + batch2_loss) / 2.0
    return loss


### tests
if __name__ == '__main__':
    bs = 8
    d = 512
    a = torch.rand(bs, d)
    b = torch.rand(bs, d)
    print('MNR')
    print(MNR_loss(a, b), MNR_loss(b, a), MNR_loss(a, a), MNR_loss(b, b))
    print('CLIP')
    print(clip_loss(a, b), clip_loss(b, a), clip_loss(a, a), clip_loss(b, b))
    print('diff')
    print(diff_loss(a, b), diff_loss(b, a), diff_loss(a, a), diff_loss(b, b))
