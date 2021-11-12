__all__ = ["to_one_hot"]
import torch
import torch.nn.functional as F


def to_one_hot(x: torch.Tensor, num_classes: int):
    if x.dim() == 2:
        # compressed encoding (B,T) -> one encoding (B,Q,T)
        x = F.one_hot(x, num_classes=num_classes)  # (B,T,Q)
        x = x.permute(0, 2, 1)  # (B,Q,T)
    return x.float()
