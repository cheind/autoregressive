__all__ = ["one_hotf"]
import torch
import torch.nn.functional as F


def one_hotf(x: torch.Tensor, quantization_levels: int):
    """Returns x one-hot encoded.

    Params
    ------
    x: (B,T) or (B,Q,T) tensor
        compressed or already one-hot encoded input
    quantization_levels: int
        The total number of quantization levels

    Returns
    -------
    o: (B,Q,T) tensor
        Floating point one-hot encoded input
    """
    if x.dim() == 2:
        # compressed encoding (B,T) -> one encoding (B,Q,T)
        x = F.one_hot(x, num_classes=quantization_levels)  # (B,T,Q)
        x = x.permute(0, 2, 1)  # (B,Q,T)
    return x.float()
