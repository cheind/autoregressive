import torch
import torch.nn.functional as F


def causal_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """Performs a cause padding to avoid data leakage. Stride is assumed to be one."""
    left_pad = (kernel_size - 1) * dilation
    return F.pad(x, (left_pad, 0))
