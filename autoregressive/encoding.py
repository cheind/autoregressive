__all__ = ["one_hotf"]
import torch
from torch._C import dtype
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


def positional_encoding_lut(
    length: int, base: int = 10000, depth: int = 128, device: torch.device = None
) -> torch.Tensor:
    """Computes a positional encoding lookup tensor based on transformer sin/cos encodings.

    Results in an encoding that is
     - unique for each timestep
     - distance between timesteps is consistent across different length sequences
     - generalizes to longer sentences without values becoming unbounded
     - deterministic
    """
    assert depth % 2 == 0, "Depth should be even"

    # 1 0 1 0...
    imask = torch.ones(depth)
    imask[::2] = 0
    imask.view(depth, 1)

    k_upper = depth // 2
    k = torch.arange(0, k_upper, 1)
    wk = 1 / torch.pow(base, (2 * k) / depth)
    wk = wk.repeat_interleave(2, 0)
    wk = wk.view(1, depth)

    t = torch.arange(0, length, 1).view(length, 1)

    posenc = torch.sin(wk * t) * imask + torch.cos(wk * t) * (1 - imask)
    return posenc.to(device).T  # (C,T)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pe = positional_encoding_lut(28 * 28, depth=64)

    plt.imshow(pe, origin="lower")
    plt.ylabel("Depth")
    plt.xlabel("Timestep")
    plt.show()