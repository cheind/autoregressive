import torch
import torch.nn
import torch.nn.functional as F


def causal_pad(x: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    """Performs a cause padding to avoid data leakage. Stride is assumed to be one."""
    left_pad = (kernel - 1) * dilation
    return F.pad(x, (left_pad, 0))


class BasicBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(causal_pad(x, self.kernel_size, self.dilation))
        y = F.leaky_relu(self.bn(y))
        return y
