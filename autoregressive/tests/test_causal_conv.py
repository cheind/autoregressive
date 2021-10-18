import torch
import torch.nn
import torch.nn.functional as F

from ..layers import causal_pad


@torch.no_grad()
class CausalConv(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 2, dilation=d)
        self.conv.weight.fill_(1.0)
        self.conv.bias.fill_(0.0)
        self.dilation = d

    def forward(self, x):
        return self.conv(causal_pad(x, 2, self.dilation))


def test_causal_convolutions():
    x = torch.arange(1, 257, 1, dtype=torch.float32)  # [1..256]
    L = 7
    layers = [CausalConv(2 ** d) for d in range(L)]
    cc = torch.nn.Sequential(*layers)
    y = cc(x.view(1, 1, -1)).view(-1)
    assert torch.isclose(y[127], x[:128].sum())
    assert torch.isclose(y[255], x[128:].sum())
    assert (sum([l.conv.weight.numel() for l in layers]), 14)
