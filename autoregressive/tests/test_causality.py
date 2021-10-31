import torch
import torch.nn

from .. import utils, wave


@torch.no_grad()
class CausalConv(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 2, dilation=d)
        self.conv.weight.fill_(1.0)
        self.conv.bias.fill_(0.0)
        self.dilation = d

    def forward(self, x):
        return self.conv(utils.causal_pad(x, 2, self.dilation))


def test_causal_padding():
    x = torch.arange(1, 257, 1, dtype=torch.float32)  # [1..256]
    L = 7
    layers = [CausalConv(2 ** d) for d in range(L)]
    cc = torch.nn.Sequential(*layers)
    y = cc(x.view(1, 1, -1)).view(-1)
    assert torch.isclose(y[127], x[:128].sum())
    assert torch.isclose(y[255], x[128:].sum())
    assert sum([l.conv.weight.numel() for l in layers]) == 14


def test_data_leakage():
    """Assert no future data is used in computation"""
    model = wave.WaveNetBase(
        in_channels=1,
        out_channels=1,
        wave_channels=8,
        num_blocks=2,
        num_layers_per_block=3,
    )
    assert model.receptive_field == 15
    x = torch.rand(1, 1, 32)
    y = model(x)  # uses all information

    for i in range(10):
        assert torch.allclose(model(x[..., i : i + 15])[..., -1], y[..., i + 15 - 1])


class WaveNetSim(torch.nn.Module):
    def __init__(self, num_blocks: int, num_layers_per_block: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[
                torch.nn.Conv1d(1, 1, 2, dilation=2 ** d)
                for _ in range(num_blocks)
                for d in range(num_layers_per_block)
            ]
        )
        self.receptive_field = wave.compute_receptive_field(
            num_blocks=num_blocks, num_layers_per_block=num_layers_per_block
        )

    def forward(self, x):
        x = utils.causal_pad(x, 2, self.receptive_field - 1)
        x = self.layers(x)
        return x


def test_input_coverage():
    """Check if all inputs are accessed once when num-blocks=1"""

    def setup_weights(m):
        """Setup weights, so that output is sum of inputs"""
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    model = WaveNetSim(num_blocks=1, num_layers_per_block=3).apply(setup_weights)
    assert model.receptive_field == 8
    x = torch.arange(0, 16, 1).view(1, 1, -1).float()
    assert torch.allclose(
        model(x),
        torch.tensor(
            [
                [
                    [
                        0.0,
                        1.0,
                        3.0,
                        6.0,
                        10.0,
                        15.0,
                        21.0,
                        28.0,
                        36.0,
                        44.0,
                        52.0,
                        60.0,
                        68.0,
                        76.0,
                        84.0,
                        92.0,
                    ]
                ]
            ]
        ),
    )
    model = WaveNetSim(num_blocks=2, num_layers_per_block=2).apply(setup_weights)
    assert model.receptive_field == 7
    torch.allclose(
        model(x),
        torch.tensor(
            [
                [
                    [
                        0.0,
                        1.0,
                        4.0,
                        10.0,
                        20.0,
                        33.0,
                        48.0,
                        64.0,
                        80.0,
                        96.0,
                        112.0,
                        128.0,
                        144.0,
                        160.0,
                        176.0,
                        192.0,
                    ]
                ]
            ]
        ),
    )
