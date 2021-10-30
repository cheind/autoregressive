import torch
from .. import wave


def test_data_leakage():
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
