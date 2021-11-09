import torch

from .. import wave


def test_receptive_field():
    assert wave.compute_receptive_field(dilations=[1], kernel_sizes=[2]) == 2
    assert wave.compute_receptive_field(dilations=[1], kernel_sizes=[3]) == 3
    assert wave.compute_receptive_field(dilations=[1, 2], kernel_sizes=[2, 2]) == 4
    assert (
        wave.compute_receptive_field(dilations=[1, 1, 2], kernel_sizes=[3, 2, 2]) == 6
    )
    assert (
        wave.compute_receptive_field(
            dilations=[1, 1, 2, 4, 8], kernel_sizes=[3, 2, 2, 2, 2]
        )
        == 18
    )

    # assert compute_receptive_field(dilation_seq=[1, 2]) == 4
    # assert compute_receptive_field(dilation_seq=[1, 2, 4]) == 8
    # assert compute_receptive_field(dilation_seq=[1, 2, 4, 1]) == 9
    # assert compute_receptive_field(dilation_seq=[1, 2, 4, 1, 2, 4]) == 15


@torch.no_grad()
def test_wavenet_layer():
    wn = wave.WaveNetLayer(kernel_size=2, dilation=1)
    assert wn.causal_left_pad == 1

    wn = wave.WaveNetLayer(kernel_size=3, dilation=1)
    assert wn.causal_left_pad == 2

    wn = wave.WaveNetLayer(kernel_size=2, dilation=4)
    assert wn.causal_left_pad == 4
    x, skip = wn(torch.rand(1, 32, 1))
    assert x.shape == (1, 32, 1)
    assert skip.shape == (1, 32, 1)
    x, skip = wn(torch.rand(1, 32, 1), h=torch.rand(1, 32, 1))
    assert x.shape == (1, 32, 1)
    assert skip.shape == (1, 32, 1)


@torch.no_grad()
def test_wavenet_input_layer():
    wn = wave.WaveNetInputLayer(kernel_size=5, wave_channels=32, input_channels=256)
    assert wn.causal_left_pad == 4
    x, skip = wn(torch.rand(1, 256, 1))
    assert x.shape == (1, 32, 1)
    assert skip.shape == (1, 32, 1)
    x, skip = wn(torch.rand(1, 256, 1), h=torch.rand(1, 256, 4))
    assert x.shape == (1, 32, 1)
    assert skip.shape == (1, 32, 1)


@torch.no_grad()
def test_wavenet_encode():
    wn = wave.WaveNet(
        quantization_levels=4,
        wave_dilations=[1, 2, 4],
        wave_kernel_size=2,
        wave_channels=8,
        input_kernel_size=1,
    )
    R = wn.receptive_field
    assert R == 8
    x = torch.rand(2, 4, 10)
    e, inputs, skips, outqueues = wn.encode(x, queues=None)
    assert outqueues is None
    assert e.shape == (2, 8, 10)
    assert len(inputs) == len(skips) == 1 + 3  # +1for input layer
    assert all([s.shape == (2, 8, 10) for s in skips])
    assert all([s.shape == (2, 8, 10) for s in inputs[1:]])
    assert inputs[0].shape == x.shape

    # assert not leakage of data
    e, _, _, _ = wn.encode(x)
    e2, _, _, _ = wn.encode(x[..., 0:8])
    assert torch.allclose(e2[..., -1], e[..., 7])
