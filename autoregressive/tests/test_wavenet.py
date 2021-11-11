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
    wn = wave.WaveNetInputLayer(
        kernel_size=5, wave_channels=32, quantization_levels=256
    )
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
    e2, _, _, _ = wn.encode(x[..., 0:R])
    assert torch.allclose(e2[..., -1], e[..., 7])

    # larger input conv
    wn = wave.WaveNet(
        quantization_levels=4,
        wave_dilations=[1, 2, 4],
        wave_kernel_size=2,
        wave_channels=8,
        input_kernel_size=5,
    )
    R = wn.receptive_field
    assert R == 12
    x = torch.rand(2, 4, 14)
    e, inputs, skips, outqueues = wn.encode(x, queues=None)
    assert outqueues is None
    assert e.shape == (2, 8, 14)
    assert len(inputs) == len(skips) == 1 + 3  # +1for input layer
    assert all([s.shape == (2, 8, 14) for s in skips])
    assert all([s.shape == (2, 8, 14) for s in inputs[1:]])
    assert inputs[0].shape == x.shape

    # assert not leakage of data
    e, _, _, _ = wn.encode(x)
    e2, _, _, _ = wn.encode(x[..., 0:R])
    assert torch.allclose(e2[..., -1], e[..., R - 1])

    # assert it works with sparse encoding as well
    x = torch.tensor(
        [
            [0, 1, 3],
            [2, 3, 1],
        ]
    )  # (B,T)
    e, inputs, skips, outqueues = wn.encode(x, queues=None)
    assert outqueues is None
    assert e.shape == (2, 8, 3)
    assert len(inputs) == len(skips) == 1 + 3  # +1for input layer
    assert all([s.shape == (2, 8, 3) for s in skips])
    assert all([s.shape == (2, 8, 3) for s in inputs[1:]])
    assert inputs[0].shape == (2, 4, 3)
    assert torch.allclose(
        inputs[0][0],
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ).T.float(),
    )
    assert torch.allclose(
        inputs[0][1],
        torch.tensor(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ]
        ).T.float(),
    )


def test_coverage_leakage():
    """Check correct inputs are accessed and no data is leaked from the future"""

    def setup_weights(m):
        """Setup weights, so that output is sum of inputs"""
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def check_gradients(model):
        # runs ones through net e = f(x) with gradient enabled for x.
        # ensures that for specific e_i only the correct fields of x receive
        # a gradient
        model = model.apply(setup_weights)
        R = model.receptive_field
        N = 2 * R
        x = torch.ones(N).float().requires_grad_()
        e = model.encode(x.view(1, 1, -1))[0].squeeze()
        for i in range(N):
            e[i].backward(inputs=x, create_graph=True, retain_graph=i < N)
            # outputs at i have positive gradients (because of inputs=1 and the way weights are setup + tanh/sigmoid) wrt to inputs (i-R,i]
            mask = torch.zeros(N, dtype=bool)
            mask[max(0, i - R + 1) : (i + 1)] = True
            assert all(x.grad.data[mask] > 0.0)
            assert all(x.grad.data[~mask] == 0.0)
            x.grad = None

    # Standard model
    model = wave.WaveNet(
        quantization_levels=1, wave_dilations=[1, 2, 4], wave_channels=1
    )
    check_gradients(model)

    # Model with larger input kernel size
    model = wave.WaveNet(
        quantization_levels=1,
        wave_dilations=[1, 2, 4],
        wave_channels=1,
        input_kernel_size=3,
    )
    check_gradients(model)

    # Model with repeated dilations
    model = wave.WaveNet(
        quantization_levels=1,
        wave_dilations=[1, 2, 4, 1, 2],
        wave_channels=1,
        input_kernel_size=3,
    )
    check_gradients(model)