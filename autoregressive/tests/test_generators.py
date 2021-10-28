import torch
import torch.nn.functional as F
from .. import wave


def identity_sampler(model, obs, x):
    return x


@torch.no_grad()
def test_generators():
    net = wave.WaveNetBase(
        in_channels=1,
        out_channels=1,
        residual_channels=8,
        skip_channels=8,
        num_blocks=1,
        num_layers_per_block=3,
    )
    assert net.receptive_field == 8
    x = torch.rand(1, 1, 16)
    y = net(x)
    assert y.shape == (1, 1, 16)

    # Next, we compare generators to direct net output y.
    # We do so, by created an artificial padded input sequences
    # (length=receptive field) via unfold and instruct the generator
    # to only generate 1 output. This should then match the content of y.

    xpad = F.pad(x, (7, 0))
    xwnd = xpad.unfold(-1, 8, 1).permute(2, 0, 1, 3).reshape(-1, 1, 8)  # 16,1,8

    gslow = wave.generate(net, xwnd, sampler=identity_sampler)
    yslow_samples, yslow_outputs = wave.slice_generator(gslow, 1)
    assert torch.allclose(yslow_samples.squeeze(), y.squeeze(), atol=1e-4)
    assert torch.allclose(yslow_outputs.squeeze(), y.squeeze(), atol=1e-4)

    gfast = wave.generate_fast(net, xwnd, sampler=identity_sampler)
    yfast_samples, yfast_outputs = wave.slice_generator(gfast, 1)
    assert torch.allclose(yfast_samples.squeeze(), y.squeeze(), atol=1e-4)
    assert torch.allclose(yfast_outputs.squeeze(), y.squeeze(), atol=1e-4)

    # Next, we compare the generators for equality when predicting more than
    # one element
    gslow = wave.generate(net, x[..., :1], sampler=identity_sampler)
    gfast = wave.generate(net, x[..., :1], sampler=identity_sampler)
    yslow_samples, _ = wave.slice_generator(gslow, 60)
    yfast_samples, _ = wave.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # gfast = wave.generate_fast(net, x[..., :1], sampler=identity_sampler)