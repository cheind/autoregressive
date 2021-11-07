import torch
import torch.nn.functional as F
from .. import wave, generators


def identity_sampler(logits):
    return logits


@torch.no_grad()
def test_generators():
    # Pretty useless to use quantization level of 1, but we use floats in the tests.
    net = wave.WaveNet(dilations=[1, 2, 4], quantization_levels=1, wave_channels=8)
    assert net.receptive_field == 8
    x = torch.rand(1, 1, 16)
    y = net(x)
    assert y.shape == (1, 1, 16)

    # Next, we compare generators to net output y.
    # We do so, by creating an artificial padded input sequence
    # (length=receptive field), then unfold into individual 16 curves of length 8 (step=1).
    # We feed these as input to the generator and compare the 16 outputs that should match y.

    xpad = F.pad(x, (7, 0))
    xwnd = xpad.unfold(-1, 8, 1).permute(2, 0, 1, 3).reshape(-1, 1, 8)  # 16,1,8
    # first sequence will be [0,0,0,0,0,0,0,x[0]] and should match y[0]
    # second will be         [0,0,0,0,0,0,x[0],x[1]] and should match y[1]
    # ...

    gslow = generators.generate(net, xwnd, sampler=identity_sampler)
    yslow_samples, yslow_outputs = generators.slice_generator(gslow, 1)
    assert torch.allclose(yslow_samples.squeeze(), y.squeeze(), atol=1e-4)
    assert torch.allclose(yslow_outputs.squeeze(), y.squeeze(), atol=1e-4)

    gfast = generators.generate_fast(net, xwnd, sampler=identity_sampler)
    yfast_samples, yfast_outputs = generators.slice_generator(gfast, 1)
    assert torch.allclose(yfast_samples.squeeze(), y.squeeze(), atol=1e-4)
    assert torch.allclose(yfast_outputs.squeeze(), y.squeeze(), atol=1e-4)

    # Next, we compare the generators for equality when predicting more than
    # one element, given a single observation (i.e empty queues)
    gslow = generators.generate(net, x[..., :1], sampler=identity_sampler)
    gfast = generators.generate_fast(net, x[..., :1], sampler=identity_sampler)
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Next, similar as above but with more inputs (partial receptive field)
    gslow = generators.generate(net, x[..., :3], sampler=identity_sampler)
    gfast = generators.generate_fast(net, x[..., :3], sampler=identity_sampler)
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Next, similar as above but with all inputs
    gslow = generators.generate(net, x, sampler=identity_sampler)
    gfast = generators.generate_fast(net, x, sampler=identity_sampler)
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Finally, we check verify that providing pre-computed layerinputs work
    # as expected.
    gslow = generators.generate(net, x, sampler=identity_sampler)
    _, layer_inputs, _ = net.encode(x, remove_left_invalid=False)
    gfast = generators.generate_fast(
        net, x, sampler=identity_sampler, layer_inputs=layer_inputs
    )
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)
