import torch
from .. import wave, generators


def identity_sampler(logits):
    return logits


@torch.no_grad()
def test_generators():
    # Pretty useless to use quantization level of 1, but we use floats in the tests.
    net = wave.WaveNet(
        quantization_levels=1,
        wave_channels=8,
        input_kernel_size=3,
        wave_dilations=[1, 2, 4],
    )
    R = net.receptive_field
    assert R == 10
    x = torch.rand(1, 1, 16)
    y, _ = net(x)
    assert y.shape == (1, 1, 16)

    # Next, we compare the initial generator prediction to net output y. This can be
    # done as generators produce as first prediction the net result of first observation.
    for i in range(16):
        gslow = generators.generate(net, x[..., : (i + 1)], sampler=identity_sampler)
        gfast = generators.generate_fast(
            net, x[..., : (i + 1)], sampler=identity_sampler
        )
        yslow_samples, yslow_outputs = generators.slice_generator(gslow, 1)  # predict 1
        yfast_samples, yfast_outputs = generators.slice_generator(gfast, 1)  # predict 1
        assert torch.allclose(yslow_samples.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yslow_outputs.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yfast_samples.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yfast_outputs.squeeze(), y[..., i].squeeze(), atol=1e-4)

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

    # Finally, we check verify that providing pre-computed layerinputs
    # gives same result as compared when not providing it.
    gslow = generators.generate(net, x, sampler=identity_sampler)
    _, layer_inputs, _, _ = net.encode(x)
    gfast = generators.generate_fast(
        net, x, sampler=identity_sampler, layer_inputs=layer_inputs
    )
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)


@torch.no_grad()
def test_rolling_origin():
    torch.manual_seed(123)
    model = wave.WaveNet(
        wave_dilations=[1, 2, 4],
        quantization_levels=1,
        wave_channels=8,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 1, 16)
    x = seq[..., :-1]
    y, _ = model(x)

    _, rolls_logits, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=True,
    )
    # first pred is [7,8,9,10] using obs 0..7
    # next is [8,9,10,11] using obs 1..8
    # up to 15
    assert torch.allclose(yidx, torch.tensor([7, 8, 9, 10, 11]))
    assert rolls_logits.shape == (5, 2, 1, 4)
    assert torch.allclose(rolls_logits[0, :, :, 0], y[..., 7])
    assert torch.allclose(rolls_logits[1, :, :, 0], y[..., 8])
    assert torch.allclose(rolls_logits[2, :, :, 0], y[..., 9])
    assert torch.allclose(rolls_logits[3, :, :, 0], y[..., 10])
    assert torch.allclose(rolls_logits[4, :, :, 0], y[..., 11])

    # Test some parameter variantions
    _, _, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=False,
    )
    assert torch.allclose(yidx, torch.arange(0, 12, 1))
    _, _, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=True,
        num_origins=2,
        random_origins=True,
    )
    assert len(set(yidx.tolist()) & set([8, 10])) == 2
    _, _, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=True,
        num_origins=2,
        random_origins=False,
    )
    assert torch.allclose(yidx, torch.tensor([7, 8]))
