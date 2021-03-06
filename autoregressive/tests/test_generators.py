import torch
import torch.nn.functional as F
from .. import wave, generators, sampling


def identity_sampler(logits):
    return logits


def test_recent_buffer():
    r = generators.RecentBuffer((4,), dtype=int)
    assert r.buffer.shape == (0,)
    r.add(torch.tensor([1, 2]))
    assert torch.allclose(r.buffer, torch.tensor([1, 2]))
    r.add(torch.tensor([3, 4, 5]))
    assert torch.allclose(r.buffer, torch.tensor([2, 3, 4, 5]))
    r.add(torch.tensor([6]))
    assert torch.allclose(r.buffer, torch.tensor([3, 4, 5, 6]))
    r.add(torch.arange(7, 100, 1))
    assert torch.allclose(r.buffer, torch.tensor([96, 97, 98, 99]))
    r.add(torch.empty((0,)))
    assert torch.allclose(r.buffer, torch.tensor([96, 97, 98, 99]))

    r = generators.RecentBuffer((2, 3, 4), dtype=torch.float32)
    assert r.buffer.shape == (2, 3, 0)

    x0 = torch.rand(2, 3, 2)
    r.add(x0)
    assert torch.allclose(r.buffer, x0)
    x1 = torch.rand(2, 3, 3)
    r.add(x1)
    assert torch.allclose(r.buffer, torch.cat((x0[..., -1:], x1), -1))
    x2 = torch.rand(2, 3, 100)
    r.add(x2)
    assert torch.allclose(r.buffer, x2[..., -4:])


@torch.no_grad()
def test_generators():
    # Pretty useless to use quantization level of 1, but we use floats in the tests.
    net = wave.WaveNet(
        quantization_levels=1,
        residual_channels=8,
        dilation_channels=8,
        skip_channels=8,
        input_kernel_size=3,
        wave_dilations=[1, 2, 4],
        cond_channels=3,
    )
    R = net.receptive_field
    assert R == 10
    x = torch.rand(1, 1, 16)
    c = torch.rand(1, 3, 1)  # global cond
    y, _ = net(x, c=c)
    assert y.shape == (1, 1, 16)

    # Next, we compare the initial generator prediction to net output y. This can be
    # done as generators produce as first prediction the net result of first observation. # noqa:E501
    for i in range(16):
        gslow = generators.generate(
            net, x[..., : (i + 1)], sampler=identity_sampler, global_cond=c
        )
        gfast = generators.generate_fast(
            net, x[..., : (i + 1)], sampler=identity_sampler, global_cond=c
        )
        yslow_samples, yslow_outputs = generators.slice_generator(gslow, 1)  # predict 1
        yfast_samples, yfast_outputs = generators.slice_generator(gfast, 1)  # predict 1
        assert torch.allclose(yslow_samples.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yslow_outputs.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yfast_samples.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yfast_outputs.squeeze(), y[..., i].squeeze(), atol=1e-4)

    # Next, we compare the generators for equality when predicting more than
    # one element, given a single observation (i.e empty queues)
    gslow = generators.generate(
        net, x[..., :1], sampler=identity_sampler, global_cond=c
    )
    gfast = generators.generate_fast(
        net, x[..., :1], sampler=identity_sampler, global_cond=c
    )
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Next, similar as above but with more inputs (partial receptive field)
    gslow = generators.generate(
        net, x[..., :3], sampler=identity_sampler, global_cond=c
    )
    gfast = generators.generate_fast(
        net, x[..., :3], sampler=identity_sampler, global_cond=c
    )
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Next, similar as above but with all inputs
    gslow = generators.generate(net, x, sampler=identity_sampler, global_cond=c)
    gfast = generators.generate_fast(net, x, sampler=identity_sampler, global_cond=c)
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)

    # Finally, we check verify that providing pre-computed layerinputs
    # gives same result as compared when not providing it.
    gslow = generators.generate(net, x, sampler=identity_sampler, global_cond=c)
    _, layer_inputs, _ = net.encode(x, c=c)
    gfast = generators.generate_fast(
        net, x, sampler=identity_sampler, layer_inputs=layer_inputs, global_cond=c
    )
    yslow_samples, _ = generators.slice_generator(gslow, 60)
    yfast_samples, _ = generators.slice_generator(gfast, 60)
    assert yslow_samples.shape == (1, 1, 60)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)


@torch.no_grad()
def test_compressed_generators():
    # Pretty useless to use quantization level of 1, but we use floats in the tests.
    net = wave.WaveNet(
        quantization_levels=4,
        residual_channels=8,
        dilation_channels=8,
        skip_channels=8,
        input_kernel_size=3,
        wave_dilations=[1, 2, 4],
    )
    R = net.receptive_field
    assert R == 10
    x = torch.randint(0, 4, (1, 16))  # (B,T)
    y, _ = net(x)
    assert y.shape == (1, 4, 16)

    # Next, we compare the initial generator prediction to net output y. This can be
    # done as generators produce as first prediction the net result of first observation. # noqa:E501
    for i in range(16):
        gslow = generators.generate(
            net, x[..., : (i + 1)], sampler=sampling.sample_greedy
        )
        gfast = generators.generate_fast(
            net, x[..., : (i + 1)], sampler=sampling.sample_greedy
        )
        yslow_samples, yslow_logits = generators.slice_generator(gslow, 1)  # predict 1
        yfast_samples, yfast_logits = generators.slice_generator(gfast, 1)  # predict 1
        assert torch.allclose(yslow_logits.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(yfast_logits.squeeze(), y[..., i].squeeze(), atol=1e-4)
        assert torch.allclose(
            yslow_samples.squeeze(), y[..., i].argmax(1).squeeze(), atol=1e-4
        )
        assert torch.allclose(
            yfast_samples.squeeze(), y[..., i].argmax(1).squeeze(), atol=1e-4
        )

    # # Next, we compare the generators for equality when predicting more than
    # # one element, given a single observation (i.e empty queues)
    gslow = generators.generate(net, x[..., :1], sampler=sampling.sample_greedy)
    gfast = generators.generate_fast(net, x[..., :1], sampler=sampling.sample_greedy)
    yslow_samples, yslow_logits = generators.slice_generator(gslow, 60)
    yfast_samples, yfast_logits = generators.slice_generator(gfast, 60)
    assert yslow_logits.shape == (1, 4, 60)
    assert yfast_logits.shape == (1, 4, 60)
    assert yslow_samples.shape == (1, 60)
    assert yfast_samples.shape == (1, 60)
    assert torch.allclose(yslow_logits, yfast_logits, atol=1e-4)
    assert torch.allclose(yslow_samples, yfast_samples, atol=1e-4)


@torch.no_grad()
def test_rolling_origin():
    torch.manual_seed(123)
    model = wave.WaveNet(
        wave_dilations=[1, 2, 4],
        quantization_levels=1,
        residual_channels=8,
        dilation_channels=8,
        skip_channels=8,
        cond_channels=3,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 1, 16)
    x = seq[..., :-1]
    c = torch.rand(1, 3, 1)  # global cond
    y, _ = model(x, c=c)

    _, rolls_logits, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=True,
        global_cond=c,
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

    # Test some parameter variations
    _, _, yidx = generators.rolling_origin(
        model, identity_sampler, x, horizon=4, skip_partial=False, global_cond=c
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
        global_cond=c,
    )
    # the following will fail if the random state has changed,
    # which might be ok (i.e. added lines of code that require
    # random numbers).
    assert len(set(yidx.tolist()) & set([7, 10])) == 2
    _, _, yidx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=4,
        skip_partial=True,
        num_origins=2,
        random_origins=False,
        global_cond=c,
    )
    assert torch.allclose(yidx, torch.tensor([7, 8]))


@torch.no_grad()
def test_collate():
    torch.manual_seed(123)
    model = wave.WaveNet(
        wave_dilations=[1, 2, 4],
        quantization_levels=4,
        residual_channels=8,
        dilation_channels=8,
        skip_channels=8,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 4, 16)
    x = seq[..., :-1]
    model_logprobs = F.log_softmax(model(x)[0], 1)
    targets = torch.randint(0, 4, (2, 15))

    N = 2
    M = 4
    _, roll_logits, roll_idx = generators.rolling_origin(
        model,
        identity_sampler,
        x,
        horizon=N,
        skip_partial=True,
        num_origins=M,
    )
    co_logits, co_targets = generators.collate_rolling_origin(
        roll_logits, roll_idx, targets
    )
    loss = F.cross_entropy(co_logits, co_targets)

    expected_loss = 0.0
    for ridx in roll_idx:
        expected_loss += -(
            model_logprobs[0, targets[0, ridx], ridx]  # noqa: W503
            + model_logprobs[1, targets[1, ridx], ridx]  # noqa: W503
            + model_logprobs[0, targets[0, ridx + 1], ridx + 1]  # noqa: W503
            + model_logprobs[1, targets[1, ridx + 1], ridx + 1]  # noqa: W503
        )
    expected_loss /= M * 4
    assert torch.allclose(loss, expected_loss, atol=1e-2)
