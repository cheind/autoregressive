import torch
import torch.nn.functional as F

from .. import wave, losses


def identity_sampler(logits):
    return logits


@torch.no_grad()
def test_rolling_nstep():
    torch.manual_seed(123)
    model = wave.WaveNet(
        dilations=[1, 2, 4],
        quantization_levels=1,
        wave_channels=8,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 1, 16)
    x = seq[..., :-1]
    y = model(x)

    _, rolls_logits, yidx = losses.rolling_nstep(
        model,
        identity_sampler,
        x,
        num_generate=4,
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
    _, _, yidx = losses.rolling_nstep(
        model,
        identity_sampler,
        x,
        num_generate=4,
        skip_partial=False,
    )
    assert torch.allclose(yidx, torch.arange(0, 12, 1))
    _, _, yidx = losses.rolling_nstep(
        model,
        identity_sampler,
        x,
        num_generate=4,
        skip_partial=True,
        max_rolls=2,
        random_rolls=True,
    )
    assert len(set(yidx.tolist()) & set([9, 10])) == 2
    _, _, yidx = losses.rolling_nstep(
        model,
        identity_sampler,
        x,
        num_generate=4,
        skip_partial=True,
        max_rolls=2,
        random_rolls=False,
    )
    assert torch.allclose(yidx, torch.tensor([7, 8]))


@torch.no_grad()
def test_rolling_nstep_ce():
    torch.manual_seed(123)
    model = wave.WaveNet(
        dilations=[1, 2, 4],
        quantization_levels=4,
        wave_channels=8,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 4, 16)
    x = seq[..., :-1]
    model_logprobs = F.log_softmax(model(x), 1)
    targets = torch.randint(0, 4, (2, 15))

    N = 2
    M = 4
    _, roll_logits, roll_idx = losses.rolling_nstep(
        model, identity_sampler, x, num_generate=N, skip_partial=True, max_rolls=M
    )
    loss = losses.rolling_nstep_ce(roll_logits, roll_idx, targets)

    expected_loss = 0.0
    for ridx in roll_idx:
        expected_loss += -(
            model_logprobs[0, targets[0, ridx], ridx]
            + model_logprobs[1, targets[1, ridx], ridx]
            + model_logprobs[0, targets[0, ridx + 1], ridx + 1]
            + model_logprobs[1, targets[1, ridx + 1], ridx + 1]
        )
    expected_loss /= M * 4
    assert torch.allclose(loss, expected_loss, atol=1e-1)
