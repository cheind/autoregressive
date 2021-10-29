import torch

from .. import wave, losses


@torch.no_grad()
def test_rolling_nstep():
    torch.manual_seed(123)
    model = wave.WaveNetBase(
        in_channels=1,
        out_channels=1,
        wave_channels=8,
        num_blocks=1,
        num_layers_per_block=3,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 1, 16)
    x = seq[..., :-1]
    y = model(x)

    yhat, yout, yidx = losses.rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=4,
        detach_sample=True,
        skip_partial=True,
    )
    # first pred is [7,8,9,10] using obs 0..7
    # next is [8,9,10,11] using obs 1..8
    # up to 15
    assert torch.allclose(yidx, torch.tensor([7, 8, 9, 10, 11]))
    assert yhat.shape == (5, 2, 1, 4)
    assert torch.allclose(yhat[0, :, :, 0], y[..., 7])
    assert torch.allclose(yhat[1, :, :, 0], y[..., 8])
    assert torch.allclose(yhat[2, :, :, 0], y[..., 9])
    assert torch.allclose(yhat[3, :, :, 0], y[..., 10])
    assert torch.allclose(yhat[4, :, :, 0], y[..., 11])

    # Test some parameter variantions
    _, _, yidx = losses.rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=4,
        detach_sample=True,
        skip_partial=False,
    )
    assert torch.allclose(yidx, torch.arange(0, 12, 1))
    _, _, yidx = losses.rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=4,
        detach_sample=True,
        skip_partial=True,
        max_rolls=2,
        random_rolls=True,
    )
    assert torch.allclose(yidx, torch.tensor([10, 9]))
    _, _, yidx = losses.rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=4,
        detach_sample=True,
        skip_partial=True,
        max_rolls=2,
        random_rolls=False,
    )
    assert torch.allclose(yidx, torch.tensor([7, 8]))


@torch.no_grad()
def test_rolling_nstep_mae():
    torch.manual_seed(123)
    model = wave.WaveNetBase(
        in_channels=1,
        out_channels=1,
        wave_channels=8,
        num_blocks=1,
        num_layers_per_block=3,
    )
    assert model.receptive_field == 8
    seq = torch.rand(2, 1, 16)
    x = seq[..., :-1]
    y = model(x)  # untrained model, use model forward as surrogate

    roll_y, _, roll_idx = losses.rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=2,
        detach_sample=True,
        skip_partial=True,
    )
    # The first generated item should be equivalent to what the model yields
    loss = losses.rolling_nstep_mae(roll_y[..., :1], roll_idx, y)
    assert torch.allclose(loss, torch.tensor(0.0))
    # More should introduce an error for an untrained model
    loss = losses.rolling_nstep_mae(roll_y, roll_idx, y)
    assert loss > 0.0
