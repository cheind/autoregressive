import torch
import torch.nn.functional as F

from .. import wave, losses, generators


def identity_sampler(logits):
    return logits


@torch.no_grad()
def test_rolling_nstep_ce():
    torch.manual_seed(123)
    model = wave.WaveNet(
        wave_dilations=[1, 2, 4],
        quantization_levels=4,
        wave_channels=8,
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
    loss = wave._rolling_origin_ce(roll_logits, roll_idx, targets)

    expected_loss = 0.0
    for ridx in roll_idx:
        expected_loss += -(
            model_logprobs[0, targets[0, ridx], ridx]
            + model_logprobs[1, targets[1, ridx], ridx]
            + model_logprobs[0, targets[0, ridx + 1], ridx + 1]
            + model_logprobs[1, targets[1, ridx + 1], ridx + 1]
        )
    expected_loss /= M * 4
    assert torch.allclose(loss, expected_loss, atol=1e-2)
