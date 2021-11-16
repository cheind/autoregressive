import torch
import torch.nn.functional as F

from .. import metrics, wave, generators


def test_sample_entropy():
    # Uniform random
    x = torch.rand(10, 1024)
    se = metrics.sample_entropy(x)
    assert se.mean() >= 2.0

    # Straight lines
    x = torch.arange(2 ** 12).float()
    se = metrics.sample_entropy(x).mean()
    assert abs(se) < 1e-3

    # Sine
    x = torch.sin(torch.linspace(0, 10 * 3.145, 2 ** 12))
    se = metrics.sample_entropy(x).mean()
    assert se < 0.2


def identity_sampler(logits):
    return logits


@torch.no_grad()
def test_cross_entropy_ro():
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
    loss = metrics.cross_entropy_ro(roll_logits, roll_idx, targets)

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
