import torch
import torch.nn.functional as F
from scipy.stats import chi2

from .. import sampling


def _chi2test(s: torch.Tensor, pi: torch.Tensor, p=0.05) -> bool:
    """Returns true if the null hypothesis of bin-probabilities pi for observations s is accepted. """  # noqa:E501
    # Number of categories
    K = len(pi)
    # Number of observations
    N = len(s)
    # Normalize
    pi = pi / pi.sum()
    # Count bin frequencies
    x = torch.histc(s.float(), bins=K, min=0, max=(K - 1))  # Count observations
    # Computed the expected frequencies under h0
    expected = pi * N
    # Compute the statistic which (approx. chi2 distributed)
    chi2_value = ((x - expected) ** 2 / expected).sum()
    # Compute the p-value, i.e P(X >= chi2_value)
    chi2_crit = chi2.ppf(1.0 - p, df=K - 1)
    # If the probability of observing a chi2_value or something more extreme
    # is less than p we reject h0
    # print(chi2_value.item(), chi2_crit)
    return chi2_value <= chi2_crit


def test_greedy_sampler():
    torch.manual_seed(123)
    logits = torch.rand(2, 10, 5000)
    samples = sampling.sample_greedy(logits)
    assert samples.shape == (2, 5000)
    assert _chi2test(samples.view(-1), torch.ones(10) * 0.1)

    logits = torch.rand(2, 10, 10000)
    logits[:, :2, :] += 0.1
    samples = sampling.sample_greedy(logits)
    assert not _chi2test(samples.view(-1), torch.ones(10) * 0.1)
    assert _chi2test(
        samples.view(-1),
        torch.tensor([0.18, 0.18, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
    )

    # Check (B,Q,T) sampling
    logits = torch.arange(24).view(4, 3, 2).float()
    samples = sampling.sample_greedy(logits)
    assert samples.shape == (4, 2)


def test_stochastic_sampler():
    torch.manual_seed(123)
    logits = torch.tensor([1.0, 2.0, 1.0, 1.0]).view(1, 4, 1).repeat(1, 1, 20000)
    samples = sampling.sample_stochastic(logits)
    assert samples.shape == (1, 20000)
    assert _chi2test(
        samples.view(-1), torch.tensor([3500, 9400, 3500, 3500]) / 20000, p=0.01
    )

    # Check (B,Q,T) sampling
    logits = torch.arange(24).view(4, 3, 2).float()
    samples = sampling.sample_stochastic(logits)
    assert samples.shape == (4, 2)

    # import matplotlib.pyplot as plt
    # plt.hist(samples.view(-1).int().numpy(), 4)
    # plt.legend()
    # plt.show()


def test_differentiable_sampler():
    torch.manual_seed(123)
    logits = torch.tensor([1.0, 2.0, 1.0, 1.0]).view(1, 4, 1).repeat(1, 1, 20000)
    print(F.log_softmax(torch.tensor([1.0, 2.0, 1.0, 1.0]), 0))
    samples = sampling.sample_differentiable(logits, tau=1.0)
    assert samples.shape == (1, 4, 20000)

    # import matplotlib.pyplot as plt
    # plt.hist(samples.argmax(1).view(-1).int().numpy(), 4)
    # plt.legend()
    # plt.show()

    assert _chi2test(
        samples.argmax(1).view(-1),
        torch.tensor([3500, 9400, 3500, 3500]) / 20000,
        p=0.01,
    )
