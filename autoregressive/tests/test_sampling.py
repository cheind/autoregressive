import torch
import numpy as np
import math

from .. import sampling


def _multinomial_prob(s: torch.Tensor, pi: torch.Tensor):
    # See https://en.wikipedia.org/wiki/Multinomial_test

    # approx for log(n!)
    log_nfac = lambda n: sum(np.log(np.arange(1, n + 1)))
    # log_nfac = (
    #     lambda n: n * np.log(n)
    #     - n
    #     + np.log(n * (1 + 4 * n * (1 + 2 * n))) / 6
    #     + np.log(math.pi) * 0.5
    # )

    N = s.shape[0]
    k = len(pi)
    # x = torch.histc(s.float(), bins=k, min=0, max=(k - 1)).int()
    x = (torch.ones(10) * 20).long().numpy()
    N = x.sum()
    assert x.sum() == N
    pi = pi / pi.sum()
    pi = pi.numpy().astype(float)
    print(pi, x)

    log_h0 = log_nfac(N) + sum(
        [(x_i * np.log(pi_i) - log_nfac(x_i)) for x_i, pi_i in zip(x, pi)]
    )
    print([(x_i * np.log(pi_i) - log_nfac(x_i)) for x_i, pi_i in zip(x, pi)])
    print(log_nfac(N))
    # print([(log_nfac(x_i), x_i) for x_i in x])
    return np.exp(log_h0)
    # see https://en.wikipedia.org/wiki/Multinomial_test#cite_note-Read-Cressie-1988-1


def test_greedy_sampler():
    torch.manual_seed(123)
    logits = torch.rand(2, 10, 10000)
    # logits[:, :2, :] *= 1.2
    samples = sampling.GreedySampler()(logits)
    # assert samples.shape == (2, 500)
    import matplotlib.pyplot as plt

    flat = samples.view(-1)

    print(_multinomial_prob(flat, torch.ones(10)))

    # plt.hist(flat.int().numpy(), 10, density=True)
    # plt.show()