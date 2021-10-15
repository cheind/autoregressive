import numpy as np
import torch

PI = float(np.pi)


def fseries_amp_phase(
    bias: torch.FloatTensor,
    n: torch.IntTensor,
    a: torch.FloatTensor,
    phase: torch.FloatTensor,
    period: torch.FloatTensor,
    t: torch.FloatTensor,
):
    """Computes the Fourier series from amplitude-phase parametrization.

    This function supports batching, so that multiple series can
    be evaluated in parallel.

    Params
    ------
    bias: (B,) or (1) tensor
        Bias term(s) aka DC values
    n: (B,N) or (N,) or tensor
        component values of the series
    a: (B,N) or (N,) tensor
        coefficient of components
    phase: (B,N) or (B,) or (1) tensor
        phase for component term
    period: (B,) or (1) tensor
        period for each of the curves, such that f(t) == f(t+T)
    t: (B,T) or (T,) tensor
        sample times for each of the curves

    Returns
    -------
    y: (B,T) tensor or (1,T)
        function values for each of the fourier series results
    """
    # https://www.seas.upenn.edu/~kassam/tcom370/n99_2B.pdf
    t = torch.atleast_2d(t).unsqueeze(1)  # (B,1,T)
    period = torch.atleast_1d(period).view(-1, 1, 1)  # (B,1,1)
    n = torch.atleast_2d(n).unsqueeze(-1)  # (B,N,1)
    phase = torch.atleast_2d(phase).unsqueeze(-1)  # (B,N,1)
    bias = torch.atleast_1d(bias).unsqueeze(-1)  # (B,1)
    a = torch.atleast_2d(a).unsqueeze(1)  # (B,1,N)

    # print(t.shape, n.shape, phase.shape, period.shape)

    f0 = 1 / period
    arg = 2 * PI * f0 * n * t + phase  # (B,N,T)
    return bias * 0.5 + (a @ torch.cos(arg)).squeeze(1)


def square_wave():
    # https://mathworld.wolfram.com/FourierSeriesSquareWave.html
    import matplotlib.pyplot as plt

    T = 10.0
    num_terms = 20
    num_samples = 1000
    n = torch.arange(1, 2 * num_terms, step=2)
    coeffs = 4.0 / (PI * n)
    # We generate multiple (B=4) approximations with increasing frequencies enabled.
    coeffs = coeffs.view(1, -1).repeat(4, 1)
    coeffs[0, 5:] = 0.0  # For the first curve, we disable all but the first 5 terms
    coeffs[1, 10:] = 0.0
    coeffs[2, 15:] = 0.0
    phase = torch.tensor(-PI / 2)  # We share phase angles (sin(phi) = cos(phi-pi/2))
    bias = torch.tensor(0.0)  # We don't have a bias aka DC
    t = torch.linspace(
        0, T, num_samples
    )  # We sample all curves at the same time intervals

    y = fseries_amp_phase(
        bias=bias,
        n=n,
        a=coeffs,
        phase=phase,
        period=torch.tensor(T),
        t=t,
    )
    yhat = torch.zeros(len(t))
    yhat[: num_samples // 2] = 1.0
    yhat[num_samples // 2 :] = -1.0
    yhat[0] = 0.0
    yhat[-1] = 0.0

    plt.title("Approximations of a step function")
    plt.plot(t, yhat, c="k")
    plt.plot(t, y[0], label="5 terms")
    plt.plot(t, y[1], label="10 terms")
    plt.plot(t, y[2], label="15 terms")
    plt.plot(t, y[3], label="20 terms")
    plt.legend()
    plt.show()


def random_waves():
    import matplotlib.pyplot as plt

    N = 5
    n = torch.arange(N)
    coeffs = torch.rand(4, N)
    phase = torch.rand(4, N)
    bias = torch.rand(4)
    period = torch.rand(4) * 5
    t = torch.stack(
        [
            torch.linspace(0, 1, 1000),
            torch.linspace(0.5, 1.5, 1000),
            torch.linspace(1.5, 2.5, 1000),
            torch.linspace(2.5, 3.5, 1000),
        ],
        0,
    )
    y = fseries_amp_phase(
        bias=bias,
        n=n,
        a=coeffs,
        phase=phase,
        period=period,
        t=t,
    )
    plt.title("Random waves sampled at random times")
    plt.plot(t[0], y[0])
    plt.plot(t[1], y[1])
    plt.plot(t[2], y[2])
    plt.plot(t[3], y[3])
    plt.show()


def main():
    square_wave()
    random_waves()


if __name__ == "__main__":
    main()
