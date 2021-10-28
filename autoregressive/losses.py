from typing import Optional
import torch
import torch.nn.functional as F

from . import wave


def rolling_nstep(
    model: wave.WaveNetBase,
    sampler: wave.ObservationSampler,
    x: torch.Tensor,
    y: torch.Tensor = None,
    num_forecast: int = 16,
    max_sequences: Optional[int] = None,
    detach_sample: bool = True,
) -> torch.FloatTensor:

    # If targets are not given, we create them from the observations
    if y is None:
        x = x[..., :-1]
        y = x[..., 1:]

    B, C, T = x.shape
    R = model.receptive_field

    # TODO we should not consider the first R inputs/outputs
    x = x[..., : T - num_forecast].unfold(-1, R, 1)  # B,C,W,R
    y = y[..., (R - 1) :].unfold(-1, num_forecast, 1)  # B,C,M,N
    W = min(x.shape[2], y.shape[2])  # sanity - believe its unnecessary though
    x = x[:, :, :W, :].permute(0, 2, 1, 3).reshape(-1, C, R)
    y = y[:, :, :W, :].permute(0, 2, 1, 3).reshape(-1, C, num_forecast)

    if max_sequences is not None:
        max_sequences = min(max_sequences, x.shape[0])
        # Stochastically draw a subset of rolling windows
        idx = torch.ones(x.shape[0]).multinomial(max_sequences, replacement=False)
        x = x[idx]
        y = y[idx]

    # print(x.shape)
    # print(y.shape)
    # print(x[0, 0])
    # print(y[0, 0])
    # print("-----")
    # print(x[1, 0])
    # print(y[1, 0])

    g = wave.generate_fast(model, x, sampler, detach_sample=detach_sample)
    samples, outputs = wave.slice_generator(g, stop=num_forecast)
    return x, y, samples, outputs


if __name__ == "__main__":
    B, C, T = 2, 1, 16
    model = wave.WaveNetBase(num_layers_per_block=3)
    print(model.receptive_field)
    torch.manual_seed(123)
    x = torch.rand(B, C, T)
    print(x)
    rolling_nstep_loss(
        model,
        lambda model, obs, x: x,
        x,
        num_forecast=5,
        num_draw=3,
        detach_sample=True,
    )
