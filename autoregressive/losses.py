import torch

from . import wave


def rolling_nstep(
    model: wave.WaveNetBase,
    sampler: wave.ObservationSampler,
    x: torch.Tensor,
    num_generate: int = 16,
    max_rolls: int = None,
    random_rolls: bool = True,
    skip_partial: bool = True,
    detach_sample: bool = True,
):
    (B, C, T), R = x.shape, model.receptive_field

    off = (R - 1) if skip_partial else 0
    roll_idx = torch.arange(off, T - num_generate + 1, 1, device=x.device)
    if max_rolls is not None:
        if random_rolls:
            ids = torch.ones(len(roll_idx)).multinomial(max_rolls, replacement=False)
            roll_idx = roll_idx[ids]
        else:
            roll_idx = roll_idx[:max_rolls]

    _, layer_inputs, _ = model.encode(x)

    rolls_yhat = []
    rolls_out = []
    for ridx in roll_idx:
        roll_obs = x[..., : (ridx + 1)]
        roll_inputs = [layer[..., : (ridx + 1)] for layer in layer_inputs]
        gen = wave.generate_fast(
            model,
            roll_obs,
            sampler,
            detach_sample=detach_sample,
            layer_inputs=roll_inputs,
        )
        yhat, out = wave.slice_generator(gen, num_generate)
        rolls_yhat.append(yhat)
        rolls_out.append(out)
    return torch.stack(rolls_yhat, 0), torch.stack(rolls_out, 0), roll_idx


if __name__ == "__main__":

    model = wave.WaveNetBase(1, 1, 8, 1, 3)
    print(model.receptive_field)

    torch.manual_seed(123)
    seq = torch.rand(1, 1, 16)
    x = seq[..., :-1]
    y = seq[..., 1:]

    print(x)
    print(model(x))
    yhat, yout, starts = rolling_nstep(
        model,
        lambda model, obs, x: x,
        x,
        num_generate=4,
        detach_sample=True,
    )
    print(yhat.shape, yout.shape, starts.shape)
    print(starts)
    print(yhat[0, 0, 0])
    print(yhat[1, 0, 0])
    print(yhat[2, 0, 0])
