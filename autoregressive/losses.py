import torch
import torch.nn.functional as F

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
    (_, _, T), R = x.shape, model.receptive_field

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


def rolling_nstep_mae(
    roll_y: torch.Tensor, roll_idx: torch.Tensor, y: torch.Tensor, margin: float = 0.0
) -> float:
    G = roll_y.shape[-1]  # number of elements predicted in roll
    sum_loss = 0.0
    num_pred = 0
    for yhat, idx in zip(roll_y, roll_idx):
        ae = F.l1_loss(yhat, y[..., idx : idx + G], reduction="none")
        th = torch.maximum(ae.new_zeros(1).expand_as(ae), ae - margin)
        sum_loss = sum_loss + th.sum()
        num_pred = num_pred + yhat.numel()
    return sum_loss / num_pred