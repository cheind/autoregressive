from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .wave import WaveNet

from . import generators


def rolling_nstep(
    model: "WaveNet",
    sampler: generators.ObservationSampler,
    x: torch.Tensor,
    num_generate: int = 16,
    max_rolls: int = None,
    random_rolls: bool = True,
    skip_partial: bool = True,
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

    _, layer_inputs, _ = model.encode(x, remove_left_invalid=False)

    all_roll_samples = []
    all_roll_logits = []
    for ridx in roll_idx:
        roll_obs = x[..., : (ridx + 1)]
        roll_inputs = [
            layer[..., : p + (ridx + 1)]
            for layer, p in zip(layer_inputs, model.num_left_invalid)
        ]
        # Note, `remove_left_invalid=False` to be able to work with partially complete
        # input windows, each input layer is of different length. The length differences
        # stems from transformation of causally padded input through each layer. Since
        # Hence, observations until t, correspond to layer inputs until p+t, where p
        # is the number of left-invalid elements in the given input layer.

        gen = generators.generate_fast(
            model,
            roll_obs,
            sampler,
            layer_inputs=roll_inputs,
        )
        roll_samples, roll_logits = generators.slice_generator(gen, num_generate)
        all_roll_logits.append(roll_logits)
        all_roll_samples.append(roll_samples)
    return torch.stack(all_roll_samples, 0), torch.stack(all_roll_logits, 0), roll_idx


def rolling_nstep_ce(
    roll_logits: torch.Tensor, roll_idx: torch.Tensor, targets: torch.Tensor
) -> float:
    G = roll_logits.shape[-1]  # number of forecast steps
    sum_loss = 0.0
    num_pred = 0
    for logits, idx in zip(roll_logits, roll_idx):
        roll_targets = targets[..., idx : idx + G]
        ce = F.cross_entropy(logits, roll_targets, reduction="sum")
        sum_loss = sum_loss + ce
        num_pred = num_pred + roll_targets.numel()
    return sum_loss / num_pred
