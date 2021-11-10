__all__ = ["generate", "generate_fast", "slice_generator", "rolling_origin"]
import itertools
import warnings
from typing import TYPE_CHECKING, Iterator, List, Tuple

import torch

from . import fast

if TYPE_CHECKING:
    from .wave import ObservationSampler, WaveNet

WaveGenerator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


def generate(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
) -> WaveGenerator:
    B, Q, T = initial_obs.shape
    if T < 1:
        raise ValueError("Need at least one observation to bootstrap generation.")

    # We need to track up to the last n samples,
    # where n equals the receptive field of the model
    R = model.receptive_field
    history = initial_obs.new_zeros((B, Q, R))
    t = min(R, T)
    history[..., :t] = initial_obs[..., -t:]

    while True:
        obs = history[..., :t]  # (B,Q,T)
        logits, _ = model.forward(obs)  # (B,Q,T)
        recent_logits = logits[..., -1:]  # yield sample for t+1 only (B,Q,1)
        s = sampler(recent_logits)
        yield s, recent_logits
        roll = int(t == R)
        history = history.roll(-roll, -1)  # no-op as long as history is not full
        t = min(t + 1, R)
        history[..., t - 1 : t] = s


def generate_fast(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
    layer_inputs: List[torch.Tensor] = None,
) -> WaveGenerator:
    B, _, T = initial_obs.shape
    if T < 1:
        raise ValueError("Need at least one observation to bootstrap.")
    # prepare queues
    if T == 1:
        queues = fast.create_zero_queues(
            model=model,
            device=initial_obs.device,
            dtype=initial_obs.dtype,
            batch_size=B,
        )
    else:
        if layer_inputs is None:
            _, layer_inputs, _, _ = model.encode(initial_obs[..., :-1])
            # TODO we should encode only necessary inputs
        else:
            layer_inputs = [inp[..., :-1] for inp in layer_inputs]
        queues = fast.create_initialized_queues(model=model, layer_inputs=layer_inputs)
    # generate
    obs = initial_obs[..., -1:]  # (B,Q,1)
    while True:
        logits, queues = model.forward(obs, queues)
        s = sampler(logits)  # (B,Q,1)
        yield s, logits
        obs = s


def slice_generator(
    gen: WaveGenerator,
    stop: int,
    step: int = 1,
    start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slices the given generator to get subsequent predictions and network outputs."""
    sl = itertools.islice(gen, start, stop, step)  # List[(sample,output)]
    samples, outputs = list(zip(*sl))
    return torch.cat(samples, -1), torch.cat(outputs, -1)


def rolling_origin(
    model: "WaveNet",
    sampler: "ObservationSampler",
    obs: torch.Tensor,
    horizon: int = 16,
    num_origins: int = None,
    random_origins: bool = False,
    skip_partial: bool = True,
):
    # See https://cran.r-project.org/web/packages/greybox/vignettes/ro.html
    (_, _, T), R = obs.shape, model.receptive_field
    if horizon == 1:
        warnings.warn(
            "Consider using wavnet.forward(), which performs a horizon 1 rolling origin more efficiently"
        )

    off = (R - 1) if skip_partial else 0
    roll_idx = torch.arange(off, T - horizon + 1, 1, device=obs.device)
    if num_origins is not None:
        if random_origins:
            ids = torch.ones(len(roll_idx)).multinomial(num_origins, replacement=False)
            roll_idx = roll_idx[ids]
        else:
            roll_idx = roll_idx[:num_origins]

    _, layer_inputs, _, _ = model.encode(obs)

    all_roll_samples = []
    all_roll_logits = []
    for ridx in roll_idx:
        roll_obs = obs[..., : (ridx + 1)]
        roll_inputs = [layer[..., : (ridx + 1)] for layer in layer_inputs]

        gen = generate_fast(
            model,
            roll_obs,
            sampler,
            layer_inputs=roll_inputs,
        )
        roll_samples, roll_logits = slice_generator(gen, horizon)
        all_roll_logits.append(roll_logits)
        all_roll_samples.append(roll_samples)
    return torch.stack(all_roll_samples, 0), torch.stack(all_roll_logits, 0), roll_idx
