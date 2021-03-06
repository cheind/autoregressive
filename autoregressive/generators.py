__all__ = [
    "Generator",
    "FastGenerator",
    "generate",
    "generate_fast",
    "slice_generator",
    "rolling_origin",
    "collate_rolling_origin",
]
import itertools
import warnings
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Iterator, List, Tuple, Union

import torch

from . import encoding, sampling

if TYPE_CHECKING:
    from .sampling import ObservationSampler
    from .wave import WaveLayerBase, WaveNet

"""WaveGenerator

A WaveGenerator allows endless generation of WaveNet sequences.
"""
WaveGenerator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


class RecentBuffer:
    """A simple deque (with max-size) implemented using a torch.Tensor"""

    def __init__(
        self,
        shape: torch.Size,
        dtype: torch.dtype = None,
        device: torch.device = None,
        empty: bool = True,
    ):
        self._buf = torch.zeros(shape, dtype=dtype, device=device)  # (B,Q,T)
        self._T = self._buf.shape[-1]
        self._start = self._T if empty else 0

    def add(self, x: torch.Tensor):
        """Append a tensor to the buffer."""
        S = x.shape[-1]
        N = min(self._T, S)
        # Both input and queue size can of zero size (temporal). For queues,
        # consider the queue for the input layer and a kernel size of 1.
        # When zero, do nothing. Infact the logic below does unintended things
        # when N==0.
        if N == 0:
            return
        self._buf = self._buf.roll(-N, -1)  # create space
        self._buf[..., -N:] = x[..., -N:]  # copy
        self._start = max(0, self._start - N)  # update start

    @property
    def buffer(self):
        """Access the current state of the buffer"""
        if self._start > 0:
            return self._buf[..., self._start :]
        else:
            return self._buf


class Generator:
    """Default WaveNet generator.

    A WaveNet generator incrementally predicts next sample distributions p(X_{i+1}|X_{0..i}).
    The default generator, requires R (receptive field) length input samples for each such predictions,
    which makes it rather slow.

    See `generate` below for usage.
    """

    def __init__(self, model: "WaveNet", batch_size: int, device: torch.device) -> None:
        self.model = model
        self.R = self.model.receptive_field
        self.Q = self.model.quantization_levels
        self.C = self.model.conditioning_channels
        self._setup(batch_size, device)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ...

    def step(
        self,
        x: torch.Tensor,
        sampler: sampling.ObservationSampler,
        c: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.input_buffer, "seed generator first"
        self.push(x, c)
        logits, _ = self.model.forward(
            self.input_buffer.buffer,
            c=self.cond_buffer.buffer if self.cond_buffer else None,
        )
        logits = logits[..., -1:]
        sample = sampler(logits)
        return sample, logits

    def push(self, x: torch.Tensor, c: torch.Tensor = None):
        x = encoding.one_hotf(x, self.Q)
        self.input_buffer.add(x)
        if self.cond_buffer:
            assert c is not None, "conditioning required"
            if c.shape[-1] == 1:
                # convert global condition to local
                c = c.repeat(1, 1, x.shape[-1])
            self.cond_buffer.add(c)

    def _setup(self, batch_size: int, device: torch.device):
        self.input_buffer = RecentBuffer(
            (batch_size, self.Q, self.R), dtype=torch.float32, device=device
        )
        self.cond_buffer = None
        if self.C:
            self.cond_buffer = RecentBuffer(
                (batch_size, self.C, self.R), dtype=torch.float32, device=device
            )


class FastGenerator:
    """Fast WaveNet generator.

    A WaveNet generator incrementally predicts next sample distributions p(X_{i+1}|X_{0..i}).
    Compared to the default generator, this generator stores previously computed intermediate results
    to speed up prediction.

    See `generate_fast` for example usage.
    """

    # note, not thread-safe. modifies global state of model using hooks.
    def __init__(self, model: "WaveNet", batch_size: int, device: torch.device) -> None:
        self.model = model
        self.R = self.model.receptive_field
        self.Q = self.model.quantization_levels
        self._hooks_enabled = False  # make thread local?
        self._hook_handles = None
        self.input_queues = None
        self._setup_queues(batch_size, device)

    def __enter__(self):
        self._setup_hooks()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._remove_hooks()

    def step(
        self,
        x: torch.Tensor,
        sampler: sampling.ObservationSampler,
        c: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._hook_handles, "enter generator context first"
        with self._enable_hooks():
            logits, _ = self.model.forward(x, c=c, causal_pad=False)
        logits = logits[..., -1:]
        sample = sampler(logits)
        return sample, logits

    def push(self, x: Union[torch.Tensor, list[torch.Tensor]], c: torch.Tensor = None):
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            if x.shape[-1] > 0:
                start = max(0, x.shape[-1] - self.R)
                _, layer_inputs, _ = self.model.encode(x[..., start:], c=c)
                self._update_queues(layer_inputs)
        else:
            layer_inputs = x
            self._update_queues(layer_inputs)

    def _setup_queues(self, batch_size: int, device: torch.device):
        queues = []
        for layer in self.model.layers:
            layer: "WaveLayerBase"
            q = RecentBuffer(
                (batch_size, layer.in_channels, layer.causal_left_pad),
                dtype=torch.float32,
                device=device,
                empty=False,
            )  # Populated with zeros will act like causal padding
            queues.append(q)
        self.input_queues = queues

    def _update_queues(self, layer_inputs: list[torch.Tensor]):
        for linp, q in zip(layer_inputs, self.input_queues):
            q.add(linp)

    def _setup_hooks(self):
        def prepare_input_pre_hook(module, input, q: RecentBuffer):
            del module
            if self._hooks_enabled:
                input = list(input)
                x = input[0]
                y = torch.cat((q.buffer, x), -1)
                q.add(x)
                input[0] = y
                input = tuple(input)
            return input

        hooks = [
            layer.register_forward_pre_hook(partial(prepare_input_pre_hook, q=q))
            for layer, q in zip(self.model.layers, self.input_queues)
        ]
        self._hook_handles = hooks

    def _remove_hooks(self):
        if self._hook_handles is not None:
            for h in self._hook_handles:
                h.remove()
        self._hook_handles = None

    @contextmanager
    def _enable_hooks(self):
        try:
            self._hooks_enabled = True
            yield
        finally:
            self._hooks_enabled = False


def generate(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
    global_cond: torch.Tensor = None,
) -> WaveGenerator:
    """Generates samples from previous observations.

    This method uses the standard generator, which requires R observations to make
    the next sample distribution prediction.

    Args:
        model: WaveNet model to use
        initial_obs: (B,Q,T) or (B,T) tensor containing past observations
        sampler: function to sample from (B,Q,T)
        global_cond: optional global condition to use during generation.

    Returns
        g: WaveGenerator
    """
    g = Generator(model, initial_obs.shape[0], initial_obs.device)
    g.push(initial_obs[..., :-1], c=global_cond)
    nextx = initial_obs[..., -1:]
    with g:
        while True:
            sample, logits = g.step(nextx, sampler, c=global_cond)
            yield sample, logits
            nextx = sample


def generate_fast(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
    layer_inputs: List[torch.Tensor] = None,
    global_cond: torch.Tensor = None,
) -> WaveGenerator:
    """Generates new samples from previous observations.

    This method uses the fast generator, which tracks intermediate results to
    speed up generation of new samples.

    Args:
        model: WaveNet model to use
        initial_obs: (B,Q,T) or (B,T) tensor containing past observations
        sampler: function to sample from (B,Q,T)
        layer_inputs: optional List[Tensor] corresponding to model layer-inputs of initial_obs
        global_cond: optional global condition to use during generation.
    Returns
        g: WaveGenerator
    """

    g = FastGenerator(model, initial_obs.shape[0], initial_obs.device)
    if layer_inputs is not None:
        g.push([layer[..., :-1] for layer in layer_inputs], c=global_cond)
    else:
        g.push(initial_obs[..., :-1], c=global_cond)
    nextx = initial_obs[..., -1:]
    with g:
        while True:
            sample, logits = g.step(nextx, sampler, c=global_cond)
            yield sample, logits
            nextx = sample


def slice_generator(
    gen: WaveGenerator,
    stop: int,
    step: int = 1,
    start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    global_cond: torch.Tensor = None,
):
    """Performs a rolling origin operation.
    See https://cran.r-project.org/web/packages/greybox/vignettes/ro.html for details.

    Args:
        model: WaveNet model
        sampler: function to sample from (B,Q,1) logits
        obs: (B,Q,T) known observations for rolling origin inputs
        horizon: number of samples to generate per origin
        num_origins: number of origins O
        random_origins: select origins randomly
        skip_partial: Only select origins for positions where R past observations are available.
        global_cond: Optional global condition to apply

    Returns:
        logits: (O,B,Q,H) tensor of rolling origin logits
        roll_idx: (O,) tensor of origin indices
        targets: (B,T) Target values
    """
    T, R = obs.shape[-1], model.receptive_field
    if horizon == 1:
        warnings.warn(
            "Consider using wavnet.forward(), which performs a horizon 1 rolling origin more efficiently"  # noqa: E501
        )

    off = (R - 1) if skip_partial else 0
    roll_idx = torch.arange(off, T - horizon + 1, 1, device=obs.device)
    if num_origins is not None:
        if random_origins:
            ids = torch.ones(len(roll_idx)).multinomial(num_origins, replacement=False)
            roll_idx = roll_idx[ids]
        else:
            roll_idx = roll_idx[:num_origins]

    _, layer_inputs, _ = model.encode(obs, c=global_cond)

    all_roll_samples = []
    all_roll_logits = []
    for ridx in roll_idx:
        roll_obs = obs[..., : (ridx + 1)]
        roll_inputs = [layer[..., : (ridx + 1)] for layer in layer_inputs]

        gen = generate_fast(
            model, roll_obs, sampler, layer_inputs=roll_inputs, global_cond=global_cond
        )
        roll_samples, roll_logits = slice_generator(gen, horizon)
        all_roll_logits.append(roll_logits)
        all_roll_samples.append(roll_samples)
    return torch.stack(all_roll_samples, 0), torch.stack(all_roll_logits, 0), roll_idx


def collate_rolling_origin(
    roll_logits: torch.Tensor, roll_idx: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collates the results of a rolling origin operation for cross entropy losses.

    Args:
        logits: (O,B,Q,H) tensor of rolling origin logits
        roll_idx: (O,) tensor of origin indices
        targets: (B,T) Target values

    Returns:
        logits: (O*B,Q,H) tensor
        targets: (O*B,H) tensor
    """
    # roll_logits: (R,B,Q,H)
    # roll_idx: (R,)
    # targets: (B,T)
    R, B, Q, H = roll_logits.shape
    logits = roll_logits.reshape(R * B, Q, H)  # (R*B,Q,H)
    targets = targets.unfold(-1, H, 1).permute(1, 0, 2)  # (W,B,H)
    targets = targets[roll_idx].reshape(R * B, H)  # (R*B,H)
    return logits, targets
