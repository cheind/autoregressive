__all__ = [
    "generate",
    "generate_fast",
    "slice_generator",
    "rolling_origin",
    "collate_rolling_origin",
]
import itertools
import warnings
import abc
from typing import TYPE_CHECKING, Iterator, List, Tuple, Union
from functools import partial

import torch
from torch._C import device

from autoregressive import sampling

from . import fast, encoding

if TYPE_CHECKING:
    from .wave import WaveNet, WaveLayerBase
    from .sampling import ObservationSampler


# class GeneratorBase(abc.ABC):
#     def __init__(self, model: "WaveNet") -> None:
#         super().__init__()
#         self.model = model


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
        S = x.shape[-1]
        if S == 0:
            return
        N = min(self._T, S)
        self._buf = self._buf.roll(-N, -1)  # create space
        self._buf[..., -N:] = x[..., -N:]  # copy
        self._start = max(0, self._start - N)  # update start

    @property
    def buffer(self):
        if self._start > 0:
            return self._buf[..., self._start :]
        else:
            return self._buf


class Generator:
    def __init__(self, model: "WaveNet", x: torch.Tensor) -> None:
        self.model = model
        self.R = self.model.receptive_field
        self.Q = self.model.quantization_levels
        self.input_buffer = None
        self._setup(x)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ...

    def step(
        self, x: torch.Tensor, sampler: sampling.ObservationSampler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.input_buffer, "seed generator first"
        self._update(x)
        logits, _ = self.model.forward(self.input_buffer.buffer)
        logits = logits[..., -1:]
        sample = sampler(logits)
        return sample, logits

    def _update(self, x):
        x = encoding.one_hotf(x, self.Q)
        self.input_buffer.add(x)

    def _setup(self, x: torch.Tensor):
        B = x.shape[0]
        self.input_buffer = RecentBuffer(
            (B, self.Q, self.R), dtype=x.dtype, device=x.device
        )
        self._update(x)


WaveGenerator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


def generate(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
) -> WaveGenerator:
    g = Generator(model, initial_obs[..., :-1])
    nextx = initial_obs[..., -1:]
    with g:
        while True:
            sample, logits = g.step(nextx, sampler)
            yield sample, logits
            nextx = sample


class FastGenerator:
    def __init__(
        self, model: "WaveNet", x: Union[torch.Tensor, list[torch.Tensor]]
    ) -> None:
        self.model = model
        self.R = self.model.receptive_field
        self.Q = self.model.quantization_levels
        self.input_queues = None
        self._hooks_enabled = False  # make thread local?
        self._hook_handles = None
        self._setup_queues(x)

    def __enter__(self):
        self._setup_hooks()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._remove_hooks()

    def step(
        self, x: torch.Tensor, sampler: sampling.ObservationSampler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.input_queues, "enter context first"
        self._hooks_enabled = True  # TODO: context this
        logits, _ = self.model.forward(x, causal_pad=False)
        logits = logits[..., -1:]
        sample = sampler(logits)
        self._hooks_enabled = False
        return sample, logits

    def _setup_queues(self, x: Union[torch.Tensor, list[torch.Tensor]]):
        is_raw_input = isinstance(x, torch.Tensor)

        if is_raw_input:
            self.input_queues = self._make_queues(x.shape[0], x.dtype, x.device)
            if x.shape[-1] > 0:
                start = max(0, x.shape[-1] - self.R)
                _, layer_inputs, _, _ = self.model.encode(x[..., start:])
                self._update_queues(layer_inputs)
        else:
            self.input_queues = self._make_queues(
                x[0].shape[0], x[0].dtype, x[0].device
            )
            self._update_queues(x)

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
                # print(q.buffer.shape, x.shape, y.shape)
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

    def _make_queues(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> List[RecentBuffer]:
        queues = []
        for layer in self.model.layers:
            layer: "WaveLayerBase"
            q = RecentBuffer(
                (batch_size, layer.in_channels, layer.causal_left_pad),
                dtype=dtype,
                device=device,
                empty=False,
            )  # Populated with zeros will act like causal padding
            queues.append(q)
        return queues


def generate_fast(
    model: "WaveNet",
    initial_obs: torch.Tensor,
    sampler: "ObservationSampler",
    layer_inputs: List[torch.Tensor] = None,
) -> WaveGenerator:
    # In case we have compressed input, we convert to one-hot style.
    Q = model.quantization_levels
    R = model.receptive_field
    initial_obs = encoding.one_hotf(initial_obs, quantization_levels=Q)
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
        start = max(0, T - R)
        end = T - 1
        if layer_inputs is None:
            _, layer_inputs, _, _ = model.encode(initial_obs[..., start:end])
        else:
            layer_inputs = [inp[..., start:end] for inp in layer_inputs]
        # create queues
        queues = fast.create_initialized_queues(model=model, layer_inputs=layer_inputs)
    # start generating beginning with most recent observation
    obs = initial_obs[..., -1:]  # (B,Q,1)
    while True:
        logits, queues = model.forward(obs, queues)
        s = sampler(logits)  # (B,Q,1) or (B,Q)
        yield s, logits
        obs = encoding.one_hotf(s, quantization_levels=Q)  # (B,Q,1)


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
):
    # See https://cran.r-project.org/web/packages/greybox/vignettes/ro.html
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


def collate_rolling_origin(
    roll_logits: torch.Tensor, roll_idx: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # roll_logits: (R,B,Q,H)
    # roll_idx: (R,)
    # targets: (B,T)
    R, B, Q, H = roll_logits.shape
    logits = roll_logits.reshape(R * B, Q, H)  # (R*B,Q,H)
    targets = targets.unfold(-1, H, 1).permute(1, 0, 2)  # (W,B,H)
    targets = targets[roll_idx].reshape(R * B, H)  # (R*B,H)
    return logits, targets
