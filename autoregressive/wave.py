import itertools
import logging
from typing import Iterator, List, Optional, Protocol, Tuple, Sequence, overload

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.distributions.constraints as constraints
import torch.nn
import torch.nn.functional as F
import torch.nn.init

from .utils import causal_pad

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)

FastQueues = List[torch.FloatTensor]
WaveGenerator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


@overload
def compute_receptive_field(dilation_seq: Sequence[int], kernel_size: int) -> int:
    ...


@overload
def compute_receptive_field(
    num_blocks: int, num_layers_per_block: int, kernel_size: int
) -> int:
    ...


def compute_receptive_field(
    dilation_seq: Sequence[int] = None,
    num_blocks: int = None,
    num_layers_per_block: int = None,
    kernel_size: int = 2,
) -> int:
    if dilation_seq is None:
        dilation_seq = [
            2 ** i for _ in range(num_blocks) for i in range(num_layers_per_block)
        ]
    return (kernel_size - 1) * sum(dilation_seq) + 1


class WaveNetLayer(torch.nn.Module):
    def __init__(self, dilation: int, wave_channels: int = 32):
        super().__init__()
        self.dilation = dilation
        self.conv_dilation = torch.nn.Conv1d(
            wave_channels,
            wave_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.conv_skip = torch.nn.Conv1d(
            wave_channels,
            wave_channels,
            kernel_size=1,
        )

    def forward(self, x, fast: bool = False):
        """
        When fast is enabled, this function assumes that x is composed
        of the last 'recurrent' input, that is `dilation` steps back and
        the current input.
        """
        d = 1 if fast else self.dilation
        x_dilated = F.conv1d(
            x,
            self.conv_dilation.weight,
            self.conv_dilation.bias,
            dilation=d,
        )
        N = x_dilated.shape[-1]
        x_filter = torch.tanh(x_dilated)
        x_gate = torch.sigmoid(x_dilated)
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        identity = x[..., -N:]
        return identity + skip, skip


class WaveNetHead(torch.nn.Module):
    def __init__(self, wave_channels: int, out_channels: int, use_skips: bool = True):
        super().__init__()
        self.use_skips = use_skips
        self.transform = torch.nn.Sequential(
            torch.nn.ReLU(),  # note, we perform non-lin first (i.e on sum of skips)
            torch.nn.Conv1d(wave_channels, wave_channels, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(wave_channels, out_channels, kernel_size=1),
        )

    def forward(self, encoded, layer_inputs, skips):
        if self.use_skips:
            # Note skips decrease temporarily due to dilated convs,
            # so we sum only the ones corresponding to not causal padded
            # results
            N = skips[-1].shape[-1]
            x = torch.stack([s[..., -N:] for s in skips], dim=0).sum(dim=0)
        else:
            x = encoded  # residual from last layer in last block
        return self.transform(x)


class WaveNetBase(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        wave_channels: int = 32,
        num_blocks: int = 1,
        num_layers_per_block: int = 7,
        use_skips: bool = True,
    ):
        super().__init__()
        self.input_conv = torch.nn.Conv1d(in_channels, wave_channels, kernel_size=1)
        self.wave_layers = torch.nn.ModuleList(
            [
                WaveNetLayer(
                    dilation=2 ** d,
                    wave_channels=wave_channels,
                )
                for _ in range(num_blocks)
                for d in range(num_layers_per_block)
            ]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wave_channels = wave_channels
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.receptive_field = compute_receptive_field(
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
            kernel_size=2,
        )
        self.head = WaveNetHead(
            wave_channels=wave_channels, out_channels=out_channels, use_skips=use_skips
        )
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")

    def encode(self, x):
        skips = []
        layer_inputs = []
        xcausal = causal_pad(x, 2, self.receptive_field - 1)
        x = self.input_conv(xcausal)
        for layer in self.wave_layers:
            layer_inputs.append(x)
            x, skip = layer(x, fast=False)
            skips.append(skip)
        return x, layer_inputs, skips

    def encode_one(self, x, queues: FastQueues):
        skips = []
        layer_inputs = []
        updated_queues = []
        x = self.input_conv(x)  # no padding here
        for layer, q in zip(self.wave_layers, queues):
            layer_inputs.append(x)
            x, qnew = self._next_input_from_queue(q, x)
            x, skip = layer(x, fast=True)
            updated_queues.append(qnew)
            skips.append(skip)
        return x, layer_inputs, skips, updated_queues

    def forward(self, x):
        encoded, layer_inputs, skips = self.encode(x)
        return self.head(encoded, layer_inputs, skips)

    def forward_one(self, x, queues: FastQueues):
        encoded, layer_inputs, skips, queues = self.encode_one(x, queues)
        return self.head(encoded, layer_inputs, skips), queues

    def _next_input_from_queue(self, q: torch.Tensor, x):
        h = q[..., 0:1]  # pop left (oldest)
        qout = q.roll(-1, -1)  # roll by one in left direction
        qout[..., -1:] = x  # push right (newest)
        x = torch.cat((h, x), -1)  # prepare input
        return x, qout

    def create_empty_queues(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
    ) -> FastQueues:
        queues = []
        for layer in zip(self.wave_layers):
            layer: WaveNetLayer
            q = torch.zeros(
                (batch_size, self.wave_channels, layer.dilation),
                dtype=dtype,
                device=device,
            )  # Populated with zeros will act like causal padding
            queues.append(q)
        return queues

    def create_initialized_queues(
        self, layer_inputs: List[torch.FloatTensor]
    ) -> FastQueues:
        queues = []
        for layer, layer_input in zip(self.wave_layers, layer_inputs):
            layer: WaveNetLayer
            q = layer_input[..., -layer.dilation :]
            queues.append(q)
        return queues


class ObservationSampler(Protocol):
    def __call__(
        self, model: WaveNetBase, obs: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        ...


def generate(
    model: WaveNetBase,
    initial_obs: torch.Tensor,
    sampler: ObservationSampler,
    detach_sample: bool = True,
) -> WaveGenerator:
    B, C, T = initial_obs.shape
    if T < 1:
        raise ValueError("Need at least one observation to bootstrap.")

    # We need to track up to the last n samples,
    # where n equals the receptive field of the model
    R = model.receptive_field
    history = initial_obs.new_empty((B, C, R))
    t = min(R, T)
    history[..., :t] = initial_obs[..., -t:]

    while True:
        obs = history[..., :t]
        x = model.forward(obs)
        s = sampler(model, obs, x[..., -1:])  # yield sample for t+1 only
        if detach_sample:
            s = s.detach()
        yield s, x[..., -1:]
        roll = int(t == R)
        history = history.roll(-roll, -1)  # no-op as long as history is not full
        t = min(t + 1, R)
        history[..., t - 1 : t] = s


def generate_fast(
    model: WaveNetBase,
    initial_obs: torch.Tensor,
    sampler: ObservationSampler,
    detach_sample: bool = True,
) -> WaveGenerator:
    B, C, T = initial_obs.shape
    R = model.receptive_field
    if T < 1:
        raise ValueError("Need at least one observation to bootstrap.")
    # prepare queues
    if T == 1:
        queues = model.create_empty_queues(
            device=initial_obs.device,
            dtype=initial_obs.dtype,
            batch_size=B,
        )
    else:
        _, layer_inputs, _ = model.encode(initial_obs[..., :-1])
        queues = model.create_initialized_queues(layer_inputs)
    # generate
    obs = initial_obs[..., -1:]
    while True:
        x, queues = model.forward_one(obs, queues)
        s = sampler(model, obs, x)
        if detach_sample:
            s = s.detach()
        yield s, x
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
