import itertools
import logging
from typing import Iterator, List, Protocol, Tuple, Sequence
import dataclasses

import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.init

from . import losses

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)

FastQueues = List[torch.FloatTensor]
WaveGenerator = Iterator[Tuple[torch.Tensor, torch.Tensor]]


def causal_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """Performs a cause padding to avoid data leakage. Stride is assumed to be one."""
    left_pad = (kernel_size - 1) * dilation
    return F.pad(x, (left_pad, 0))


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


def compute_receptive_field(
    dilation_seq: Sequence[int] = None,
    kernel_size: int = 2,
) -> int:
    return (kernel_size - 1) * sum(dilation_seq) + 1


class WaveNetLayer(torch.nn.Module):
    def __init__(self, dilation: int, wave_channels: int = 32):
        super().__init__()
        self.dilation = dilation
        self.wave_channels = wave_channels
        self.conv_dilation = torch.nn.Conv1d(
            wave_channels,
            2 * wave_channels,  # See PixelCNN, we stack W f,k and W g,k
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
        T = x_dilated.shape[-1]

        x_filter = torch.tanh(x_dilated[:, : self.wave_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.wave_channels :])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        out = x[..., -T:] + skip  # trim-off causal padded results
        return out, skip


class WaveNetLogitsHead(torch.nn.Module):
    def __init__(self, wave_channels: int, out_channels: int):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.ReLU(),  # note, we perform non-lin first (i.e on sum of skips)
            torch.nn.Conv1d(
                wave_channels, wave_channels * 2, kernel_size=1
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.ReLU(),
            torch.nn.Conv1d(wave_channels * 2, out_channels, kernel_size=1),  # logits
        )

    def forward(self, encoded, skips):
        # Note skips decrease temporarily due to dilated convs,
        # so we sum only the ones corresponding to none causally padded
        # results
        N = skips[-1].shape[-1]
        trimmed_skips = [s[..., -N:] for s in skips]
        return self.transform(torch.stack(trimmed_skips, dim=0).sum(dim=0))


@dataclasses.dataclass
class WaveNetTrainOpts:
    skip_partial_receptive_field: bool = True
    lr: float = 1e-3
    sched_patience: int = 25
    val_unroll_steps: int = 32
    val_max_rolls: int = 8


class WaveNet(pl.LightningModule):
    def __init__(
        self,
        dilations: Sequence[int] = tuple([2 ** i for i in range(8)] * 2),
        quantization_levels: int = 2 ** 8,
        wave_channels: int = 32,
        train_opts: WaveNetTrainOpts = WaveNetTrainOpts(),
    ):
        super().__init__()
        self.input_conv = torch.nn.Conv1d(
            quantization_levels, wave_channels, kernel_size=1
        )
        self.wave_layers = torch.nn.ModuleList(
            [
                WaveNetLayer(
                    dilation=d,
                    wave_channels=wave_channels,
                )
                for d in dilations
            ]
        )
        self.dilations = dilations
        self.quantization_channels = quantization_levels
        self.wave_channels = wave_channels
        self.receptive_field = compute_receptive_field(dilations, kernel_size=2)
        self.logits = WaveNetLogitsHead(
            wave_channels=wave_channels,
            out_channels=quantization_levels,
        )
        self.train_opts = train_opts
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")
        self.save_hyperparameters()

    def encode(self, x, strip_padding: bool = True):
        skips = []
        layer_inputs = []
        x = causal_pad(x, 2, self.receptive_field - 1)
        x = self.input_conv(x)
        for layer in self.wave_layers:
            layer_inputs.append(x)
            x, skip = layer(x, fast=False)
            skips.append(skip)
        if strip_padding:
            layer_inputs = self._strip_layer_input_padding(layer_inputs)
        return x, layer_inputs, skips

    def encode_one(self, x, queues: FastQueues):
        skips = []
        updated_queues = []
        x = self.input_conv(x)  # no padding here
        for layer, q in zip(self.wave_layers, queues):
            x, qnew = self._next_input_from_queue(q, x)
            x, skip = layer(x, fast=True)
            updated_queues.append(qnew)
            skips.append(skip)
        return x, skips, updated_queues

    def forward(self, x, is_one_hot: bool = False):
        if not is_one_hot:
            # (B,T) -> (B,Q,T)
            x = F.one_hot(x, num_classes=self.quantization_channels)
            x = x.permute(0, 2, 1)  # (B,Q,T)
        encoded, _, skips = self.encode(x)
        return self.logits(encoded, skips)

    def forward_one(self, x, queues: FastQueues, is_one_hot: bool = False):
        if not is_one_hot:
            # (B,1) -> (B,Q,1)
            x = F.one_hot(x, num_classes=self.quantization_channels)
            x = x.permute(0, 2, 1)
        encoded, skips, queues = self.encode_one(x, queues)
        return self.logits(encoded, skips), queues

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
        for layer in self.wave_layers:
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
            assert (
                layer_input.shape[-1] >= layer.dilation
            )  # you might have stripped layer inputs
            q = layer_input[..., -layer.dilation :]
            queues.append(q)
        return queues

    def _strip_layer_input_padding(self, layer_inputs: torch.Tensor) -> torch.Tensor:
        # Note, layer inputs contain results from causal padding
        # at front. This decreases over layers based on dilation factors.
        # Hence when computing layer-inputs
        result = []
        p = self.receptive_field - 1
        for inp, layer in zip(layer_inputs, self.wave_layers):
            layer: WaveNetLayer
            result.append(inp[..., p:])
            p -= layer.dilation
        return result

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.5,
                    patience=self.sched_patience,
                    min_lr=5e-5,
                    threshold=1e-7,
                ),
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }

    def training_step(self, batch, batch_idx):
        targets: torch.Tensor = batch["b"][..., 1:]
        inputs: torch.Tensor = batch["b"][..., :-1]
        inputs = F.one_hot(inputs, num_classes=self.quantization_channels)  # (B,T,Q)
        inputs = inputs.permute(0, 2, 1)  # (B,Q,T)
        logits = self(inputs)

        r = self.receptive_field if self.train_opts.skip_partial_receptive_field else 0
        loss = F.cross_entropy(logits[..., r:], targets[..., r:])

        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        targets: torch.Tensor = batch["b"][..., 1:]
        inputs: torch.Tensor = batch["b"][..., :-1]

        roll_y, _, roll_idx = losses.rolling_nstep(
            self,
            self.create_sampler(),
            inputs,
            num_generate=self.train_opts.val_unroll_steps,
            max_rolls=self.train_opts.val_max_rolls,
            random_rolls=False,
            skip_partial=self.train_opts.skip_partial_receptive_field,
        )
        loss = losses.rolling_nstep_ce(roll_y, roll_idx, targets)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)

    def create_sampler(self, greedy: bool = False):
        def greedy_sampler(logits):
            return torch.argmax(logits, dim=1, keepdim=True)  # (B,1)

        def stochastic_sampler(logits):
            bin = D.Categorical(logits=logits.permute(0, 2, 1)).sample()  # (B,1)
            return bin

        if greedy:
            return greedy_sampler
        else:
            return stochastic_sampler


class ObservationSampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        ...


def generate(
    model: WaveNet,
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
    history = initial_obs.new_zeros(
        (B, C, R)
    )  # TODO C=Q, should maybe not be part of sampler, except if one_hot is True
    t = min(R, T)
    history[..., :t] = initial_obs[..., -t:]

    while True:
        obs = history[..., :t]
        x = model.forward(obs)
        s = sampler(model, obs, x[..., -1:])  # yield sample for t+1 only
        yield s, x[..., -1:]
        if detach_sample:
            s = s.detach()
        roll = int(t == R)
        history = history.roll(-roll, -1)  # no-op as long as history is not full
        t = min(t + 1, R)
        history[..., t - 1 : t] = s


def generate_fast(
    model: WaveNet,
    initial_obs: torch.Tensor,
    sampler: ObservationSampler,
    detach_sample: bool = True,
    layer_inputs: List[torch.Tensor] = None,
) -> WaveGenerator:
    B, _, T = initial_obs.shape
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
        if layer_inputs is None:
            _, layer_inputs, _ = model.encode(
                initial_obs[..., :-1], strip_padding=False
            )  # TODO we should encode only necessary inputs
        else:
            layer_inputs = [inp[..., :-1] for inp in layer_inputs]
        queues = model.create_initialized_queues(layer_inputs)
    # generate
    obs = initial_obs[..., -1:]
    while True:
        x, queues = model.forward_one(obs, queues)
        s = sampler(model, obs, x)
        yield s, x
        if detach_sample:
            s = s.detach()
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
