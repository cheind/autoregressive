import dataclasses
import logging
from typing import Iterable, Protocol

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn
import torch.nn.functional as F
import torch.nn.init

from . import fast, generators, sampling, compression

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=1e-3)


def compute_receptive_field(
    dilations: Iterable[int],
    kernel_sizes: Iterable[int],
) -> int:
    return sum([(k - 1) * d for k, d in zip(kernel_sizes, dilations)]) + 1


class WaveLayerBase(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        in_channels: int,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_left_pad = (kernel_size - 1) * dilation
        self.temporal_queue_size = self.causal_left_pad
        self.in_channels = in_channels


class WaveNetLayer(WaveLayerBase):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        wave_channels: int = 32,
    ):
        super().__init__(
            kernel_size=kernel_size,
            dilation=dilation,
            in_channels=wave_channels,
        )
        self.wave_channels = wave_channels
        self.conv_dilation = torch.nn.Conv1d(
            wave_channels,
            2 * wave_channels,  # See PixelCNN, we stack W f,k and W g,k
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv_skip = torch.nn.Conv1d(
            wave_channels,
            wave_channels,
            kernel_size=1,
        )

    def forward(self, x, h: torch.Tensor = None):
        """
        When fast is enabled, this function assumes that x is composed
        of the last 'recurrent' inputs, that is `dilation*(kernel-1)` steps back and
        the current input.
        """
        if h is not None:
            assert h.shape[-1] == (self.kernel_size - 1)
            cc = torch.cat((h, x), -1)
            x_dilated = F.conv1d(
                cc,
                self.conv_dilation.weight,
                self.conv_dilation.bias,
                dilation=1,
            )
        else:
            x_dilated = self.conv_dilation(F.pad(x, (self.causal_left_pad, 0)))
        x_filter = torch.tanh(x_dilated[:, : self.wave_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.wave_channels :])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        out = x + skip
        return out, skip


class WaveNetInputLayer(WaveLayerBase):
    def __init__(
        self,
        quantization_levels: int,
        kernel_size: int = 1,
        wave_channels: int = 32,
    ):
        super().__init__(
            kernel_size=kernel_size, dilation=1, in_channels=quantization_levels
        )
        self.wave_channels = wave_channels
        self.input_channels = quantization_levels
        self.conv = torch.nn.Conv1d(
            quantization_levels,
            wave_channels,
            kernel_size=kernel_size,
            dilation=self.dilation,
        )

    def forward(self, x, h: torch.Tensor = None):
        if h is not None:
            assert h.shape[-1] == (self.kernel_size - 1)
            x = torch.cat((h, x), -1)
        else:
            x = F.pad(x, (self.causal_left_pad, 0))
        x = self.conv(x)
        return x, x.new_zeros(*x.shape)  # zeros are not changing head


class WaveNetLogitsHead(WaveLayerBase):
    def __init__(self, wave_channels: int, out_channels: int):
        super().__init__(kernel_size=1, dilation=1, in_channels=wave_channels)
        self.transform = torch.nn.Sequential(
            torch.nn.ReLU(),  # note, we perform non-lin first (i.e on sum of skips)
            torch.nn.Conv1d(
                wave_channels, wave_channels * 2, kernel_size=1
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.ReLU(),
            torch.nn.Conv1d(wave_channels * 2, out_channels, kernel_size=1),  # logits
        )

    def forward(self, encoded, skips):
        del encoded
        skips = skips[1:]  # remove input-layer dummy skip
        return self.transform(torch.stack(skips, dim=0).sum(dim=0))


@dataclasses.dataclass
class WaveNetTrainOpts:
    skip_partial: bool = True
    lr: float = 1e-3
    sched_patience: int = 25
    val_ro_horizon: int = 32
    val_ro_num_origins: int = 8


class WaveNet(pl.LightningModule):
    def __init__(
        self,
        quantization_levels: int = 2 ** 8,
        wave_dilations: list[int] = [2 ** i for i in range(8)] * 2,
        wave_kernel_size: int = 2,
        wave_channels: int = 32,
        input_kernel_size: int = 1,
        train_opts: WaveNetTrainOpts = WaveNetTrainOpts(),
    ):
        super().__init__()
        layers = [
            WaveNetInputLayer(
                kernel_size=input_kernel_size,
                quantization_levels=quantization_levels,
                wave_channels=wave_channels,
            )
        ]
        layers += [
            WaveNetLayer(
                kernel_size=wave_kernel_size,
                dilation=d,
                wave_channels=wave_channels,
            )
            for d in wave_dilations
        ]
        self.layers = torch.nn.ModuleList(layers)
        self.logits = WaveNetLogitsHead(
            wave_channels=wave_channels,
            out_channels=quantization_levels,
        )
        self.quantization_levels = quantization_levels
        self.train_opts = train_opts
        self.receptive_field = compute_receptive_field(
            kernel_sizes=[layer.kernel_size for layer in self.layers],
            dilations=[layer.dilation for layer in self.layers],
        )
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")
        self.save_hyperparameters()

    def encode(self, x, queues: fast.FastQueues = None):
        x = compression.to_one_hot(x, self.quantization_levels)
        if queues is None:
            queues = [None] * len(self.layers)

        skips = []
        layer_inputs = []
        out_queues = []

        for layer, q in zip(self.layers, queues):
            layer: WaveLayerBase
            layer_inputs.append(x)
            h = None
            if q is not None:
                h = fast.read_queue(q, layer.kernel_size - 1, layer.dilation)
                q = fast.push_queue(q, x)
            out_queues.append(q)
            x, skip = layer(x, h=h)
            skips.append(skip)

        if queues[0] is None:
            out_queues = None

        return x, layer_inputs, skips, out_queues

    def forward(self, x, queues: fast.FastQueues = None):
        # x (B,Q,T) or (B,T)
        encoded, _, skips, outqueues = self.encode(x, queues=queues)
        return self.logits(encoded, skips), outqueues

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=self.train_opts.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.5,
                    patience=self.train_opts.sched_patience,
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
        targets: torch.Tensor = batch["x_k"][..., 1:]  # (B,T)
        inputs: torch.Tensor = batch["x_k"][..., :-1]  # (B,T)
        logits, _ = self(inputs)

        r = self.receptive_field if self.train_opts.skip_partial else 0
        loss = F.cross_entropy(logits[..., r:], targets[..., r:])

        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        inputs: torch.Tensor = batch["x_k"][..., :-1]
        targets: torch.Tensor = batch["x_k"][..., 1:]

        if self.train_opts.val_ro_horizon > 1:
            _, roll_logits, roll_idx = generators.rolling_origin(
                self,
                sampler=sampling.GreedySampler(),
                obs=inputs,
                horizon=self.train_opts.val_ro_horizon,
                num_origins=self.train_opts.val_ro_num_origins,
                random_origins=True,
                skip_partial=self.train_opts.skip_partial,
            )
            loss = _rolling_origin_ce(roll_logits, roll_idx, targets)
        else:
            logits, _ = self(inputs)
            r = self.receptive_field if self.train_opts.skip_partial else 0
            loss = F.cross_entropy(logits[..., r:], targets[..., r:])
        return {"val_loss": loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)

    def create_sampler(self) -> sampling.ObservationSampler:
        return sampling.StochasticSampler()


def _rolling_origin_ce(
    roll_logits: torch.Tensor, roll_idx: torch.Tensor, targets: torch.Tensor
) -> float:
    H = roll_logits.shape[-1]  # horizon
    sum_loss = 0.0
    num_pred = 0
    for logits, idx in zip(roll_logits, roll_idx):
        roll_targets = targets[..., idx : idx + H]
        ce = F.cross_entropy(logits, roll_targets, reduction="sum")
        sum_loss = sum_loss + ce
        num_pred = num_pred + roll_targets.numel()
    return sum_loss / num_pred
