import itertools
import logging
from typing import Iterable, Iterator, List, Protocol, Tuple, Sequence
import dataclasses

import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.init

from . import losses
from . import fast

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


def causal_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """Performs a cause padding to avoid data leakage. Stride is assumed to be one."""
    left_pad = (kernel_size - 1) * dilation
    return F.pad(x, (left_pad, 0))


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, std=1e-3)


def compute_receptive_field(
    dilation_seq: Iterable[int] = None,
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
        dilations: list[int] = [2 ** i for i in range(8)] * 2,
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
        self.dilations = list(dilations)
        self.quantization_levels = quantization_levels
        self.wave_channels = wave_channels
        self.receptive_field = compute_receptive_field(dilations, kernel_size=2)
        self.num_left_invalid = self._compute_left_invalid()
        self.logits = WaveNetLogitsHead(
            wave_channels=wave_channels,
            out_channels=quantization_levels,
        )
        self.train_opts = train_opts
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")
        self.save_hyperparameters()

    def encode(self, x, remove_left_invalid: bool = True):
        skips = []
        layer_inputs = []
        x = causal_pad(x, 2, self.receptive_field - 1)
        x = self.input_conv(x)
        for layer in self.wave_layers:
            layer_inputs.append(x)
            x, skip = layer(x, fast=False)
            skips.append(skip)
        if remove_left_invalid:
            layer_inputs = self._remove_left_invalid(layer_inputs)
        return x, layer_inputs, skips

    def encode_one(self, x, queues: fast.FastQueues):
        skips = []
        updated_queues = []
        x = self.input_conv(x)  # no padding here
        for layer, q in zip(self.wave_layers, queues):
            h, qnew = fast.pop_push_queue(q, x)
            x = torch.cat((h, x), -1)  # prepare input
            x, skip = layer(x, fast=True)
            updated_queues.append(qnew)
            skips.append(skip)
        return x, skips, updated_queues

    def forward(self, x):
        # x (B,Q,T)
        encoded, _, skips = self.encode(x)
        return self.logits(encoded, skips)

    def forward_one(self, x, queues: fast.FastQueues):
        # x (B,Q,1)
        encoded, skips, queues = self.encode_one(x, queues)
        return self.logits(encoded, skips), queues

    def _compute_left_invalid(self):
        # TODO does not incorporate filter size other than 2
        p = self.receptive_field - 1
        num_inv = []
        for layer in self.wave_layers:
            num_inv.append(p)
            p = p - layer.dilation
        return num_inv

    def _remove_left_invalid(self, layer_inputs: torch.Tensor) -> torch.Tensor:
        # Note, layer inputs contain results from causal padding
        # at front. This decreases over layers based on dilation factors.
        # Hence when computing layer-inputs
        return [layer[..., p:] for layer, p in zip(layer_inputs, self.num_left_invalid)]

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
        targets: torch.Tensor = batch["y"][..., 1:]  # (B,T)
        inputs: torch.Tensor = batch["x"][..., :-1]  # (B,Q,T)
        logits = self(inputs)

        r = self.receptive_field if self.train_opts.skip_partial_receptive_field else 0
        loss = F.cross_entropy(logits[..., r:], targets[..., r:])

        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        inputs: torch.Tensor = batch["x"][..., :-1]
        targets: torch.Tensor = batch["y"][..., 1:]

        _, roll_logits, roll_idx = losses.rolling_nstep(
            self,
            self.create_sampler(greedy=True),
            inputs,
            num_generate=self.train_opts.val_unroll_steps,
            max_rolls=self.train_opts.val_max_rolls,
            random_rolls=True,
            skip_partial=self.train_opts.skip_partial_receptive_field,
        )
        loss = losses.rolling_nstep_ce(roll_logits, roll_idx, targets)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)

    def create_sampler(self, greedy: bool = False):
        def greedy_sampler(logits):
            # logits (B,Q,1)
            amax = torch.argmax(logits, dim=1, keepdim=False)  # (B,1)
            oh = F.one_hot(amax, num_classes=self.quantization_levels)  # (B,1,Q)
            return oh.permute(0, 2, 1).float()  # (B,Q,1)

        def stochastic_sampler(logits):
            # logits (B,Q,1)
            bins = D.Categorical(logits=logits.permute(0, 2, 1)).sample()  # (B,1)
            oh = F.one_hot(bins, num_classes=self.quantization_levels)  # (B,1,Q)
            return oh.permute(0, 2, 1).float()  # (B,Q,1)

        if greedy:
            return greedy_sampler
        else:
            return stochastic_sampler
