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
    dilations: Iterable[int],
    kernel_sizes: Iterable[int],
) -> int:
    return sum((k - 1) * d for k, d in zip(kernel_sizes, dilations)) + 1


class WaveNetLayer(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        wave_channels: int = 32,
    ):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.wave_channels = wave_channels
        self.causal_left_pad = (kernel_size - 1) * dilation
        self.recurrent_size = self.kernel_size - 1
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
            assert h.shape[-1] == self.recurrent_size
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


class WaveNetInputLayer(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        wave_channels: int = 32,
    ):
        super().__init__()
        self.dilation = 1
        self.kernel_size = kernel_size
        self.wave_channels = wave_channels
        self.input_channels = input_channels
        self.causal_left_pad = (kernel_size - 1) * self.dilation
        self.recurrent_size = self.kernel_size - 1
        self.conv = torch.nn.Conv1d(
            input_channels,
            wave_channels,
            kernel_size=kernel_size,
            dilation=self.dilation,
        )

    def forward(self, x, h: torch.Tensor = None):
        if h is not None:
            assert h.shape[-1] == self.recurrent_size
            x = torch.cat((h, x), -1)
        else:
            x = F.pad(x, (self.causal_left_pad, 0))
        x = self.conv(x)
        return x, x.new_zeros(*x.shape)  # zeros are not changing head


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
        del encoded
        skips = skips[1:]  # remove input-layer dummy skip
        return self.transform(torch.stack(skips, dim=0).sum(dim=0))


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
        quantization_levels: int = 2 ** 8,
        wave_dilations: list[int] = [2 ** i for i in range(8)] * 2,
        wave_kernel_size: int = 2,
        wave_channels: int = 32,
        input_kernel_size: int = 1,
        train_opts: WaveNetTrainOpts = WaveNetTrainOpts(),
    ):
        super().__init__()
        self.input_conv = torch.nn.Conv1d(
            quantization_levels,
            wave_channels,
            kernel_size=input_kernel_size,
            dilation=1,
        )
        layers = [
            WaveNetInputLayer(
                kernel_size=input_kernel_size,
                input_channels=quantization_levels,
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
        self.wave_channels = wave_channels
        self.input_kernel_size = input_kernel_size
        self.train_opts = train_opts
        self.receptive_field = compute_receptive_field(
            kernel_sizes=[layer.kernel_size for layer in self.layers],
            dilations=[layer.dilation for layer in self.layers],
        )
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")
        self.save_hyperparameters()

    def encode(self, x, queues: fast.FastQueues = None):
        skips = []
        layer_inputs = []
        out_queues = []
        if queues is None:
            queues = [None] * len(self.layers)

        for layer, q in zip(self.layers, queues):
            layer_inputs.append(x)
            h = None
            if q is not None:
                h, q = fast.pop_push_queue(q, x)
            out_queues.append(q)
            x, skip = layer(x, h=h)
            skips.append(skip)

        if queues[0] is None:
            out_queues = None

        return x, layer_inputs, skips, out_queues

    # def encode_one(self, x, queues: fast.FastQueues):
    #     skips = []
    #     updated_queues = []
    #     x = self.input_conv(x)  # no padding here
    #     for layer, q in zip(self.wave_layers, queues):
    #         h, qnew = fast.pop_push_queue(q, x)
    #         x = torch.cat((h, x), -1)  # prepare input
    #         x, skip = layer(x, fast=True)
    #         updated_queues.append(qnew)
    #         skips.append(skip)
    #     return x, skips, updated_queues

    def forward(self, x, queues: fast.FastQueues = None):
        # x (B,Q,T) or (B,Q,1)
        encoded, _, skips, outqueues = self.encode(x)
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
