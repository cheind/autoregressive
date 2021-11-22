import dataclasses
import logging
from typing import Iterable

import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.init

from . import generators, sampling, encoding

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
        self.in_channels = in_channels


class WaveNetLayer(WaveLayerBase):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        wave_channels: int = 32,
        cond_channels: int = None,
    ):
        super().__init__(
            kernel_size=kernel_size,
            dilation=dilation,
            in_channels=wave_channels,
        )
        self.wave_channels = wave_channels
        self.cond_channels = cond_channels
        self.conv_dilation = torch.nn.Conv1d(
            wave_channels,
            2 * wave_channels,  # We stack W f,k and W g,k, similar to PixelCNN
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv_skip = torch.nn.Conv1d(
            wave_channels,
            wave_channels,
            kernel_size=1,
        )
        if cond_channels is not None:
            self.conv_cond = torch.nn.Conv1d(
                cond_channels,
                wave_channels * 2,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, causal_pad: bool = True):
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x_dilated = self.conv_dilation(F.pad(x, p))
        if self.cond_channels:
            assert c is not None, "conditioning required"
            x_cond = self.conv_cond(c)
            x_dilated = x_dilated + x_cond
        x_filter = torch.tanh(x_dilated[:, : self.wave_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.wave_channels :])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        if causal_pad:
            out = x + skip
        else:
            out = x[..., self.causal_left_pad :] + skip
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

    def forward(self, x, c: torch.Tensor = None, causal_pad: bool = True):
        del c
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x = self.conv(F.pad(x, p))
        return x, x.new_zeros(*x.shape)  # zeros are not changing head


class WaveNetLogitsHead(WaveLayerBase):
    def __init__(self, wave_channels: int, out_channels: int):
        super().__init__(kernel_size=1, dilation=1, in_channels=wave_channels)
        self.transform = torch.nn.Sequential(
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (i.e on sum of skips) # noqa:E501
            torch.nn.Conv1d(
                wave_channels, wave_channels * 2, kernel_size=1
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.LeakyReLU(),
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
    train_ro_horizon: int = 1
    train_ro_num_origins: int = None
    train_ro_loss_lambda: float = 1.0
    val_ro_horizon: int = 32
    val_ro_num_origins: int = 8


class WaveNet(pl.LightningModule):
    def __init__(
        self,
        quantization_levels: int = 2 ** 8,
        wave_dilations: list[int] = [2 ** i for i in range(8)] * 2,
        wave_kernel_size: int = 2,
        wave_channels: int = 32,
        cond_channels: int = None,
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
                cond_channels=cond_channels,
            )
            for d in wave_dilations
        ]
        self.layers = torch.nn.ModuleList(layers)
        self.logits = WaveNetLogitsHead(
            wave_channels=wave_channels,
            out_channels=quantization_levels,
        )
        self.quantization_levels = quantization_levels
        self.conditioning_channels = cond_channels
        self.train_opts = train_opts
        self.receptive_field = compute_receptive_field(
            kernel_sizes=[layer.kernel_size for layer in self.layers],
            dilations=[layer.dilation for layer in self.layers],
        )
        self.apply(wave_init_weights)
        _logger.info(f"Receptive field of WaveNet {self.receptive_field}")
        self.save_hyperparameters()

    def encode(self, x, c: torch.Tensor = None, causal_pad: bool = True):
        x = encoding.one_hotf(x, quantization_levels=self.quantization_levels)

        skips = []
        layer_inputs = []

        for layer in self.layers:
            layer: WaveLayerBase
            layer_inputs.append(x)
            x, skip = layer(x, c=c, causal_pad=causal_pad)
            skips.append(skip)

        return x, layer_inputs, skips

    def forward(self, x, c: torch.Tensor = None, causal_pad: bool = True):
        # x (B,Q,T) or (B,T)
        # c None, (B,C,T) or (B,C,1)
        encoded_result = self.encode(x, c=c, causal_pad=causal_pad)
        return self.logits(encoded_result[0], encoded_result[2]), encoded_result

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
        # One-step dense loss
        one_loss, _ = self._step(
            batch,
            horizon=1,
            num_origins=None,
            sampler=None,
        )

        # N-Step unrolling sparse loss
        n_loss = 0.0
        if (
            self.train_opts.train_ro_horizon > 1
            and self.train_opts.train_ro_loss_lambda > 0.0  # noqa:W503
        ):
            n_loss, _ = self._step(
                batch,
                horizon=self.train_opts.train_ro_horizon,
                num_origins=self.train_opts.train_ro_num_origins,
                sampler=sampling.sample_differentiable,
            )

        # Combine losses
        loss = one_loss + self.train_opts.train_ro_loss_lambda * n_loss
        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        sampler = None
        if self.train_opts.val_ro_horizon > 1:
            sampler = sampling.sample_greedy

        loss, acc = self._step(
            batch,
            horizon=self.train_opts.val_ro_horizon,
            num_origins=self.train_opts.val_ro_num_origins,
            sampler=sampler,
        )
        return {"val_loss": loss, "val_acc": acc}

    def _step(
        self,
        batch,
        horizon: int,
        num_origins: int,
        sampler: sampling.ObservationSampler,
    ):
        series, _ = batch
        inputs: torch.Tensor = series["x"][..., :-1]  # (B,T)
        targets: torch.Tensor = series["x"][..., 1:]  # (B,T)
        conds = None
        if "c" in series:
            conds = series["c"]  # (B,C,T)

        if horizon == 1:
            logits, _ = self.forward(inputs, c=conds)
            if self.train_opts.skip_partial:
                r = self.receptive_field
                logits = logits[..., :-r]
                targets = inputs[..., r:]
        else:
            _, logits, ridx = generators.rolling_origin(
                self,
                sampler=sampler,
                obs=inputs,
                horizon=horizon,
                num_origins=num_origins,
                random_origins=True,
                skip_partial=self.train_opts.skip_partial,
            )
            logits, targets = generators.collate_rolling_origin(logits, ridx, targets)

        loss = F.cross_entropy(logits, targets)
        acc = torch.sum(logits.argmax(1) == targets) / targets.numel()
        return loss, acc

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=False)

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=False)
        self.log("val_acc_epoch", avg_acc, prog_bar=True)
