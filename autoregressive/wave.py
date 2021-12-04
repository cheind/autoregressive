import dataclasses
import logging
from typing import Iterable, Optional

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
            torch.nn.init.normal_(m.bias, mean=1e-3, std=1e-2)


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
        dilation_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 32,
        cond_channels: int = None,
        in_channels: int = None,
        bias: bool = False,
    ):
        in_channels = in_channels or residual_channels
        super().__init__(
            kernel_size=kernel_size,
            dilation=dilation,
            in_channels=residual_channels,
        )
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels
        self.conv_dilation = torch.nn.Conv1d(
            in_channels,
            2 * dilation_channels,  # We stack W f,k and W g,k, similar to PixelCNN
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.conv_res = torch.nn.Conv1d(
            dilation_channels,
            residual_channels,
            kernel_size=1,
            bias=bias,
        )
        self.conv_skip = torch.nn.Conv1d(
            dilation_channels,
            skip_channels,
            kernel_size=1,
            bias=bias,
        )
        self.conv_cond = None
        if cond_channels is not None:
            self.conv_cond = torch.nn.Conv1d(
                cond_channels,
                dilation_channels * 2,
                kernel_size=1,
                bias=bias,
            )

        self.conv_input = None
        if in_channels != residual_channels:
            self.conv_input = torch.nn.Conv1d(
                in_channels,
                residual_channels,
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, causal_pad: bool = True):
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x_dilated = self.conv_dilation(F.pad(x, p))
        if self.cond_channels:
            assert c is not None, "conditioning required"
            x_cond = self.conv_cond(c)
            x_dilated = x_dilated + x_cond
        x_filter = torch.tanh(x_dilated[:, : self.dilation_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.dilation_channels :])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        res = self.conv_res(x_h)

        if self.conv_input is not None:
            x = self.conv_input(x)  # convert to res channels

        if causal_pad:
            out = x + res
        else:
            out = x[..., self.causal_left_pad :] + res
        return out, skip


class WaveNetInputLayer(WaveLayerBase):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        residual_channels: int = 32,
        bias: bool = False,
    ):
        super().__init__(kernel_size=kernel_size, dilation=1, in_channels=in_channels)
        self.residual_channels = residual_channels
        self.conv = torch.nn.Conv1d(
            in_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=1,
            bias=bias,
        )

    def forward(self, x, c: torch.Tensor = None, causal_pad: bool = True):
        del c
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x = self.conv(F.pad(x, p))
        return x, x.new_zeros(*x.shape)  # zeros are not changing head


class WaveNetLogitsHead(WaveLayerBase):
    def __init__(
        self,
        skip_channels: int,
        head_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super().__init__(kernel_size=1, dilation=1, in_channels=skip_channels)
        self.transform = torch.nn.Sequential(
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (i.e on sum of skips) # noqa:E501
            torch.nn.Conv1d(
                skip_channels,
                head_channels,
                kernel_size=1,
                bias=bias,
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                head_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),  # logits
        )

    def forward(self, encoded, skips):
        del encoded
        return self.transform(sum(skips))


@dataclasses.dataclass
class WaveNetTrainOpts:
    skip_partial: bool = True
    lr: float = 1e-3
    sched_patience: int = 25
    train_ro_horizon: int = 1
    train_ro_num_origins: Optional[int] = None
    train_ro_loss_lambda: float = 1.0
    val_ro_horizon: int = 32
    val_ro_num_origins: Optional[int] = 8


class WaveNet(pl.LightningModule):
    def __init__(
        self,
        quantization_levels: int = 2 ** 8,
        wave_dilations: list[int] = [2 ** i for i in range(8)] * 2,
        wave_kernel_size: int = 2,
        extra_in_channels: int = 0,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 32,
        cond_channels: int = None,
        head_channels: int = 32,
        input_kernel_size: int = 1,
        conv_bias: bool = False,
        train_opts: WaveNetTrainOpts = WaveNetTrainOpts(),
    ):
        super().__init__()
        in_channels = quantization_levels + extra_in_channels
        layers = [
            WaveNetLayer(
                kernel_size=input_kernel_size,
                dilation=1,
                in_channels=in_channels,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=True,
            )
        ]
        layers += [
            WaveNetLayer(
                kernel_size=wave_kernel_size,
                dilation=d,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=conv_bias,
            )
            for d in wave_dilations
        ]
        self.layers = torch.nn.ModuleList(layers)
        self.logits = WaveNetLogitsHead(
            skip_channels=skip_channels,
            head_channels=head_channels,
            out_channels=quantization_levels,
            bias=True,
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
                    min_lr=5e-6,
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
        self._log_training_details(batch_idx)
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
        cond = None
        if "c" in series:
            # TODO Note, we support only global conditioning in training for now,
            # until we figure out how to implement local conditioning nicely
            # in functional generators and rolling origin.
            cond = series["c"].float()
            if cond.shape[-1] == series["x"].shape[-1]:
                # local conditioning - cut to same length as inputs
                cond = cond[..., :-1]
                # # As a test we add cond directly to input
                # inputs = encoding.one_hotf(
                #     inputs, quantization_levels=self.quantization_levels
                # )
                # inputs = torch.cat((inputs, cond), 1)
                # cond = None

        if horizon == 1:
            logits, _ = self.forward(inputs, c=cond)  # supports local cond as well
            if self.train_opts.skip_partial:
                r = self.receptive_field
                logits = logits[..., r:]
                targets = targets[..., r:]
        else:
            _, logits, ridx = generators.rolling_origin(
                self,
                sampler=sampler,
                obs=inputs,
                horizon=horizon,
                num_origins=num_origins,
                random_origins=True,
                skip_partial=self.train_opts.skip_partial,
                global_cond=cond,  # only supports global cond for now!
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

    def _log_training_details(self, batch_idx: int):
        if batch_idx % 100 != 0:
            return

        histogram_inputs = []
        for i in range(len(self.layers)):
            layer: WaveNetLayer = self.layers[i]
            histogram_inputs.append((f"layer{i}_dilation", layer.conv_dilation))
            histogram_inputs.append((f"layer{i}_skip", layer.conv_skip))
            histogram_inputs.append((f"layer{i}_res", layer.conv_res))
            if layer.conv_cond:
                histogram_inputs.append((f"layer{i}_cond", layer.conv_cond))

        tb = self.logger.experiment
        for h in histogram_inputs:
            tb.add_histogram(h[0], h[1].weight, global_step=self.global_step)
