from typing import List, Optional
import logging

import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.init

from .utils import causal_pad

FastQueues = List[torch.FloatTensor]


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class WaveNetLayer(torch.nn.Module):
    def __init__(
        self,
        dilation: int,
        residual_channels: int = 32,
        skip_channels: int = 32,
    ):
        super().__init__()
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.conv_dilation = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.conv_tanh = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1,
        )
        self.conv_sig = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1,
        )
        self.conv_skip = torch.nn.Conv1d(
            residual_channels,
            skip_channels,
            kernel_size=1,
        )
        self.conv_residual = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1,
        )

    def forward(self, x, fast: bool = False):
        """
        When fast is enabled, this function assumes that x is composed
        of the last 'recurrent' input, that is `dilation` steps back and
        the current input.
        """
        if fast:
            x_dilated = F.conv1d(
                x,
                self.conv_dilation.weight,
                self.conv_dilation.bias,
                dilation=1,
            )
        else:
            x_dilated = self.conv_dilation(
                causal_pad(x, 2, self.dilation),
            )
        x_filter = torch.tanh(self.conv_tanh(x_dilated))
        x_gate = torch.sigmoid(self.conv_sig(x_dilated))
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        return x_h + x_dilated, skip


class WaveNetBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers_per_block: int = 7,
    ):
        super().__init__()
        self.conv_input = torch.nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.wave_layers = torch.nn.ModuleList(
            [
                WaveNetLayer(
                    dilation=2 ** d,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                )
                for _ in range(num_blocks)
                for d in range(num_layers_per_block)
            ]
        )
        kernel_size = 2
        self.receptive_field = (kernel_size - 1) * sum(
            [2 ** i for _ in range(num_blocks) for i in range(num_layers_per_block)]
        ) + 1
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.in_channels = in_channels

    def encode(self, x):
        skips = []
        layer_inputs = []
        x = self.conv_input(x)
        for layer in self.wave_layers:
            layer_inputs.append(x)
            x, skip = layer(x, fast=False)
            skips.append(skip)
        return x, layer_inputs, skips

    def encode_one(self, x, queues: FastQueues):
        skips = []
        layer_inputs = []
        updated_queues = []
        x = self.conv_input(x)
        for layer, q in zip(self.wave_layers, queues):
            layer_inputs.append(x)
            x, qnew = self._next_input_from_queue(q, x)
            x, skip = layer(x, fast=True)
            updated_queues.append(qnew)
            skips.append(skip)
        return x, layer_inputs, skips, updated_queues

    def _next_input_from_queue(self, q: torch.Tensor, x):
        h = q[..., 0:1]  # pop left (oldest)
        qout = q.roll(-1, -1)  # roll by one in left direction
        qout[..., -1:] = x  # push right (newest)
        x = torch.cat((h, x), -1)  # prepare input
        return x, qout

    def create_fast_queues(
        self,
        layer_inputs: Optional[List[torch.FloatTensor]],
        device: Optional[torch.device] = None,
    ) -> FastQueues:
        assert not (layer_inputs is None and device is None)
        if layer_inputs is None:
            layer_inputs = [None] * len(self.wave_layers)
        queues = []
        for layer, layer_input in zip(self.wave_layers, layer_inputs):
            layer: WaveNetLayer
            if layer_input is None:
                q = torch.zeros(
                    (1, layer.residual_channels, layer.dilation),
                    dtype=layer.conv_dilation.weight.dtype,
                    device=device,
                )  # That's the same as causal padding
            else:
                q = layer_input[..., -layer.dilation :].detach().clone()
            queues.append(q)
        return queues


class RegressionWaveNet(WaveNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers_per_block: int = 7,
        forecast_steps: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
        )

        self.conv_mid = torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_output = torch.nn.Conv1d(skip_channels, forecast_steps, kernel_size=1)
        self.forecast_steps = forecast_steps
        self.apply(wave_init_weights)

    def forward(self, x):
        e, layer_inputs, skips = self.encode(x)
        x = self._head(e, layer_inputs, skips)
        return x

    def forward_one(self, x, queues: FastQueues):
        e, layer_inputs, skips, queues = self.encode_one(x, queues)
        x = self._head(e, layer_inputs, skips)
        return x, queues

    def _head(self, encoded, layer_inputs, skips):
        del encoded, layer_inputs
        x = torch.stack(skips, dim=0).sum(dim=0)
        x = F.gelu(x)
        x = F.gelu(self.conv_mid(x))
        x = self.conv_output(x)
        return x


def _exp_weights(T, n):
    """Returns exponential decaying weights for T terms from 1.0 down to n (n>0)"""
    n = torch.as_tensor(n).float()
    x = torch.arange(0, T, 1)
    b = -torch.log(n) / (T - 1)
    return torch.exp(-x * b)


_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class LitRegressionWaveNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        forecast_steps: int = 64,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers_per_block: int = 7,
        train_full_receptive_field: bool = True,
        train_exp_decay: bool = False,
    ) -> None:
        super().__init__()
        self.wavenet = RegressionWaveNet(
            in_channels=in_channels,
            forecast_steps=forecast_steps,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
        )
        self.train_full_receptive_field = train_full_receptive_field
        self.receptive_field = self.wavenet.receptive_field
        self.train_exp_decay = train_exp_decay
        if self.train_exp_decay and forecast_steps > 1:
            self.register_buffer(
                "l1_weights", _exp_weights(forecast_steps, 1e-3).view(1, -1, 1)
            )
        else:
            self.l1_weights = 1.0
        _logger.info(f"Receptive field of model {self.wavenet.receptive_field}")
        super().save_hyperparameters()

    # def forward(self, x, return_outputs: bool = False):
    #     return self.wave(x, return_outputs=return_outputs)

    # def forward_one(self, x, queues):
    #     return self.wave.forward_one(x, queues)

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=1, min_lr=1e-7, threshold=1e-7
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _step(self, batch, batch_idx) -> torch.FloatTensor:
        del batch_idx
        x = batch["x"][..., :-1].unsqueeze(1)
        y = batch["xo"][..., 1:]
        y = y.unfold(-1, self.wavenet.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        yhat = self.wavenet(x)
        r = self.receptive_field if self.train_full_receptive_field else 0
        losses = F.l1_loss(yhat[..., r:n], y[..., r:], reduction="none")
        return torch.mean(losses * self.l1_weights)


# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py