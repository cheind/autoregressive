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
