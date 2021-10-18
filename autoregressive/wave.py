import torch
import torch.nn
import torch.nn.functional as F

from .utils import causal_pad


class WaveNetLayer(torch.nn.Module):
    def __init__(
        self,
        dilation: int,
        residual_channels: int = 32,
        skip_channels: int = 32,
    ):
        super().__init__()
        self.dilation = dilation
        self.conv_dilation = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=2,
            dilation=dilation,
        )
        self.conv_tanh = torch.nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=2,
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

    def forward(self, x):
        x_dilated = self.conv_dilation(causal_pad(x, 2, self.dilation))
        x_filter = torch.tanh(self.conv_tanh(x_dilated))
        x_gate = torch.sigmoid(self.conv_sig(x_dilated))
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        return x_h + x_dilated, skip


class WaveNetBackbone(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers: int = 7,
    ):
        super().__init__()
        self.conv_input = torch.nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.layers = torch.nn.ModuleList(
            [
                WaveNetLayer(
                    dilation=2 ** d,
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                )
            ]
            for _ in range(num_blocks)
            for d in range(num_layers)
        )
        kernel_size = 2
        self.receptive_field = (kernel_size - 1) * sum(
            [2 ** i for _ in range(num_blocks) for i in range(num_layers)]
        ) + 1

    def forward(self, x):
        x = self.conv_input(x)
        skip_aggregate = 0.0
        for layer in self.layers:
            x, skip = layer(x)
            skip_aggregate = skip_aggregate + skip
        return skip_aggregate


class WaveNetForecaster(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        forecast_steps: int = 64,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers: int = 7,
    ):
        super().__init__()
        self.features = WaveNetBackbone(
            in_channels, residual_channels, skip_channels, num_blocks, num_layers
        )
        self.conv_mid = torch.nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.conv_output = torch.nn.Conv1d(skip_channels, forecast_steps, kernel_size=1)
        self.forecast_steps = forecast_steps

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = F.relu(self.conv_mid(x))
        x = self.conv_output(x)
        return x

    @property
    def receptive_field(self):
        return self.features.receptive_field
