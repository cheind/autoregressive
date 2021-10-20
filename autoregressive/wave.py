import torch
import torch.nn
import torch.nn.init
import torch.nn.functional as F

from .utils import causal_pad


def _init_weights(m):
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

    def forward(self, x):
        x_dilated = self.conv_dilation(causal_pad(x, 2, self.dilation))
        return self._forward_dilated(x_dilated)

    def forward_fast(self, x):
        """Fast wave layer forward.

        This function assumes that x is composed of the last 'recurrent'
        input, that is `dilation` steps back and the current input.

        Params
        ------
        x: (B,C,2) tensor
            Combination of last recurrent state and current input.

        Returns
        -------
        y: (B,C,1) tensor
            Result of gated-dilated convolution
        """
        x_dilated = F.conv1d(
            x,
            self.conv_dilation.weight,
            self.conv_dilation.bias,
            dilation=1,
        )
        return self._forward_dilated(x_dilated)

    def _forward_dilated(self, x_dilated):
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
                for _ in range(num_blocks)
                for d in range(num_layers)
            ]
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

    def forward_fast(self, x, queues):
        x = self.conv_input(x)
        skip_aggregate = 0.0
        out_queues = []
        for layer, q in zip(self.layers, queues):
            layer: WaveNetLayer
            h, qout = self._pop_push_queue(q, x)
            out_queues.append(qout)
            c = torch.cat((h, x), -1)
            x, skip = layer.forward_fast(c)
            skip_aggregate = skip_aggregate + skip
        return skip_aggregate, out_queues

    def _pop_push_queue(self, q, x):
        h = q[..., -1:]  # pop last
        qout = q.roll(1, -1)  # move last to front
        qout[..., 0:1] = x  # push front
        return h, qout

    def create_fast_queues(self, device: torch.device):
        queues = []
        for layer in self.layers:
            layer: WaveNetLayer
            q = torch.zeros(
                (1, layer.residual_channels, layer.dilation),
                dtype=layer.conv_dilation.weight.dtype,
                device=device,
            )
            queues.append(q)
        return queues

        # a = torch.cat((torch.narrow(a, -1, 1, a.shape[-1]-1), torch.rand(2,3,1)),-1);


class WaveNetLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
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
        self.conv_output = torch.nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels
        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = F.gelu(x)
        x = F.gelu(self.conv_mid(x))
        x = self.conv_output(x)
        return x

    def forward_fast(self, x, queues):
        x, queues = self.features.forward_fast(x, queues)
        x = F.gelu(x)
        x = F.gelu(self.conv_mid(x))
        x = self.conv_output(x)
        return x, queues

    @property
    def receptive_field(self):
        return self.features.receptive_field
