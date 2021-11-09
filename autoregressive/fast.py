import itertools
from typing import Protocol, List, TYPE_CHECKING, Iterator, Tuple

import torch

if TYPE_CHECKING:
    from .wave import WaveNet, WaveNetLayer

FastQueues = List[torch.FloatTensor]


def pop_push_queue(q: torch.Tensor, x, pop_size: int = 1) -> torch.Tensor:
    h = q[..., 0:pop_size]  # pop left (oldest)
    qout = q.roll(-1, -1)  # roll by one in left direction
    qout[..., -1:] = x  # push right (newest)
    return h, qout


def create_empty_queues(
    model: "WaveNet",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> FastQueues:
    queues = []
    for layer in model.layers:
        layer: "WaveNetLayer"
        q = torch.zeros(
            (batch_size, layer.wave_channels, layer.recurrent_size),
            dtype=dtype,
            device=device,
        )  # Populated with zeros will act like causal padding
        queues.append(q)
    return queues


def create_initialized_queues(
    model: "WaveNet", layer_inputs: List[torch.FloatTensor]
) -> FastQueues:
    queues = []
    for layer, layer_input in zip(model.wave_layers, layer_inputs):
        layer: WaveNetLayer
        assert layer_input.shape[-1] >= layer.dilation
        # you might have removed left-invalids when you shouldn't
        # see model.forward
        q = layer_input[..., -layer.dilation :]
        queues.append(q)
    return queues
