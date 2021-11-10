import itertools
from typing import Protocol, List, TYPE_CHECKING, Iterator, Tuple

import torch

if TYPE_CHECKING:
    from .wave import WaveNet, WaveLayerBase

FastQueues = List[torch.FloatTensor]


def read_queue(q: torch.Tensor, n: int, stride: int) -> torch.Tensor:
    h = q[..., 0 : stride * n : stride]
    return h


def push_queue(q: torch.Tensor, x) -> torch.Tensor:
    qout = q.roll(-1, -1)  # roll by one in left direction
    qout[..., -1:] = x  # push right (newest)
    return qout


def create_zero_queues(
    model: "WaveNet",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> FastQueues:
    queues = []
    for layer in model.layers:
        layer: "WaveLayerBase"
        q = torch.zeros(
            (batch_size, layer.in_channels, layer.temporal_queue_size),
            dtype=dtype,
            device=device,
        )  # Populated with zeros will act like causal padding
        queues.append(q)
    return queues


def create_initialized_queues(
    model: "WaveNet", layer_inputs: List[torch.FloatTensor]
) -> FastQueues:
    dev = layer_inputs[0].device
    dtype = layer_inputs[0].dtype
    B = layer_inputs[0].shape[0]
    queues = create_zero_queues(model, device=dev, dtype=dtype, batch_size=B)
    for q, layer_input in zip(queues, layer_inputs):
        t_layer = layer_input.shape[-1]
        t_queue = q.shape[-1]
        t = min(t_layer, t_queue)
        q[..., t_queue - t :] = layer_input[..., t_layer - t :]
        # copy most recent inputs, note, do not use `-t:` alone,
        # since it will copy everything when t=0
    return queues
