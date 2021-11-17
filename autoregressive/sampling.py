__all__ = ["ObservationSampler", "GreedySampler", "StochasticSampler"]
from typing import Protocol

import torch
import torch.distributions as D


class ObservationSampler(Protocol):
    """Protocol for all sampling strategies from model logits.
    Generally used in generative mode."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample according to logits.

        Params
        ------
        logits: (B,Q,T) tensor
            model logits for a single temporal timestep.

        Returns
        -------
        sample: (B,T) or (B,Q,T)
            Sample either compressed or 'one'-hot encoded
        """
        ...


class GreedySampler(ObservationSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=1, keepdim=False)  # (B,T)


class StochasticSampler(ObservationSampler):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # Note, sampling from dists requires (*,Q) layout
        logits = logits.permute(0, 2, 1)
        return D.Categorical(logits=logits).sample()  # (*,)
