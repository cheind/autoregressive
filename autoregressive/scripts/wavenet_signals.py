from typing import Any

import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import abc
import time
import pytorch_lightning as pl
from torchvision.utils import make_grid
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import datasets, generators, sampling, wave


class BaseCommand(abc.ABC):
    @classmethod
    def get_arguments(cls):
        parser = jsonargparse.ArgumentParser()
        parser.add_class_arguments(cls, None)
        cls._add_config_arg(parser)
        return parser

    @abc.abstractmethod
    def run():
        ...

    @property
    def default_device(self):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return dev

    @staticmethod
    def _add_config_arg(parser: jsonargparse.ArgumentParser):
        # When loading from PL config file, ignore all other entries except data.
        # See https://github.com/PyTorchLightning/pytorch-lightning/discussions/10956#discussioncomment-1765546
        from jsonargparse import SUPPRESS

        for key in ["model", "trainer", "seed_everything", "optimizer", "lr_scheduler"]:
            parser.add_argument(f"--{key}", type=Any, help=SUPPRESS)
        parser.add_argument("--config", action=jsonargparse.ActionConfigFile)


class SampleSignalsCommand(BaseCommand):
    """Samples signals x~p(X|y) with optional conditioning."""

    def __init__(
        self,
        ckpt: str,
        data: pl.LightningDataModule = None,
        horizon: int = None,
        condition: int = None,
        num_curves: int = 4,
        seed_center: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: data module (unused)
            horizon: number of samples to generate per curve
            condition: optional period condition (int) to apply
            num_curves: number of curves to sample
            seed_center: whether or not all curves start at Q//2
        """
        del data
        self.horizon = horizon
        self.condition = condition
        self.num_curves = num_curves
        self.seed_center = seed_center
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model: wave.WaveNet = self.model.to(dev).eval()
        seeds = torch.randint(0, model.quantization_levels, size=(self.num_curves, 1))
        if self.seed_center:
            seeds.fill_(model.quantization_levels // 2)
        seeds = seeds.to(dev)
        horizon = self.horizon or self.model.receptive_field

        global_condition = None
        if self.condition is not None:
            global_condition = (
                F.one_hot(
                    torch.tensor(self.condition),
                    num_classes=model.conditioning_channels,
                )
                .view(1, -1, 1)
                .float()
                .to(dev)
            )

        g = generators.generate_fast(
            model=model,
            initial_obs=seeds,
            sampler=sampling.sample_stochastic,
            global_cond=global_condition,
        )

        t0 = time.time()
        curves, _ = generators.slice_generator(g, stop=horizon)  # (B,T)
        curves = torch.cat((seeds, curves), 1)
        print(f"Generation took {(time.time()-t0):.3f} secs")

        # Plot
        fig = plt.figure(figsize=(8, 3))
        grid = ImageGrid(
            fig=fig,
            rect=111,
            nrows_ncols=(1, 1),
            axes_pad=0.05,
            share_all=True,
            # label_mode="1",
            aspect=False,
        )
        t = torch.arange(0, horizon + 1, 1)
        for curve in curves.cpu():
            grid[0].step(t, curve, alpha=0.8)
        grid[0].set_ylim(0, model.quantization_levels)
        grid[0].set_xlabel("Timestep")
        grid[0].set_ylabel("Quantization Level")
        fig.tight_layout()
        fig.savefig("tmp/wavenet_samples.svg", bbox_inches="tight")
        plt.show()


@torch.no_grad()
def main():

    command_map = {
        "sample": SampleSignalsCommand,
    }

    parser = jsonargparse.ArgumentParser("WaveNet on 1D Signals")
    subcommands = parser.add_subcommands()
    for cmd, klass in command_map.items():
        subcommands.add_subcommand(cmd, klass.get_arguments())
    config = parser.parse_args()
    configinit = parser.instantiate_classes(config)

    cmdname = configinit.subcommand
    cmd = command_map[cmdname](**(configinit[cmdname].as_dict()))
    cmd.run()


if __name__ == "__main__":
    main()
