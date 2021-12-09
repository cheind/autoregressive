import abc
import time
from typing import Any

import jsonargparse
from functools import partial
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import generators, sampling, wave


def load_curves(data: pl.LightningDataModule, n: int, seed: int = None):
    ds = (
        data.test_dataloader().dataset
        if data.test_dataloader() is not None
        else data.val_dataloader().dataset
    )
    g = torch.default_generator
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, len(ds), (n,), generator=g)

    curves = []
    conds = []

    for idx in ids:
        sm = ds[idx]
        curves.append(sm[0]["x"])
        conds.append(sm[0].get("c", None))

    return curves, conds


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
        self.horizon = horizon
        self.condition = condition
        self.num_curves = num_curves
        self.seed_center = seed_center
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()
        self.dt = data.dt if hasattr(data, "dt") else 1.0

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
        t = torch.arange(0, horizon + 1, 1) * self.dt
        for curve in curves.cpu():
            grid[0].plot(t, curve, alpha=0.8)
        grid[0].set_ylim(0, model.quantization_levels)
        grid[0].set_xlabel("Timestep")
        grid[0].set_ylabel("Quantization Level")
        fig.tight_layout()
        fig.savefig("tmp/wavenet_samples.svg", bbox_inches="tight")
        plt.show()


class PredictSignalsCommand(BaseCommand):
    """Predicts signals x~p(X_future|X_past,y) with optional conditioning."""

    def __init__(
        self,
        ckpt: str,
        data: pl.LightningDataModule = None,
        num_observed: int = None,
        horizon: int = None,
        num_curves: int = 4,
        num_trajectories: int = 1,
        show_confidence: bool = False,
        seed: int = None,
        noise_scale: int = 0,
        tau: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: data module (unused)
            horizon: number of samples to generate per curve
            observed: number of samples observed
            condition: optional period condition (int) to apply
            num_curves: number of curves to sample
            num_trajectories: number of trajectories to sample per curve
            seed_center: whether or not all curves start at Q//2
            tau: temperature scaling of logits
        """
        self.horizon = horizon
        self.num_observed = num_observed
        self.num_curves = num_curves
        self.num_trajectories = num_trajectories
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()
        self.dt = data.dt if hasattr(data, "dt") else 1.0
        self.data = data
        self.show_confidence = show_confidence
        self.seed = seed
        self.tau = tau
        self.noise_scale = noise_scale

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model: wave.WaveNet = self.model.to(dev).eval()

        horizon = self.horizon or self.model.receptive_field * 2
        num_obs = self.num_observed or self.model.receptive_field
        # Load curves from dataset
        curves, conditions = load_curves(self.data, self.num_curves, seed=self.seed)
        curves = torch.stack(curves, 0).to(dev)
        if conditions[0] is not None:
            conditions = torch.stack(conditions, 0).to(dev)
        else:
            conditions = None

        # Add noise to the observable part
        noise = (
            torch.randn_like(curves[..., :num_obs], dtype=torch.float32)
            * self.noise_scale
        )
        noise = noise.round().long()
        curves[..., :num_obs] += noise

        # Repeat if we need more than one trajectory per curve
        if self.num_trajectories > 1:
            curves = curves.repeat_interleave(self.num_trajectories, 0)
            if conditions is not None:
                conditions = conditions.repeat_interleave(self.num_trajectories, 0)

        g = generators.generate_fast(
            model=model,
            initial_obs=curves[..., :num_obs],
            sampler=partial(sampling.sample_stochastic, tau=self.tau),
            global_cond=conditions,
        )

        curves_pred, _ = generators.slice_generator(g, stop=horizon)
        curves_pred = curves_pred.cpu()

        # Plot
        t = torch.arange(0, max(num_obs + horizon, curves.shape[-1])) * self.dt
        fig = plt.figure(figsize=(8, 3))
        grid = ImageGrid(
            fig=fig,
            rect=111,
            nrows_ncols=(self.num_curves, 1),
            axes_pad=0.05,
            share_all=True,
            label_mode="1",
            aspect=False,
        )
        for idx, (c, ax) in enumerate(
            zip(curves[:: self.num_trajectories].cpu(), grid)
        ):
            if not self.show_confidence or self.num_trajectories < 10:
                # plot individual curves
                for j in range(self.num_trajectories):
                    ax.plot(
                        t[num_obs : num_obs + horizon],
                        curves_pred[idx * self.num_trajectories + j],
                        label="predicted" if (idx == 0 and j == 0) else None,
                    )
            else:
                # plot as mean and std
                batch_curves = curves_pred[
                    idx * self.num_trajectories : self.num_trajectories * (1 + idx)
                ]
                mean = batch_curves.float().mean(0)
                std = batch_curves.float().std(0)
                ax.fill_between(
                    t[num_obs : num_obs + horizon],
                    mean - 2 * std,
                    mean + 2 * std,
                    alpha=0.2,
                    label="predicted +/- 2$\sigma$",
                )
                ax.plot(
                    t[num_obs : num_obs + horizon], mean, "-", label="predicted mean"
                )

            ax.plot(
                t[:num_obs],
                c[:num_obs],
                linewidth=1.0,
                label="observed",
                c="k",
            )
            ax.plot(
                t[num_obs : curves.shape[-1]],
                c[num_obs : curves.shape[-1]],
                linewidth=1.0,
                label="gt",
                linestyle="--",
                alpha=0.8,
                c="k",
            )
            ax.set_ylim(0, model.quantization_levels)
            # ax.set_xlim(0, (num_obs + horizon + 100) * self.dt)
            ax.axvline(x=num_obs * self.dt, c="r", linestyle="--")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Quantization Level")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
        fig.tight_layout()
        fig.savefig("tmp/predict.svg", bbox_inches="tight")
        plt.show()


@torch.no_grad()
def main():

    command_map = {"sample": SampleSignalsCommand, "predict": PredictSignalsCommand}

    parser = jsonargparse.ArgumentParser("WaveNet on 1D Signals")
    subcommands = parser.add_subcommands()
    for cmd, klass in command_map.items():
        subcommands.add_subcommand(cmd, klass.get_arguments())
    config = parser.parse_args()
    configinit = parser.instantiate_classes(config)

    cmdname = configinit.subcommand
    cmd = command_map[cmdname](**(configinit[cmdname].as_dict()))
    cmd.run()

    # python -m autoregressive.scripts.wavenet_signals sample --config models\fseries_q127\config.yaml --ckpt "models\fseries_q127\wavenet-epoch=17-val_acc_epoch=0.9065.ckpt" --condition 4 --horizon 1000

    # python -m autoregressive.scripts.wavenet_signals predict --config models\fseries_q127\config.yaml --ckpt "models\fseries_q127\wavenet-epoch=17-val_acc_epoch=0.9065.ckpt" --horizon 1500 --num_observed 600 --num_trajectories 20 --num_curves 1 --seed 123 --show_confidence true

    # Apply to MNIST
    # python -m autoregressive.scripts.wavenet_signals predict --config models\mnist_q256\config.yaml --ckpt "models\mnist_q256\wavenet-epoch=09-val_acc_epoch=0.8954.ckpt" --horizon 392 --num_observed 392


if __name__ == "__main__":
    main()
