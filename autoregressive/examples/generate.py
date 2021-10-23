import argparse
import itertools
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_lightning.utilities.cli import LightningCLI

from .. import dataset, trainer, wave


def geometry(arg: str) -> Tuple[int, int]:
    return tuple(map(int, arg.split("x")))


class InstantiateOnlyLightningCLI(LightningCLI):
    def fit(self) -> None:
        return None


class GenerateLightningCLI(InstantiateOnlyLightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("-num-traj", type=int, default=1)
        parser.add_argument("-num-steps", type=int, default=512)
        parser.add_argument("-curves", default="4x1", metavar="ROWSxCOLS")
        parser.add_argument(
            "--fast-wavenet", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument("ckpt", type=str)
        return super().add_arguments_to_parser(parser)


@torch.no_grad()
def main():
    cli = GenerateLightningCLI(
        wave.WaveNetBase,
        trainer.FSeriesDataModule,
        subclass_mode_model=True,
    )
    cfg = cli.config
    model = cli.model
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(cfg["ckpt"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(dev)
    sampler = model.create_sampler()
    curve_layout = geometry(cfg["curves"])
    num_curves = curve_layout[0] * curve_layout[1]

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure()
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=curve_layout,
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # Get samples from dataset. These are n-dicts
    curves = list(itertools.islice(dataset_val, num_curves))

    # Prepare observations. We batch all observations and then repeat
    # these observations for the number of trajectories
    obs = torch.stack([c["x"] for c in curves], 0).unsqueeze(1)  # (B,1,T)
    obs = obs.repeat(cfg["num_traj"], 1, 1).to(dev)

    # Generate. Note, we leave the last obs out so the first item predicted
    # overlaps the last observation and we hence get a smooth plot
    if cfg["fast_wavenet"]:
        g = wave.generate_fast(
            model=model,
            initial_obs=obs[..., :-1],
            sampler=sampler,
        )
    else:
        g = wave.generate(
            model=model,
            initial_obs=obs[..., :-1],
            sampler=sampler,
        )

    t0 = time.time()
    trajs = torch.cat(list(itertools.islice(g, cfg["num_steps"])), -1).squeeze(1).cpu()
    print(f"Generation took {(time.time()-t0):.3f} secs")

    # Plot
    for idx, (ax, s) in enumerate(zip(grid, curves)):
        xo, t = s["xo"], s["t"]
        dt = t[-1] - t[-2]
        tn = torch.arange(0.0, dt * cfg["num_steps"], dt) + t[-1]

        ax.plot(t, xo, c="k", linewidth=0.5, label="input")
        for tidx, xn in enumerate(trajs[idx::num_curves]):
            # Note, above we step with num_curves to get all trajectories
            # for this axis. Related to repeat statement above.
            ax.plot(tn, xn, label="generated" if tidx == 0 else "")
        ax.set_ylim(-3, 3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.suptitle(f"Sample generation by {type(model).__name__}")
    fig.savefig(f"tmp/generate_{type(model).__name__}.pdf")
    plt.show()


if __name__ == "__main__":
    # python -m autoregressive.examples.generate --config config.yaml C:\dev\autoregressive\lightning_logs\version_45\checkpoints\wavenet-epoch=28-val_loss=0.0002.ckpt
    main()