import argparse
import itertools
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_lightning.utilities.cli import LightningCLI


from autoregressive import wave, generators


def geometry(arg: str) -> Tuple[int, int]:
    return tuple(map(int, arg.split("x")))


class InstantiateOnlyLightningCLI(LightningCLI):
    def fit(self) -> None:
        return None


class GenerateLightningCLI(InstantiateOnlyLightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("-num-traj", type=int, default=1)
        parser.add_argument("-num-steps", type=int, default=31)
        parser.add_argument("-shift", type=int, default=0)
        parser.add_argument("-curves", default="4x1", metavar="ROWSxCOLS")
        parser.add_argument(
            "--fast-wavenet", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument("ckpt", type=str)
        parser.link_arguments(
            "data.quantization_levels",
            "model.quantization_levels",
            apply_on="instantiate",
        )
        return super().add_arguments_to_parser(parser)


@torch.no_grad()
def main():
    cli = GenerateLightningCLI(
        wave.WaveNet,
        pl.LightningDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
    )
    cfg = cli.config
    model: wave.WaveNet = cli.model
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(cfg["ckpt"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(dev)
    sampler = model.create_sampler()
    curve_layout = geometry(cfg["curves"])
    num_curves = curve_layout[0] * curve_layout[1]

    dm = cli.datamodule
    ds = dm.val_ds
    # ds.transform = datasets.Noise(scale=1e-1, p=1.0)
    # S = min(model.receptive_field + cfg["shift"], ds[0]["x"].shape[-1])
    S = 1

    fig = plt.figure()
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=curve_layout,
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
        aspect=True,
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # Get samples from dataset. These are n-dicts
    curves = list(itertools.islice(ds, num_curves))

    # Prepare observations. We batch all observations and then repeat
    # these observations for the number of trajectories
    obs = torch.stack([c["x"] for c in curves], 0)  # (B,Q,T)
    obs = obs.repeat(cfg["num_traj"], 1, 1).to(dev)

    # Generate. Note, we leave the last obs out so the first item predicted
    # overlaps the last observation and we hence get a smooth plot
    if cfg["fast_wavenet"]:
        g = generators.generate_fast(
            model=model,
            initial_obs=obs[..., :S],
            sampler=sampler,
        )
    else:
        g = generators.generate(
            model=model,
            initial_obs=obs[..., :S],
            sampler=sampler,
        )

    t0 = time.time()
    trajs, _ = generators.slice_generator(g, stop=cfg["num_steps"])  # (B,1,T)
    trajs = trajs.squeeze(1).cpu()
    tn = torch.arange(S, S + cfg["num_steps"], 1) * dm.dt
    print(f"Generation took {(time.time()-t0):.3f} secs")

    # Plot
    for idx, (ax, s) in enumerate(zip(grid, curves)):
        x, t = s["x"], s["t"]
        # ax.step(t, xo, c="k", linewidth=0.5, label="input", linestyle="--")
        # ax.scatter(
        #     t, x.argmax(0), c="k", s=10, linewidth=0.5, label="input", linestyle="--"
        # )

        for tidx, xn in enumerate(trajs[idx::num_curves]):
            # Note, above we step with num_curves to get all trajectories
            # for this axis. Related to repeat statement above.
            ax.step(tn, xn.argmax(0), label="generated" if tidx == 0 else "")
        # ax.set_ylim(-3, 3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")
    fig.suptitle(f"Sample generation by {type(model).__name__}")
    # fig.tight_layout()
    fig.savefig(f"tmp/generate_{type(model).__name__}.pdf")
    plt.show()


if __name__ == "__main__":
    # python -m autoregressive.examples.generate --config config.yaml C:\dev\autoregressive\lightning_logs\version_45\checkpoints\wavenet-epoch=28-val_loss=0.0002.ckpt
    main()