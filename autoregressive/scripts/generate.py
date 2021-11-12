import argparse
import itertools
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_lightning.utilities.cli import LightningCLI


from autoregressive import wave, generators, signal, datasets


def geometry(arg: str) -> Tuple[int, int]:
    return tuple(map(int, arg.split("x")))


class InstantiateOnlyLightningCLI(LightningCLI):
    def fit(self) -> None:
        return None


class GenerateLightningCLI(InstantiateOnlyLightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("-samples", type=int, default=1)
        parser.add_argument("-horizon", type=int, default=None)
        parser.add_argument("-initial-obs", type=int)
        parser.add_argument("-curves", default="4x1", metavar="ROWSxCOLS")
        parser.add_argument("-noise-scale", type=float)
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
    if cfg["noise_scale"] is not None:
        ds.transform = datasets.transforms.chain_transforms(
            datasets.transforms.Noise(scale=cfg["noise_scale"]), ds.transform
        )

    # ds.transform = datasets.Noise(scale=1e-1, p=1.0)
    # S = min(model.receptive_field + cfg["shift"], ds[0]["x"].shape[-1])
    R = cfg["initial_obs"] or model.receptive_field
    steps = cfg["horizon"] or ds[0]["x_k"].shape[-1] - R

    fig = plt.figure()
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=curve_layout,
        axes_pad=0.05,
        share_all=True,
        # label_mode="1",
        aspect=True,
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # Get samples from dataset. These are n-dicts
    curves = list(itertools.islice(ds, num_curves))

    # Prepare observations. We batch all observations and then repeat
    # these observations for the number of trajectories
    obs = torch.stack([c["x_k"] for c in curves], 0)  # (B,T) or (B,Q,T)
    rep = [1] * obs.dim()
    rep[0] = cfg["samples"]
    obs = obs.repeat(*rep).to(dev)

    # Generate. Note, we leave the last obs out so the first item predicted
    # overlaps the last observation and we hence get a smooth plot
    if cfg["fast_wavenet"]:
        g = generators.generate_fast(
            model=model,
            initial_obs=obs[..., :R],
            sampler=sampler,
        )
    else:
        g = generators.generate(
            model=model,
            initial_obs=obs[..., :R],
            sampler=sampler,
        )

    t0 = time.time()

    trajs, _ = generators.slice_generator(g, stop=steps)  # (B,1,T)
    trajs = trajs.squeeze(1).cpu()
    tn = torch.arange(R, R + steps, 1) * dm.dt
    tx = torch.arange(0, R + steps, 1) * dm.dt
    print(f"Generation took {(time.time()-t0):.3f} secs")

    # Plot
    for idx, (ax, s) in enumerate(zip(grid, curves)):
        ed = signal.EncoderDecoder(
            s["encode.num_levels"],
            s["encode.input_range"],
            s["encode.bin_shift"],
            s["encode.one_hot"],
        )
        if "x" in s:
            ax.plot(
                tx,
                s["x"][..., : R + steps],
                c="k",
                linewidth=0.5,
                label="input",
                linestyle="--",
                alpha=0.3,
            )
        else:
            ax.step(
                tx,
                ed.decode(s["x_k"])[..., : R + steps],
                c="k",
                linewidth=0.5,
                label="input",
                linestyle="--",
                alpha=0.3,
            )

        for tidx, xn in enumerate(trajs[idx::num_curves]):
            # Note, above we step with num_curves to get all trajectories
            # for this axis. Related to repeat statement above.
            ax.step(tn, ed.decode(xn), label="generated" if tidx == 0 else "")
        r = s["encode.input_range"]
        ax.set_ylim(r[0], r[1])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(f"tmp/generate_{type(model).__name__}.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # python -m autoregressive.examples.generate --config config.yaml C:\dev\autoregressive\lightning_logs\version_45\checkpoints\wavenet-epoch=28-val_loss=0.0002.ckpt
    main()
