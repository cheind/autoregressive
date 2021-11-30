import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl


from .. import wave, generators, sampling
from .forecast import InstantiateOnlyLightningCLI, load_model, create_fig


class GenerateLightningCLI(InstantiateOnlyLightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("-horizon", type=int, default=None)
        parser.add_argument("-num-curves", type=int, default=4)
        parser.add_argument("-condition", type=int, default=None)
        parser.add_argument(
            "--fast-wavenet", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--seed-center", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument("ckpt", type=str)
        return super().add_arguments_to_parser(parser)


def get_condition(cli, model: wave.WaveNet):
    cfg = cli.config
    if cfg["condition"] is None:
        return None
    else:
        return (
            F.one_hot(
                torch.tensor(cfg["condition"]), num_classes=model.conditioning_channels
            )
            .view(1, -1, 1)
            .float()
        )


@torch.no_grad()
def main():
    cli = GenerateLightningCLI(
        wave.WaveNet,
        pl.LightningDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
    )

    cfg = cli.config
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(cli).to(dev)
    dm = cli.datamodule
    seeds = torch.randint(0, model.quantization_levels, size=(cfg["num_curves"], 1))
    if cfg["seed_center"]:
        seeds.fill_(model.quantization_levels // 2)
    seeds = seeds.to(dev)

    horizon = cfg["horizon"] or model.receptive_field * 2
    t = torch.arange(0, horizon + 1, 1) * dm.train_params.dt

    global_cond = get_condition(cli, model)
    if global_cond is not None:
        global_cond = global_cond.to(dev)
    print("using condition", global_cond.view(-1))

    # Generate. Note, we leave the last obs out so the first item predicted
    # overlaps the last observation and we hence get a smooth plot
    if cfg["fast_wavenet"]:
        g = generators.generate_fast(
            model=model,
            initial_obs=seeds,
            sampler=sampling.sample_stochastic,
            global_cond=global_cond,
        )
    else:
        g = generators.generate(
            model=model,
            initial_obs=seeds,
            sampler=sampling.sample_stochastic,
            global_cond=global_cond,
        )
    t0 = time.time()
    trajs, _ = generators.slice_generator(g, stop=horizon)  # (B,T)
    trajs = torch.cat((seeds, trajs), 1)
    print(f"Generation took {(time.time()-t0):.3f} secs")

    # Plot
    fig, grid = create_fig(num_curves=1, figsize=(8, 3))
    for traj in trajs:
        grid[0].step(t, traj.cpu(), alpha=0.8)
    grid[0].set_ylim(0, model.quantization_levels)
    grid[0].set_xlabel("Timestep")
    grid[0].set_ylabel("Quantization Level")
    fig.tight_layout()
    fig.savefig("tmp/prior_samples.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # python -m autoregressive.scripts.generate --config models\fseries\config.yaml "models\fseries\wavenet-epoch=16-val_loss_epoch=4.9223.ckpt" # noqa:E501
    main()
