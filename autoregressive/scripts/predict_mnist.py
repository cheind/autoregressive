import jsonargparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch._C import device
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl

from torchvision.utils import make_grid
import torchvision.transforms.functional as tvf

from .. import wave, generators, sampling
from ..datasets import MNISTDataModule
from .forecast import InstantiateOnlyLightningCLI, load_model, create_fig

import matplotlib.pyplot as plt


class GenerateDigitsCommand:
    def __init__(self, ckpt: str = None, num_samples_per_digit: int = 10) -> None:
        self.num_samples_per_digit = num_samples_per_digit
        self.ckpt = ckpt

    @torch.no_grad()
    def run(self, dev: torch.device):
        model = wave.WaveNet.load_from_checkpoint(self.ckpt).to(dev).eval()
        seeds = torch.zeros(
            (10, self.num_samples_per_digit), dtype=torch.long, device=dev
        ).view(-1, 1)
        targets = torch.arange(0, 10, 1).repeat(self.num_samples_per_digit)
        targets = F.one_hot(targets, num_classes=10).unsqueeze(-1).to(dev).float()

        g = generators.generate_fast(
            model=model,
            initial_obs=seeds,
            sampler=sampling.sample_stochastic,
            global_cond=targets,
        )
        digits, _ = generators.slice_generator(g, stop=(28 * 28 - 1))  # (B,784)
        digits = torch.cat((seeds, digits), 1).view(-1, 1, 28, 28)  # (B,C,H,W)
        grid = make_grid(digits, nrow=10)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(grid.cpu().float().permute(1, 2, 0)[..., 0], cmap="viridis")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.savefig("tmp/generate_digits.png", bbox_inches="tight")
        plt.show()


@torch.no_grad()
def main():
    parser = jsonargparse.ArgumentParser("Predict MNIST")
    generate_parser = jsonargparse.ArgumentParser()
    generate_parser.add_class_arguments(GenerateDigitsCommand, None)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("generate", generate_parser)
    config = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config.subcommand == "generate":
        cmd = GenerateDigitsCommand(**(config.generate.as_dict()))

    cmd.run(dev)

    # seeds = torch.randint(0, model.quantization_levels, size=(cfg["num_curves"], 1))
    # if cfg["seed_center"]:
    #     seeds.fill_(model.quantization_levels // 2)
    # seeds = seeds.to(dev)

    # horizon = cfg["horizon"] or model.receptive_field * 2
    # t = torch.arange(0, horizon + 1, 1) * dm.train_params.dt

    # global_cond = get_condition(cli, model)
    # if global_cond is not None:
    #     global_cond = global_cond.to(dev)
    # print("using condition", global_cond.view(-1))

    # # Generate. Note, we leave the last obs out so the first item predicted
    # # overlaps the last observation and we hence get a smooth plot
    # if cfg["fast_wavenet"]:
    #     g = generators.generate_fast(
    #         model=model,
    #         initial_obs=seeds,
    #         sampler=sampling.sample_stochastic,
    #         global_cond=global_cond,
    #     )
    # else:
    #     g = generators.generate(
    #         model=model,
    #         initial_obs=seeds,
    #         sampler=sampling.sample_stochastic,
    #         global_cond=global_cond,
    #     )
    # t0 = time.time()
    # trajs, _ = generators.slice_generator(g, stop=horizon)  # (B,T)
    # trajs = torch.cat((seeds, trajs), 1)
    # print(f"Generation took {(time.time()-t0):.3f} secs")

    # # Plot
    # fig, grid = create_fig(num_curves=1, figsize=(8, 3))
    # for traj in trajs:
    #     grid[0].step(t, traj.cpu(), alpha=0.8)
    # grid[0].set_ylim(0, model.quantization_levels)
    # grid[0].set_xlabel("Timestep")
    # grid[0].set_ylabel("Quantization Level")
    # fig.tight_layout()
    # fig.savefig("tmp/prior_samples.svg", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # python -m autoregressive.scripts.generate --config models\fseries\config.yaml "models\fseries\wavenet-epoch=16-val_loss_epoch=4.9223.ckpt" # noqa:E501
    main()
