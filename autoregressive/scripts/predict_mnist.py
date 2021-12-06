import jsonargparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision.utils import make_grid

from .. import generators, sampling, wave


class SampleDigitsCommand:
    def __init__(
        self,
        ckpt: str = None,
        num_samples_per_digit: int = 10,
        img_shape: str = "28x28",
    ) -> None:
        self.num_samples_per_digit = num_samples_per_digit
        self.img_shape = list(map(int, img_shape.split("x")))
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
        digits, _ = generators.slice_generator(
            g, stop=(self.img_shape[0] * self.img_shape[1] - 1)
        )  # (B,784)
        digits = torch.cat((seeds, digits), 1).view(
            -1, 1, self.img_shape[0], self.img_shape[1]
        )  # (B,C,H,W)
        grid = make_grid(digits, nrow=10)

        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        ax.imshow(grid.cpu().float().permute(1, 2, 0)[..., 0], cmap="viridis")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.savefig("tmp/generate_digits.png", bbox_inches="tight")
        plt.show()


@torch.no_grad()
def main():
    parser = jsonargparse.ArgumentParser("WaveNet on MNIST")
    sample_parser = jsonargparse.ArgumentParser()
    sample_parser.add_class_arguments(SampleDigitsCommand, None)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("sample", sample_parser)
    config = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config.subcommand == "sample":
        cmd = SampleDigitsCommand(**(config.sample.as_dict()))

    cmd.run(dev)


if __name__ == "__main__":
    # python -m autoregressive.scripts.predict_mnist sample --ckpt "v56\checkpoints\wavenet-epoch=06-val_acc_epoch=0.9705.ckpt"
    main()
