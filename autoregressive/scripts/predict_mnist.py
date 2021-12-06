import jsonargparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import yaml
from torchvision.utils import make_grid

from .. import datasets, generators, sampling, wave


class SampleDigitsCommand:
    """Samples from the condition distribution p(x|digit) where x is an mnist image."""

    def __init__(
        self,
        ckpt: str,
        num_samples_per_digit: int = 10,
        img_shape: str = "28x28",
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            num_samples_per_digit: number of images per digit category to generate
            img_shape: WxH shape of images to generate
        """
        self.num_samples_per_digit = num_samples_per_digit
        self.img_shape = list(map(int, img_shape.split("x")))
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self, dev: torch.device):
        model: wave.WaveNet = self.model.to(dev).eval()
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
        grid = make_grid(
            digits, nrow=10, value_range=[0, model.quantization_levels - 1]
        )

        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        ax.imshow(grid.cpu().float().permute(1, 2, 0)[..., 0])
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.savefig("tmp/sample_digits.png", bbox_inches="tight")
        plt.show()

    @staticmethod
    def add_arguments_to_parser(outer_parser):
        parser = jsonargparse.ArgumentParser()
        parser.add_class_arguments(SampleDigitsCommand, None)
        outer_parser.add_subcommand("sample", parser)


class InfillDigitsCommand:
    """Reconstructs partial MNIST images."""

    def __init__(
        self,
        ckpt: str,
        config: str,
        num_images: int = 25,
        num_pix_observed: int = 392,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            config: Path to config.yaml to read datamodule configuration from
            num_images: Number of images to generate
            num_pix_observed: Number of pixels (scanline order) of input image to be considered observed by the network
        """
        self.num_images = num_images
        self.num_pix_observed = num_pix_observed
        with open(config, "r") as f:
            plcfg = yaml.safe_load(f.read())
        self.data = datasets.MNISTDataModule(**plcfg["data"]["init_args"])
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    def _load_images_targets(self):
        ds = self.data.test_dataloader().dataset
        ids = torch.randint(0, len(ds), (self.num_images,))

        imgs = []
        targets = []

        for idx in ids:
            sm = ds[idx]
            imgs.append(sm[0]["x"])
            targets.append(sm[1]["digit"])

        return imgs, targets

    @torch.no_grad()
    def run(self, dev: torch.device):
        model = self.model.to(dev)

        imgs, targets = self._load_images_targets()
        imgs = torch.stack(imgs, 0).to(dev)
        targets = torch.stack(targets, 0)
        targets = F.one_hot(targets, num_classes=10).unsqueeze(-1).float().to(dev)

        g = generators.generate_fast(
            model=model,
            initial_obs=imgs[..., : self.num_pix_observed],
            sampler=sampling.sample_stochastic,
            global_cond=targets,
        )
        digits, _ = generators.slice_generator(
            g, stop=(28 * 28 - self.num_pix_observed)
        )  # (B,784)
        digits = torch.cat((imgs[..., : self.num_pix_observed], digits), 1).view(
            -1, 1, 28, 28
        )  # (B,C,H,W)
        grid = make_grid(digits, nrow=5)
        gridorig = make_grid(imgs.view(-1, 1, 28, 28), nrow=5)

        fig = plt.figure(figsize=(8, 8), frameon=False)
        axs = (plt.subplot(121), plt.subplot(122))
        axs[0].imshow(gridorig.cpu().float().permute(1, 2, 0)[..., 0])
        axs[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[1].imshow(grid.cpu().float().permute(1, 2, 0)[..., 0])
        axs[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.savefig("tmp/infill_digits.png", bbox_inches="tight")
        plt.show()

    @staticmethod
    def add_arguments_to_parser(outer_parser):
        parser = jsonargparse.ArgumentParser()
        parser.add_class_arguments(InfillDigitsCommand, None)
        outer_parser.add_subcommand("infill", parser)


@torch.no_grad()
def main():
    parser = jsonargparse.ArgumentParser("WaveNet on MNIST")
    subcommands = parser.add_subcommands()
    SampleDigitsCommand.add_arguments_to_parser(subcommands)
    InfillDigitsCommand.add_arguments_to_parser(subcommands)
    config = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config.subcommand == "sample":
        cmd = SampleDigitsCommand(**(config.sample.as_dict()))
    elif config.subcommand == "infill":
        cmd = InfillDigitsCommand(**(config.infill.as_dict()))

    cmd.run(dev)


if __name__ == "__main__":
    # python -m autoregressive.scripts.predict_mnist sample --ckpt "v56\checkpoints\wavenet-epoch=06-val_acc_epoch=0.9705.ckpt"
    main()
