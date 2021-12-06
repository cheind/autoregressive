import jsonargparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import yaml
from torchvision.utils import make_grid

from .. import datasets, generators, sampling, wave


def load_images_targets(data: datasets.MNISTDataModule, n: int, seed: int = None):
    ds = data.test_dataloader().dataset
    g = torch.default_generator
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, len(ds), (n,), generator=g)

    imgs = []
    targets = []

    for idx in ids:
        sm = ds[idx]
        imgs.append(sm[0]["x"])
        targets.append(sm[1]["digit"])

    return imgs, targets


@torch.no_grad()
def compute_log_pxy(imgs: torch.Tensor, model: wave.WaveNet):
    """Computes p(X=x|Y) for each possible value of y as a BxQ tensor"""
    # log p(X=x) = log sum_y p(X=x|Y=y)p(Y=y)
    #            = log sum_y exp(log p(X=x|Y=y)p(Y=y))
    #            = log sum_y exp(log p(X=x|Y=y) + log p(Y=y))
    #            = log sum_y exp(log p(X0|Y=y)p(X1|X0,Y=y)...p(XT|XT-1...X0,Y=y) + log p(Y=y))
    #            = log sum_y exp(log p(X0|Y=y)+log p(X1|X0,Y=y)+...+log p(XT|XT-1...X0,Y=y) + log p(Y=y))
    # this can be conveniently computed using torch.logsumexp
    Y = model.conditioning_channels
    Q = model.quantization_levels
    B, T = imgs.shape
    log_py = torch.log(torch.tensor(1.0 / 10))

    y_conds = (
        (F.one_hot(torch.arange(0, Y, 1), num_classes=Y).permute(0, 1).view(Y, Y, 1))
        .to(imgs.device)
        .float()
    )

    # imgs is (B,T)
    log_pxys = []
    for y in range(10):
        logits, _ = model.forward(x=imgs, c=y_conds[y : y + 1])  # (B,Q,T)
        log_px_y = F.log_softmax(logits, 1)
        # plt.imshow(torch.exp(log_px_y)[0, 0, :].cpu().view(28, 28))
        # plt.show()
        log_px_y = log_px_y[
            ..., :-1
        ]  # TODO document what's happening here: predictions vs input
        log_px_y = log_px_y.permute(0, 2, 1).reshape(-1, Q)  # (B*T,Q)
        log_px_y = log_px_y[torch.arange(B * (T - 1)), imgs[:, 1:].reshape(-1)].view(
            B, (T - 1)
        )  # (B,T)
        log_pxy = log_px_y.sum(-1) + log_py
        log_pxys.append(log_pxy)
        del logits
    return torch.stack(log_pxys, -1)


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

        cmap = None  # "gray_r"
        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        ax.imshow(grid.cpu().float().permute(1, 2, 0)[..., 0], cmap=cmap)
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

    @torch.no_grad()
    def run(self, dev: torch.device):
        model = self.model.to(dev)

        imgs, targets = load_images_targets(self.data, self.num_images)
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


class DensityEstimationCommand:
    """Estimates marginal log p(X=x), assuming p(Y=y)=1/|Y|"""

    def __init__(
        self,
        ckpt: str,
        config: str,
        num_images: int = 5,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            config: Path to config.yaml to read datamodule configuration from
            num_images: Number of images to generate
        """
        self.num_images = num_images
        with open(config, "r") as f:
            plcfg = yaml.safe_load(f.read())
        self.data = datasets.MNISTDataModule(**plcfg["data"]["init_args"])
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self, dev: torch.device):
        model = self.model.to(dev)

        imgs, _ = load_images_targets(self.data, self.num_images)
        imgs = torch.stack(imgs, 0).to(dev)
        log_pxys = compute_log_pxy(imgs, model)
        log_px = torch.logsumexp(log_pxys, -1)
        print("log p(x) x~p(MNIST)", log_px)

        imgs_rand = torch.randint(0, 256, (5, 784)).to(dev)
        log_pxys = compute_log_pxy(imgs_rand, model)
        log_px = torch.logsumexp(log_pxys, -1)
        print("log p(x) x~U(0,255)", log_px)

    @staticmethod
    def add_arguments_to_parser(outer_parser):
        parser = jsonargparse.ArgumentParser()
        parser.add_class_arguments(DensityEstimationCommand, None)
        outer_parser.add_subcommand("density", parser)


class ClassificationCommand:
    """Estimates marginal log p(Y=y|X=x), assuming p(Y=y)=1/|Y|"""

    def __init__(
        self,
        ckpt: str,
        config: str,
        num_images: int = 5,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            config: Path to config.yaml to read datamodule configuration from
            num_images: Number of images to generate
        """
        self.num_images = num_images
        with open(config, "r") as f:
            plcfg = yaml.safe_load(f.read())
        self.data = datasets.MNISTDataModule(**plcfg["data"]["init_args"])
        self.data.batch_size = 10
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self, dev: torch.device):
        self.plot_hist(dev)
        # model = self.model.to(dev)

        # dl = self.data.train_dataloader()

        # series, meta = next(iter(dl))
        # imgs = series["x"].to(dev)
        # targets = torch.tensor([m["digit"] for m in meta]).to(dev)

        # log_pxys = compute_log_pxy(imgs, model)
        # log_px = torch.logsumexp(log_pxys, -1)
        # py_x = torch.exp(log_pxys - log_px.unsqueeze(-1))

    def plot_hist(self, dev: torch.device):
        model = self.model.to(dev)

        dl = self.data.test_dataloader()

        batch = next(iter(dl))
        imgs = batch[0]["x"]
        py_x, targets = self._compute_batch_results(batch, model, dev)
        fig, axs = plt.subplots(self.data.batch_size, 2, sharex="col", sharey="col")
        for i in range(self.data.batch_size):
            axs[i, 0].imshow(imgs[i].view(28, 28))
            axs[i, 1].bar(torch.arange(0, 10, 1).float(), py_x[i].cpu())
            axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if i < self.data.batch_size - 1:
                axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            else:
                axs[i, 1].set(
                    xticks=torch.arange(0, 10, 1),
                    xticklabels=[str(i) for i in range(10)],
                )

        plt.subplots_adjust(wspace=0.1)
        plt.show()

    def _compute_batch_results(self, batch, model: wave.WaveNet, dev: torch.device):
        series, meta = batch
        imgs = series["x"].to(dev)
        targets = torch.tensor([m["digit"] for m in meta]).to(dev)

        log_pxys = compute_log_pxy(imgs, model)
        log_px = torch.logsumexp(log_pxys, -1)
        py_x = torch.exp(log_pxys - log_px.unsqueeze(-1))
        return py_x, targets

    @staticmethod
    def add_arguments_to_parser(outer_parser):
        parser = jsonargparse.ArgumentParser()
        parser.add_class_arguments(ClassificationCommand, None)
        outer_parser.add_subcommand("classify", parser)


@torch.no_grad()
def main():
    parser = jsonargparse.ArgumentParser("WaveNet on MNIST")
    subcommands = parser.add_subcommands()
    SampleDigitsCommand.add_arguments_to_parser(subcommands)
    InfillDigitsCommand.add_arguments_to_parser(subcommands)
    DensityEstimationCommand.add_arguments_to_parser(subcommands)
    ClassificationCommand.add_arguments_to_parser(subcommands)
    config = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config.subcommand == "sample":
        cmd = SampleDigitsCommand(**(config.sample.as_dict()))
    elif config.subcommand == "infill":
        cmd = InfillDigitsCommand(**(config.infill.as_dict()))
    elif config.subcommand == "density":
        cmd = DensityEstimationCommand(**(config.density.as_dict()))
    elif config.subcommand == "classify":
        cmd = ClassificationCommand(**(config.classify.as_dict()))

    cmd.run(dev)


if __name__ == "__main__":
    # python -m autoregressive.scripts.mnist sample --ckpt "v58\checkpoints\wavenet-epoch=13-val_acc_epoch=0.8960.ckpt"

    # python -m autoregressive.scripts.mnist infill --config v58\config.yaml --ckpt "v58\checkpoints\wavenet-epoch=13-val_acc_epoch=0.8960.ckpt"

    # python -m autoregressive.scripts.mnist density --config v58\config.yaml --ckpt "v58\checkpoints\wavenet-epoch=13-val_acc_epoch=0.8960.ckpt"

    # python -m autoregressive.scripts.mnist classify --config v58\config.yaml --ckpt "v58\checkpoints\wavenet-epoch=13-val_acc_epoch=0.8960.ckpt"

    main()
