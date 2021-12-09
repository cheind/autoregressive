import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision.utils import make_grid
from matplotlib.gridspec import GridSpec

from .. import datasets, generators, sampling, wave
from . import wavenet_signals


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
def compute_log_pxy(
    imgs: torch.Tensor, model: wave.WaveNet, horizon: int = None, tau: float = 1.0
):
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
    horizon = horizon or (T - 1)
    horizon = min(horizon, (T - 1))
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
        log_px_y = F.log_softmax(logits / tau, 1)
        log_px_y = log_px_y[
            ..., :-1
        ]  # TODO document what's happening here: predictions vs input
        log_px_y = log_px_y.permute(0, 2, 1).reshape(-1, Q)  # (B*T,Q)
        log_px_y = log_px_y[torch.arange(B * (T - 1)), imgs[:, 1:].reshape(-1)].view(
            B, (T - 1)
        )  # (B,T)
        log_pxy = log_px_y[:, :horizon].sum(-1) + log_py
        log_pxys.append(log_pxy)
        del logits
    return torch.stack(log_pxys, -1)


class SampleImagesCommand(wavenet_signals.BaseCommand):
    """Samples from the condition distribution p(x|digit) where x is an mnist image."""

    def __init__(
        self,
        ckpt: str,
        data: datasets.MNISTDataModule = None,
        num_samples_per_digit: int = 10,
        img_shape: str = "28x28",
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: MNIST data module
            num_samples_per_digit: number of images per digit category to generate
            img_shape: WxH shape of images to generate
        """
        del data
        self.num_samples_per_digit = num_samples_per_digit
        self.img_shape = list(map(int, img_shape.split("x")))
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self):
        dev = self.default_device
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
        fig.savefig("tmp/mnist-sample.png", bbox_inches="tight")
        plt.show()


class PredictImagesCommand(wavenet_signals.BaseCommand):
    """Reconstructs partial MNIST images."""

    def __init__(
        self,
        ckpt: str,
        data: datasets.MNISTDataModule,
        num_images: int = 7,
        num_samples_per_digit: int = 6,
        num_pix_observed: int = 392,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: MNIST data module
            num_images: Number of images to predict for
            num_samples_per_digit: Number of samples (predictions) per digit
            num_pix_observed: Number of pixels (scanline order) of input image to be considered observed by the network
        """
        self.num_images = num_images
        self.num_pix_observed = num_pix_observed
        self.num_samples_per_digit = num_samples_per_digit
        self.data = data
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model = self.model.to(dev)

        imgs, targets = load_images_targets(self.data, self.num_images)
        imgs = torch.stack(imgs, 0).to(dev)
        targets = torch.stack(targets, 0)
        targets = F.one_hot(targets, num_classes=10).unsqueeze(-1).float().to(dev)

        if self.num_samples_per_digit > 1:
            imgs = imgs.repeat(self.num_samples_per_digit, 1)
            targets = targets.repeat(self.num_samples_per_digit, 1, 1)

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

        # top-row: orig, other rows: prediction
        gs = GridSpec(2, 1, height_ratios=[1, self.num_samples_per_digit])
        grid_orig = make_grid(
            imgs[: self.num_images].view(-1, 1, 28, 28), nrow=self.num_images, padding=0
        )
        grid_pred = make_grid(digits, nrow=self.num_images, padding=0)
        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, :])
        ax0.set_ylabel("orig")
        ax1.set_ylabel("pred")
        ax0.set(yticklabels=[], xticklabels=[], xticks=[], yticks=[])
        ax1.set(yticklabels=[], xticklabels=[], xticks=[], yticks=[])
        ax0.imshow(grid_orig.cpu().float().permute(1, 2, 0)[..., 0])
        ax1.imshow(grid_pred.cpu().float().permute(1, 2, 0)[..., 0])
        fig.savefig("tmp/mnist-predict.svg", bbox_inches="tight")
        plt.show()


class DensityEstimationCommand(wavenet_signals.BaseCommand):
    """Estimates marginal log p(X=x), assuming p(Y=y)=1/|Y|"""

    def __init__(
        self,
        ckpt: str,
        data: datasets.MNISTDataModule,
        num_images: int = 5,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: MNIST data module
            num_images: Number of images to generate
        """
        del kwargs
        self.num_images = num_images
        self.data = data
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model = self.model.to(dev)
        Q = self.model.quantization_levels

        imgs, _ = load_images_targets(self.data, self.num_images)
        imgs = torch.stack(imgs, 0).to(dev)
        log_pxys = compute_log_pxy(imgs, model)
        log_px = torch.logsumexp(log_pxys, -1)
        print("log p(x) x~p(MNIST)", log_px)

        imgs_rand = torch.randint(0, Q, (5, 784)).to(dev)
        log_pxys = compute_log_pxy(imgs_rand, model)
        log_px = torch.logsumexp(log_pxys, -1)
        print(f"log p(x) x~U(0,{Q})", log_px)


class ClassificationCommand(wavenet_signals.BaseCommand):
    """Estimates p(Y=y|X=x)"""

    def __init__(
        self,
        ckpt: str,
        data: datasets.MNISTDataModule,
        hist_on_error: bool = True,
        show_hist: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            config: Path to config.yaml to read datamodule configuration from
            num_images: Number of images to generate
        """
        self.data = data
        self.data.batch_size = 5
        self.show_hist = show_hist
        self.hist_on_error = hist_on_error
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model = self.model.to(dev)
        dl = self.data.test_dataloader()

        all_preds = []
        all_targets = []
        all_probs = []
        for batchidx, batch in enumerate(dl):
            py_x, targets = self._compute_batch_results(batch, model, dev)
            acc = (py_x.argmax(1) == targets).sum()
            if (self.hist_on_error and acc != py_x.shape[0]) or self.show_hist:
                self.plot_hist(batchidx, batch, py_x, show=self.show_hist)
            all_probs.append(py_x)
            all_preds.append(py_x.argmax(1))
            all_targets.append(targets)
            if batchidx % 50 == 0:
                print(f"batch {batchidx}/{len(dl)}")

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs, 0)

        print(
            f"avg. accuracy {(all_preds==all_targets).sum()/all_targets.shape[0]:.4f}"
        )
        np.savez(
            "tmp/mnist-classify.npz",
            preds=all_preds.cpu().numpy(),
            targets=all_targets.cpu().numpy(),
            probs=all_probs.cpu().numpy(),
        )

    def plot_hist(self, batch_idx, batch, py_x: torch.Tensor, show: bool):
        N = self.data.batch_size
        imgs = batch[0]["x"]
        fig, axs = plt.subplots(N, 2, sharex="col", sharey="col", figsize=(3, 4))
        for i in range(N):
            axs[i, 0].imshow(imgs[i].view(28, 28))
            axs[i, 1].bar(torch.arange(0, 10, 1).float(), py_x[i].cpu(), width=0.9)
            axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[N - 1, 1].set(
            xticks=torch.arange(0, 10, 1),
            xticklabels=[str(i) for i in range(10)],
        )
        axs[0, 0].set_title("Input")
        axs[0, 1].set_title("Prediction")
        plt.tight_layout()
        fig.savefig(f"tmp/mnist-classify-{batch_idx:03}.svg")
        if show:
            plt.show()

    def _compute_batch_results(self, batch, model: wave.WaveNet, dev: torch.device):
        series, meta = batch
        imgs = series["x"].to(dev)
        targets = torch.tensor([m["digit"] for m in meta]).to(dev)

        log_pxys = compute_log_pxy(imgs, model)
        log_px = torch.logsumexp(log_pxys, -1)
        py_x = torch.exp(log_pxys - log_px.unsqueeze(-1))
        return py_x, targets


class ProgressiveClassificationCommand(wavenet_signals.BaseCommand):
    """Estimates p(Y=y|X=x) progressively, by observing more pixels in each iteration."""

    def __init__(
        self,
        ckpt: str,
        data: datasets.MNISTDataModule,
        show_hist: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            ckpt: Path to model parameters
            data: MNIST data module
            num_images: Number of images to generate
            show_hist: interactive display of histogram
        """
        self.data = data
        self.data.batch_size = 5
        self.model = wave.WaveNet.load_from_checkpoint(ckpt).eval()
        self.show_hist = show_hist

    @torch.no_grad()
    def run(self):
        dev = self.default_device
        model = self.model.to(dev)
        dl = self.data.test_dataloader()

        for batchidx, batch in enumerate(dl):
            for h in range(1, 28 * 28 - 1, 28):
                series, meta = batch
                imgs = series["x"].to(dev)
                log_pxys = compute_log_pxy(imgs, model, horizon=h, tau=1.0)
                log_px = torch.logsumexp(log_pxys, -1)
                py_x = torch.exp(log_pxys - log_px.unsqueeze(-1))
                self.plot_hist(batch, batchidx, py_x, h)
            print("generated", batchidx)

    def plot_hist(self, batch, batchidx, py_x: torch.Tensor, horizon: int):
        imgs = batch[0]["x"]
        B = imgs.shape[0]
        fig, axs = plt.subplots(B, 2, sharey="col", figsize=(3, 4))
        mask = torch.zeros((28, 28, 4), dtype=torch.uint8).view(-1, 4)
        mask[horizon:, 3] = 255

        for i in range(B):
            axs[i, 0].imshow(imgs[i].view(28, 28))
            axs[i, 0].imshow(mask.view(28, 28, 4))
            axs[i, 1].bar(torch.arange(0, 10, 1).float(), py_x[i].cpu(), width=0.9)
            axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[i, 1].set_ylim(0, 1)
            axs[i, 1].set(
                xticks=torch.arange(0, 10, 1),
                xticklabels=[str(i) for i in range(10)],
            )
        axs[0, 0].set_title("Input")
        axs[0, 1].set_title("Prediction")
        plt.tight_layout()
        fig.savefig(f"tmp/mnist-classify-progressive-{batchidx:03}-h{horizon:03}.png")
        if self.show_hist:
            plt.show()
        plt.close(fig)


@torch.no_grad()
def main():

    command_map = {
        "sample": SampleImagesCommand,
        "predict": PredictImagesCommand,
        "density": DensityEstimationCommand,
        "classify": ClassificationCommand,
        "progressive": ProgressiveClassificationCommand,
    }

    parser = jsonargparse.ArgumentParser("WaveNet on MNIST")
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
