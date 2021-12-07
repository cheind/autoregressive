__all__ = ["MNISTSeriesDataset", "MNISTDataModule", "peano_map", "peano_inv_map"]

import logging
import warnings
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from . import series_dataset as sd
from .. import encoding

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class MNISTSeriesDataset(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[[sd.SeriesMeta], tuple[sd.SeriesMeta]] = None,
        binarize: bool = False,
        apply_peano_map: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(
            root, train=train, download=download, transform=None, target_transform=None
        )
        self.series_transform = transform
        self.binarize = binarize
        self.apply_peano_map = apply_peano_map

    def __getitem__(self, index: int) -> sd.SeriesMeta:
        x, y = super().__getitem__(index)
        x = np.asarray(x)
        x = torch.tensor(x)
        if self.binarize:
            x = (x > 127).int()
        if self.apply_peano_map:
            x = peano_map(x)
        else:
            x = x.view(-1)
        series = {"x": x.long()}
        meta = {"digit": torch.tensor(y)}
        if self.series_transform is not None:
            series, meta = self.series_transform((series, meta))
        return series, meta


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        binarize: bool = True,
        apply_peano_map: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        digit_conditioning: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.quantization_levels = 2 if binarize else 256
        self.binarize = binarize
        self.c = digit_conditioning
        self.num_workers = num_workers
        self.batch_size = batch_size

        if digit_conditioning:
            _logger.info("Added digit conditioning: 10 condition channels required")
            transform = add_digit_conditioning
        else:
            transform = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = MNISTSeriesDataset(
                "./tmp",
                train=True,
                download=True,
                binarize=binarize,
                apply_peano_map=apply_peano_map,
                transform=transform,
            )
            if seed:
                g = torch.Generator().manual_seed(seed)
            else:
                g = torch.default_generator
            self.train_ds, self.val_ds = random_split(data, [55000, 5000], generator=g)
            self.test_ds = MNISTSeriesDataset(
                "./tmp",
                train=False,
                download=True,
                binarize=binarize,
                apply_peano_map=apply_peano_map,
                transform=transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=sd.series_collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sd.series_collate,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=sd.series_collate,
        )


def _make_peano_mnist_ids():
    ids = np.arange(0, 28 * 28, 1).reshape((28, 28))
    ids[1::2] = ids[1::2, ::-1]
    return torch.tensor(ids.copy()).view(-1).long()


PEANO_PERM_IDS = _make_peano_mnist_ids()


def peano_map(x: torch.Tensor):
    """Unroll image to 1D using Peano-curve.
    Every pixel that is adjacent in the result is also adjacent in space. However,
    for a pixel in the center column, its north neighbor is width pixels away.
    The distance to north neighbors varies accross pixel columns.
    """
    return x.view(-1)[PEANO_PERM_IDS]


def peano_inv_map(x: torch.Tensor):
    return x[PEANO_PERM_IDS].reshape(28, 28)


def add_digit_conditioning(sm: sd.SeriesMeta) -> sd.SeriesMeta:
    series, meta = sm
    d = F.one_hot(meta["digit"], num_classes=10).view(-1, 1)  # (C,1)
    series["c"] = d.float()
    return series, meta


POSENC = encoding.positional_encoding_lut(length=28 * 28, depth=64)


def add_pos_conditioning(sm: sd.SeriesMeta) -> sd.SeriesMeta:
    series, meta = sm
    series["c"] = POSENC
    return series, meta


def demo_peano():
    import matplotlib.pyplot as plt

    ds = MNIST("./tmp", train=True, download=True)
    x, y = ds[0]
    x = np.asarray(x)
    x = torch.tensor(x)
    fx = peano_map(x)
    rx = peano_inv_map(fx)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x)
    axs[1].imshow(fx.view(28, 28))
    axs[2].imshow(rx.view(28, 28))
    plt.show()


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1 import ImageGrid

    # demo_peano()
    # print(_make_peano_mnist_ids())
    dm = MNISTDataModule(digit_conditioning=True, binarize=True)
    series, meta = dm.train_ds[0]
    plt.imshow(peano_inv_map(series["x"]))
    plt.show()


if __name__ == "__main__":
    main()
