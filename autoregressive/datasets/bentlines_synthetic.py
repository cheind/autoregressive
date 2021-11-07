__all__ = [
    "BentLinesDataset",
    "BentLinesDataModule",
]

from typing import Any, Callable, Dict, Tuple
import dataclasses

import torch
import torch.utils.data
import pytorch_lightning as pl

from .common import SeriesDataset, Sample
from . import transforms


@dataclasses.dataclass
class BentLinesParams:
    num_curves: int = 2048
    num_tsamples: int = 32
    dt: float = 1.0
    seed: int = None
    include_params: bool = False


class BentLinesDataset(SeriesDataset):
    """Two straight lines with random slopes, connected to each other at a random location
    https://fleuret.org/dlc/materials/dlc-handout-10-1-autoregression.pdf
    """

    def __init__(
        self,
        params: BentLinesParams = BentLinesParams(),
        transform: Callable[[Sample], Sample] = None,
    ) -> None:
        self.params = params
        self.transform = transform
        if params.seed is None:
            rng = torch.default_generator
        else:
            rng = torch.Generator().manual_seed(params.seed)
        self.curve_params = [self._sample_params(rng) for _ in range(params.num_curves)]

    @property
    def dt(self):
        return self.params.dt

    def __len__(self):
        return self.params.num_curves

    def __getitem__(self, index) -> Sample:
        p = self.curve_params[index]
        t = torch.arange(0, self.params.num_tsamples) * self.dt
        tb = p["tbent"]
        x1 = t[:tb] * p["slopes"][0] + p["d"]
        x2 = (t[tb:] - t[tb]) * p["slopes"][1] + p["d"]
        x = torch.cat((x1, x2))
        sample = {"x": x, "xo": x.clone(), "t": t}
        if self.params.include_params:
            sample["p"] = p
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _sample_params(self, g: torch.Generator) -> Dict[str, Any]:
        """Returns sampled parameters."""

        def uniform(r, n: int):
            return (r[1] - r[0]) * torch.rand(n, generator=g) + r[0]

        tbent = torch.randint(0, self.params.num_tsamples, (1,), generator=g)
        slopes = uniform((-1.0, 1.0), 2)
        d = 0.0

        return {"tbent": tbent, "slopes": slopes, "d": d}


class BentLinesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_params: BentLinesParams = BentLinesParams(),
        val_params: BentLinesParams = BentLinesParams(num_curves=256),
        quantization_levels: int = 256,
        quantization_range: Tuple[float, float] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        if quantization_range is None:
            train_ds = BentLinesDataset(train_params)
            val_ds = BentLinesDataset(val_params)
            quantization_range = transforms.Normalize.find_range(train_ds, val_ds)
        transform = transforms.chain_transforms(
            transforms.Normalize(quantization_range, (0.0, 1.0)),
            transforms.Quantize(num_bins=quantization_levels),
        )
        self.train_ds = BentLinesDataset(train_params, transform=transform)
        self.val_ds = BentLinesDataset(val_params, transform=transform)
        self.train__params = train_params
        self.val_params = val_params
        self.quantization_levels = quantization_levels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dt = self.train_ds.dt
        self.save_hyperparameters()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def __str__(self) -> str:
        return f"BentLinesDataModule(train_params={self.train__params}, val_params={self.val_params})"


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    # dm = FSeriesDataModule(
    #     train_fseries_params=FSeriesParams(smoothness=0.75), batch_size=512
    # )
    dm = BentLinesDataModule()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(5, 5), axes_pad=0.05, share_all=False, aspect=False
    )

    for ax, s in zip(grid, dm.train_ds):
        # ax.step(s["t"], s["x"])
        ax.scatter(s["t"], s["x"], s=1.0)
        ax.set_ylim(0, 1.0)
    plt.show()


if __name__ == "__main__":
    main()
