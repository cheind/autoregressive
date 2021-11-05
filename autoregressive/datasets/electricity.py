__all__ = ["AEPHourlyDataset", "AEPHourlyDataModule"]
import io
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from autoregressive.datasets import common

from . import functional
from .common import Sample


class ElectricityDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Union[str, Path], normalize: bool = True) -> None:
        data_path = Path(data_path)
        assert data_path.is_file()

        with open(data_path, "r") as f:
            raw = f.read().replace(":", ",")
        data = np.genfromtxt(
            io.StringIO(raw), skip_header=17, delimiter=",", dtype=np.float32
        )
        self.data = torch.from_numpy(data[:, 2:])  # remove date columns
        self.mean, self.std = functional.find_series_mean_std(self.data)
        self.normalize = normalize
        if normalize:
            self.data = functional.standardize_series(self.data, self.mean, self.std)
        self.t = torch.arange(0, self.data.shape[1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> Sample:
        return {"x": self.data[index], "t": self.t}

        print(self.data.shape)
        # https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5


class AEPHourlyDataset(torch.utils.data.Dataset):
    # https://www.kaggle.com/robikscube/hourly-energy-consumption
    def __init__(
        self,
        data_path: Union[str, Path],
        normalize: bool = True,
        chunk_size: int = None,
    ) -> None:
        data_path = Path(data_path)
        assert data_path.is_file()

        data = np.genfromtxt(
            data_path, skip_header=1, delimiter=",", usecols=(1,), dtype=np.float32
        )
        data = torch.from_numpy(data)
        self.source_range = functional.find_series_range([data])
        self.target_range = (-1.0, 1.0) if normalize else self.source_range
        data = functional.normalize_series(data, self.source_range, self.target_range)
        if chunk_size is None:
            chunk_size = len(data)
        self.data = data.unfold(0, chunk_size, chunk_size)
        self.dt = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> Sample:
        return {"x": self.data[index], "t": torch.arange(0, self.data.shape[-1])}

        print(self.data.shape)
        # https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5


class AEPHourlyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[str, Path] = "./AEP_hourly.csv",
        chunk_size: int = 8760 // 2,  # 365*24, i.e roughly a half-year
        normalize: bool = True,
        repeat_train: int = 100,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

        self.ds = AEPHourlyDataset(
            data_path=data_path, normalize=normalize, chunk_size=chunk_size
        )
        self.train_ds, self.val_ds, self.test_ds = common.split_ds_fractional(
            self.ds, [0.6, 0.2, 0.2]
        )
        if repeat_train > 1:
            self.train_ds = torch.utils.data.ConcatDataset(
                [self.train_ds] * repeat_train
            )

        self.dt = 1.0

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .. import measures

    dm = AEPHourlyDataModule(r"C:\data\AEP_hourly.csv", normalize=True, batch_size=64)
    b = next(iter(dm.val_dataloader()))
    print(measures.sample_entropy(b["x"]).mean())
