from typing import Any, Dict, Union
import io
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import pytorch_lightning as pl


class ElectricityDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Union[str, Path]) -> None:
        data_path = Path(data_path)
        assert data_path.is_file()

        with open(data_path, "r") as f:
            raw = f.read().replace(":", ",")
        data = np.genfromtxt(
            io.StringIO(raw), skip_header=17, delimiter=",", dtype=np.float32
        )
        self.data = data[:, 2:]  # remove date columns
        print(self.data.shape)
        # https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5


if __name__ == "__main__":
    ds = ElectricityDataset(r"D:\timeseries\electricity_hourly_dataset_short.tsf")
