import logging

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import torch
import torch.utils.data as data
import torch.nn.functional as F

from . import wave, dataset

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


def _exp_weights(T, n):
    """Returns exponential decaying weights for T terms from 1.0 down to n (n>0)"""
    n = torch.as_tensor(n).float()
    x = torch.arange(0, T, 1)
    b = -torch.log(n) / (T - 1)
    return torch.exp(-x * b)


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.fseries_train, self.fseries_val = dataset.create_default_datasets()
        self.batch_size = batch_size

    def train_dataloader(self):
        return data.DataLoader(
            self.fseries_train, batch_size=self.batch_size, num_workers=4
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.fseries_val, batch_size=self.batch_size, num_workers=4
        )

    # def test_dataloader(self):
    #     return data.DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return data.DataLoader(self.mnist_test, batch_size=self.batch_size)


class LitRegressionWaveNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        forecast_steps: int = 1,
        residual_channels: int = 64,
        skip_channels: int = 64,
        num_blocks: int = 1,
        num_layers_per_block: int = 9,
        train_full_receptive_field: bool = True,
        train_exp_decay: bool = False,
    ) -> None:
        super().__init__()
        self.wavenet = wave.RegressionWaveNet(
            in_channels=in_channels,
            forecast_steps=forecast_steps,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
        )
        self.train_full_receptive_field = train_full_receptive_field
        self.receptive_field = self.wavenet.receptive_field
        self.train_exp_decay = train_exp_decay
        if self.train_exp_decay and forecast_steps > 1:
            self.register_buffer(
                "l1_weights", _exp_weights(forecast_steps, 1e-3).view(1, -1, 1)
            )
        else:
            self.l1_weights = 1.0
        _logger.info(f"Receptive field of model {self.wavenet.receptive_field}")
        super().save_hyperparameters()

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=1, min_lr=1e-7, threshold=1e-7
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _step(self, batch, batch_idx) -> torch.FloatTensor:
        del batch_idx
        x = batch["x"][..., :-1].unsqueeze(1)
        y = batch["xo"][..., 1:]
        y = y.unfold(-1, self.wavenet.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        yhat = self.wavenet(x)
        r = self.receptive_field if self.train_full_receptive_field else 0
        losses = F.l1_loss(yhat[..., r:n], y[..., r:], reduction="none")
        return torch.mean(losses * self.l1_weights)


# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/autoencoder.py


def cli_main():
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        filename="regressionwavenet-{epoch:02d}-{val_loss:.4f}",
    )
    lrm = LearningRateMonitor(logging_interval="step")

    num_train_curves: int = 2 ** 12
    num_val_curves: int = 2 ** 9
    cli = LightningCLI(
        LitRegressionWaveNet,
        FSeriesDataModule,
        # seed_everything_default=1234,
        # run=False,
        save_config_overwrite=True,
        trainer_defaults={
            "callbacks": [ckpt, lrm],
            "max_epochs": 30,
            "gpus": 1,
            "limit_train_batches": num_train_curves // 64,
            "limit_val_batches": num_val_curves // 64,
            "log_every_n_steps": 25,
        },
    )
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule) # use when run=False is avail. 1.5.0
    _logger.info(f"Best val model path: {ckpt.best_model_path}")


if __name__ == "__main__":
    cli_main()
