import logging

import pytorch_lightning as pl
import torch.utils.data as data

from . import dataset, wave

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class FSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_train_curves: int = 2 ** 12,
        num_val_curves: int = 2 ** 9,
        num_workers: int = 0,
        batch_size: int = 64,
        train_seed: int = None,
        val_seed: int = None,
    ):
        super().__init__()
        self.fseries_train, self.fseries_val = dataset.create_default_datasets(
            num_train_curves, num_val_curves, train_seed, val_seed
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return data.DataLoader(
            self.fseries_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.fseries_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


# class LitBimodalWaveNet(pl.LightningModule):
#     def __init__(
#         self,
#         in_channels: int = 1,
#         residual_channels: int = 64,
#         skip_channels: int = 64,
#         num_blocks: int = 1,
#         num_layers_per_block: int = 9,
#         train_full_receptive_field: bool = True,
#     ) -> None:
#         super().__init__()
#         self.wavenet = wave.WaveNet(
#             in_channels=in_channels,
#             out_channels=(2 + 2 + 2),
#             residual_channels=residual_channels,
#             skip_channels=skip_channels,
#             num_blocks=num_blocks,
#             num_layers_per_block=num_layers_per_block,
#         )
#         self.train_full_receptive_field = train_full_receptive_field
#         self.receptive_field = self.wavenet.receptive_field
#         self.default_sampler = wave.bimodal_sampler
#         self.positive_scale = D.transform_to(constraints.greater_than(0.0))
#         _logger.info(f"Receptive field of model {self.wavenet.receptive_field}")
#         super().save_hyperparameters()

#     def configure_optimizers(self):
#         import torch.optim.lr_scheduler as sched

#         opt = torch.optim.Adam(self.parameters(), lr=1e-4)
#         return {
#             "optimizer": opt,
#             "lr_scheduler": {
#                 "scheduler": sched.ReduceLROnPlateau(
#                     opt, mode="min", factor=0.5, patience=1, min_lr=1e-7, threshold=1e-7
#                 ),
#                 "monitor": "val_loss",
#                 "interval": "epoch",
#                 "frequency": 1,
#             },
#         }

#     def training_step(self, batch, batch_idx):
#         loss = self._step(batch, batch_idx)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self._step(batch, batch_idx)
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def _step(self, batch, batch_idx) -> torch.FloatTensor:
#         del batch_idx
#         x = batch["x"][..., :-1].unsqueeze(1)  # (B,1,T)
#         y = batch["xo"][..., 1:]  # (B,T)
#         theta = self.wavenet(x)
#         gmm = wave.bimodal_dist(theta)
#         return -gmm.log_prob(y).mean()  # neg-log-likelihood


def cli_main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    class MyLightningCLI(LightningCLI):
        def after_fit(self):
            ckpt = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            if len(ckpt) > 0:
                _logger.info(f"Best val. model path: {ckpt[0].best_model_path}")
            return super().after_fit()

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        filename="wavenet-{epoch:02d}-{val_loss:.4f}",
    )
    lrm = LearningRateMonitor(logging_interval="step")

    _ = MyLightningCLI(
        wave.WaveNetBase,
        FSeriesDataModule,
        subclass_mode_model=True,
        # seed_everything_default=1234,
        # run=False,
        save_config_overwrite=True,
        trainer_defaults={
            "callbacks": [ckpt, lrm],
            "max_epochs": 30,
            "gpus": 1,
            "log_every_n_steps": 25,
        },
    )


if __name__ == "__main__":
    cli_main()
    # python -m autoregressive.trainer --model autoregressive.variants.RegressionWaveNet --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.trainer.LitBimodalWaveNet --print_config > bimodal.yaml
    # python -m autoregressive.trainer --config config.yaml
