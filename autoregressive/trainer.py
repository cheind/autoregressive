import logging

from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from . import datasets, wave

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


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


# _positive_scale = D.transform_to(constraints.greater_than(0.0))


# def bimodal_dist(theta: torch.Tensor) -> D.MixtureSameFamily:
#     theta = theta.permute(0, 2, 1)  # (B,6,T) -> (B,T,6)
#     mix = D.Categorical(logits=theta[..., :2])
#     comp = D.Normal(loc=theta[..., 2:4], scale=_positive_scale(theta[..., 4:]))
#     gmm = D.MixtureSameFamily(mix, comp)
#     return gmm


# def bimodal_sampler(model: WaveNetBase, obs: torch.Tensor, x: torch.Tensor):
#     gmm = bimodal_dist(x)
#     s = gmm.sample()
#     return s.unsqueeze(1)


def cli_main():
    class MyLightningCLI(LightningCLI):
        def after_fit(self):
            ckpt = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            if len(ckpt) > 0:
                _logger.info(f"Best val. model path: {ckpt[0].best_model_path}")
            return super().after_fit()

    ckpt = ModelCheckpoint(
        monitor="val_loss_epoch",
        filename="wavenet-{epoch:02d}-{val_loss_epoch:.4f}",
        save_top_k=3,
    )
    lrm = LearningRateMonitor(logging_interval="step")
    es = EarlyStopping(
        monitor="train_loss_epoch",
        check_on_train_epoch_end=True,
        min_delta=1e-4,
        patience=1,
        verbose=True,
    )

    _ = MyLightningCLI(
        wave.WaveNetBase,
        datasets.FSeriesDataModule,
        subclass_mode_model=True,
        # seed_everything_default=1234,
        # run=False,
        save_config_overwrite=True,
        trainer_defaults={
            "callbacks": [ckpt, lrm, es],
            "max_epochs": 30,
            "gpus": 1,
            "log_every_n_steps": 25,
        },
    )


if __name__ == "__main__":
    cli_main()
    # python -m autoregressive.trainer --model autoregressive.models.RegressionWaveNet --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.variants.QuantizedWaveNet --data.num_bins 32 --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.trainer.LitBimodalWaveNet --print_config > bimodal.yaml
    # python -m autoregressive.trainer --config config.yaml
