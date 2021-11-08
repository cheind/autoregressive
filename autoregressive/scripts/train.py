import logging

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from .. import wave

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


def cli_main():
    class MyLightningCLI(LightningCLI):
        def after_fit(self):
            ckpt = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            if len(ckpt) > 0:
                _logger.info(f"Best val. model path: {ckpt[0].best_model_path}")
            return super().after_fit()

        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.quantization_levels",
                "model.quantization_levels",
                apply_on="instantiate",
            )

    ckpt = ModelCheckpoint(
        monitor="train_loss_epoch",
        filename="wavenet-{epoch:02d}-{train_loss_epoch:.4f}",
        save_top_k=3,
    )
    lrm = LearningRateMonitor(logging_interval="step")
    # es = EarlyStopping(
    #     monitor="train_loss_epoch",
    #     check_on_train_epoch_end=True,
    #     min_delta=1e-4,
    #     patience=2,
    #     verbose=True,
    # )

    _ = MyLightningCLI(
        wave.WaveNet,
        pl.LightningDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
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
    # python -m autoregressive.scripts.train --data autoregressive.datasets.BentLinesDataModule --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.models.RegressionWaveNet --data autoregressive.datasets.AEPHourlyDataModule --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.variants.QuantizedWaveNet --data.num_bins 32 --print_config > config.yaml
    # python -m autoregressive.trainer --model autoregressive.trainer.LitBimodalWaveNet --print_config > bimodal.yaml
    # python -m autoregressive.trainer --config config.yaml
