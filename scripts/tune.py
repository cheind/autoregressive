import math
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from autoregressive import datasets, models


def train_tune(config, checkpoint_dir=None, max_epochs=30, batch_size=64, num_gpus=1):

    tunecb = TuneReportCheckpointCallback(
        {"loss": "val_loss"}, on="validation_end", filename="checkpoint"
    )
    lrcb = LearningRateMonitor(logging_interval="step")

    kwargs = {
        "max_epochs": max_epochs,
        "log_every_n_steps": 25,
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."
        ),
        "callbacks": [tunecb, lrcb],
        "progress_bar_refresh_rate": 0,
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    model = models.RegressionWaveNet(**config)
    dm = datasets.FSeriesDataModule(batch_size=batch_size)
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model, datamodule=dm)


def hypertune(
    num_samples=30, max_epochs=30, gpus_per_trial=0.5, tune_tag="regression_tune"
):
    config = {
        "wave_channels": tune.choice([16, 32, 64]),
        "num_blocks": tune.choice([1, 4, 8]),
        "num_layers_per_block": tune.choice([5, 9, 12]),
        "lr": tune.choice([1e-3, 1e-4]),
        "sched_patience": tune.choice([25, 50]),
    }

    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        parameter_columns=[
            "wave_channels",
            "num_blocks",
            "num_layers_per_block",
        ],
        metric_columns=["loss", "training_iteration"],
    )
    analysis = tune.run(
        tune.with_parameters(
            train_tune, max_epochs=max_epochs, num_gpus=gpus_per_trial, batch_size=64
        ),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./ray_results",
        name="tune_regression",
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num-samples", type=int, default=30)
    parser.add_argument("-max-epochs", type=int, default=30)
    parser.add_argument("-name", default="tune_regression")
    args = parser.parse_args()
    hypertune(
        num_samples=args.num_samples, max_epochs=args.max_epochs, tune_tag=args.name
    )


if __name__ == "__main__":
    main()
