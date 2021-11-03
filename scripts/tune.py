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


def train_tune(
    config,
    checkpoint_dir=None,
    num_epochs: int = 30,
    batch_size: int = 64,
    gpu_per_trial: float = 1.0,
    train_seed: int = None,
    val_seed: int = None,
):

    tunecb = TuneReportCheckpointCallback(
        {"loss": "val_loss"}, on="validation_end", filename="checkpoint"
    )
    lrcb = LearningRateMonitor(logging_interval="step")

    kwargs = {
        "max_epochs": num_epochs,
        "log_every_n_steps": 25,
        "gpus": math.ceil(gpu_per_trial),
        "logger": TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."
        ),
        "callbacks": [tunecb, lrcb],
        "progress_bar_refresh_rate": 0,
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    train_params = datasets.FSeriesParams(smoothness=0.75, seed=train_seed)
    val_params = datasets.FSeriesParams(smoothness=0.75, num_curves=512, seed=val_seed)
    dm = datasets.FSeriesDataModule(
        batch_size=batch_size,
        train_fseries_params=train_params,
        val_fseries_params=val_params,
    )

    model = models.RegressionWaveNet(**config)

    trainer = pl.Trainer(**kwargs)
    trainer.fit(model, datamodule=dm)


def hypertune(args):
    config = {
        "wave_channels": tune.choice([16, 32, 64]),
        "num_blocks": tune.choice([1, 2]),
        "num_layers_per_block": tune.choice([5, 9, 10]),
        "lr": tune.choice([1e-3, 1e-4]),
        "sched_patience": tune.choice([25, 50]),
    }

    scheduler = ASHAScheduler(
        max_t=args.num_epochs,
        grace_period=min(10, args.num_epochs),
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
            train_tune,
            num_epochs=args.num_epochs,
            gpu_per_trial=args.gpu_per_trial,
            train_seed=args.train_seed,
            val_seed=args.val_seed,
            batch_size=64,
        ),
        resources_per_trial={"cpu": 1, "gpu": args.gpu_per_trial},
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./ray_results",
        name=args.experiment_name,
        trial_dirname_creator=lambda trial: str(trial),
        log_to_file=True,
    )
    print("Best hyperparameters found were: ", analysis.best_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num-samples", type=int, default=30)
    parser.add_argument("-gpu-per-trial", type=float, default=0.5)
    parser.add_argument("-num-epochs", type=int, default=30)
    parser.add_argument("-experiment-name", default="tune_regression")
    parser.add_argument("-train-seed", type=int, default=None)
    parser.add_argument("-val-seed", type=int, default=None)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.num_epochs = 1
        args.num_samples = 1
        args.gpu_per_trial = 0.0
    hypertune(args)


if __name__ == "__main__":
    main()
