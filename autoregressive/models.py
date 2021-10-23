import logging

import pytorch_lightning as pl
import torch
import torch.nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim

from . import dataset, wave, trainer

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


def fit(
    model: pl.LightningModule,
    dataset_train,
    dataset_val,
    batch_size: int = 64,
    max_epochs: int = 30,
    num_train_curves: int = 2 ** 11,
    num_val_curves: int = 2 ** 9,
):
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    is_train_iterable = isinstance(dataset_train, torch.utils.data.IterableDataset)
    is_val_iterable = isinstance(dataset_val, torch.utils.data.IterableDataset)

    train_loader = data.DataLoader(dataset_train, batch_size, num_workers=4)
    val_loader = data.DataLoader(dataset_val, batch_size, num_workers=4)

    ckpt = ModelCheckpoint(
        monitor="val_loss",
        filename="regressionwavenet-{epoch:02d}-{val_loss:.4f}",
    )
    lrm = LearningRateMonitor(logging_interval="step")

    trainer_args = {
        "gpus": 1,
        "callbacks": [ckpt, lrm],
        "max_epochs": max_epochs,
    }
    if is_train_iterable:
        trainer_args.update({"limit_train_batches": num_train_curves // batch_size}),
    if is_val_iterable:
        trainer_args.update({"limit_val_batches": num_val_curves // batch_size}),

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    _logger.info(f"Best model on validation {ckpt.best_model_path}")


# class NStepPrediction:
#     """Predict n-steps using direct model output"""

#     def __init__(self, model) -> None:
#         self.model = model
#         self.model.eval()

#     @torch.no_grad()
#     def predict(self, x: torch.Tensor, t: torch.Tensor, horizon: int) -> torch.Tensor:
#         assert horizon <= self.model.forecast_steps
#         x = x.view((1, 1, -1))
#         t = t.view((1, 1, -1))
#         if self.model.use_positional_encoding:
#             y = self.model(torch.cat((x, t), 1))
#         else:
#             y = self.model(x)
#         return y[0, :horizon, -1]


def train(args):
    logging.basicConfig(level=logging.INFO)

    dataset_train, dataset_val = dataset.create_default_datasets()
    model = wave.LitRegressionWaveNet(
        in_channels=1,
        forecast_steps=1,
        residual_channels=128,
        skip_channels=128,
        num_blocks=1,
        num_layers_per_block=9,
        train_full_receptive_field=True,
        train_exp_decay=True,
    )
    fit(model, dataset_train, dataset_val, batch_size=64)


def eval(args):
    # torch.use_deterministic_algorithms(True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    from .dataset import PI, FSeriesIterableDataset

    # torch.random.manual_seed(123)
    net = trainer.LitRegressionWaveNet.load_from_checkpoint(args.ckpt).wavenet
    preds = [
        # (NStepPrediction(net), "n-step prediction"),
        (FastGeneration(net), "fast n-step generation"),
        # (NaiveGeneration(net), "naive n-step generation"),
    ]

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(4, 4),
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # horizon = net.forecast_steps
    horizon = 511

    for ax, s in zip(grid, dataset_val):
        x, xo, t = s["x"], s["xo"], s["t"]

        see = torch.randint(
            net.receptive_field, x.shape[-1] - horizon, size=(1,)
        ).item()

        # see = np.random.randint(0, 300)
        ax.plot(t, xo, c="k", linestyle="--", linewidth=0.5)
        ax.plot(t[:see], x[:see], c="k", linewidth=0.5)
        for p, label in preds:
            y = p.predict(x[:see], t[:see], horizon)
            ax.plot(t[see : see + horizon], y, label=label)
        ax.set_ylim(-2, 2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.show()


import torch.distributions as D


def custom_sampler(
    model: wave.WaveNet, obs: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    y = D.Normal(loc=x, scale=1e-4).sample()
    return y


@torch.no_grad()
def eval2(args):
    # torch.use_deterministic_algorithms(True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import itertools

    from .dataset import PI, FSeriesIterableDataset

    # torch.random.manual_seed(123)
    net = trainer.LitRegressionWaveNet.load_from_checkpoint(args.ckpt).wavenet
    net.eval().cuda()

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # horizon = net.forecast_steps

    for ax, s in zip(grid, dataset_val):
        x, xo, t = s["x"], s["xo"], s["t"]
        dt = t[-1] - t[-2]

        N = 1

        g = wave.generate_fast(
            net,
            x[..., :-1].cuda().view(1, 1, -1).repeat(N, 1, 1),
            sampler=wave.regression_sampler,
        )
        xn_fast = torch.cat(list(itertools.islice(g, 256)), -1).view(N, -1).cpu()

        g = wave.generate(
            net,
            x[..., :-1].cuda().view(1, 1, -1).repeat(N, 1, 1),
            sampler=wave.regression_sampler,
        )
        xn_slow = torch.cat(list(itertools.islice(g, 256)), -1).view(N, -1).cpu()

        tn = torch.arange(0.0, dt * xn_fast.shape[-1], dt) + t[-1]

        ax.plot(t, xo, c="k", linewidth=0.5)
        for xnf, xns in zip(xn_fast, xn_slow):
            ax.plot(tn, xnf, alpha=0.7)
            ax.plot(tn, xns, alpha=0.7)

        #     ax.plot(t[:see], x[:see], c="k", linewidth=0.5)
        #     for p, label in preds:
        #         y = p.predict(x[:see], t[:see], horizon)
        #         ax.plot(t[see : see + horizon], y, label=label)
        ax.set_ylim(-2, 2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.show()


@torch.no_grad()
def eval3(args):
    # torch.use_deterministic_algorithms(True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import itertools

    # torch.random.manual_seed(123)
    net = trainer.LitBimodalWaveNet.load_from_checkpoint(args.ckpt).wavenet
    net.eval().cuda()

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(1, 1),
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # horizon = net.forecast_steps

    for ax, s in zip(grid, dataset_val):
        x, xo, t = s["x"], s["xo"], s["t"]
        dt = t[-1] - t[-2]

        N = 10
        F = 256

        g = wave.generate_fast(
            net,
            x[..., :-1].cuda().view(1, 1, -1).repeat(N, 1, 1),
            sampler=wave.bimodal_sampler,
        )
        xn_fast = torch.cat(list(itertools.islice(g, F)), -1).view(N, -1).cpu()

        tn = torch.arange(0.0, dt * xn_fast.shape[-1], dt) + t[-1]

        ax.plot(t, xo, c="k", linewidth=0.5)
        for xnf in xn_fast:
            ax.plot(tn, xnf, alpha=0.7)

        #     ax.plot(t[:see], x[:see], c="k", linewidth=0.5)
        #     for p, label in preds:
        #         y = p.predict(x[:see], t[:see], horizon)
        #         ax.plot(t[see : see + horizon], y, label=label)
        ax.set_ylim(-4, 4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.show()


#  To increase reception field exponentially with linearly increasing number of parameters


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_train = subparsers.add_parser("train", help="train")
    parser_eval = subparsers.add_parser("eval", help="eval cartpole agent")
    parser_eval.add_argument("ckpt", type=Path, help="model checkpoint file")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        eval3(args)


if __name__ == "__main__":
    main()
