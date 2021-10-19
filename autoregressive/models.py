import logging

import pytorch_lightning as pl
import torch
import torch.nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim

from . import dataset, wave

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class RegressionWaveNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        forecast_steps: int = 64,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 1,
        num_layers: int = 7,
        loss_require_full_receptive_field: bool = True,
    ) -> None:
        super().__init__()
        self.wave = wave.WaveNetLinear(
            in_channels,
            forecast_steps,
            residual_channels,
            skip_channels,
            num_blocks,
            num_layers,
        )
        self.forecast_steps = forecast_steps
        self.loss_require_full_receptive_field = loss_require_full_receptive_field
        self.receptive_field = self.wave.receptive_field
        self.use_positional_encoding = in_channels > 1
        _logger.info(f"Receptive field of model {self.receptive_field}")
        super().save_hyperparameters()

    def forward(self, x):
        return self.wave(x)

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=1, min_lr=1e-7, threshold=1e-7
                ),
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
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
        x = batch["x"][..., :-1]
        t = batch["t"][..., :-1]
        y = batch["xo"][..., 1:]
        y = y.unfold(-1, self.forecast_steps, 1).permute(0, 2, 1)
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        if self.use_positional_encoding:
            y_hat = self(torch.cat((x, t), 1))
        else:
            y_hat = self(x)
        n = y.shape[-1]
        r = self.receptive_field if self.loss_require_full_receptive_field else 0
        loss = F.l1_loss(y_hat[..., r:n], y[..., r:])
        return loss

    def fit(self, dataset_train, dataset_val, batch_size: int = 64):
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

        train_loader = data.DataLoader(dataset_train, batch_size, num_workers=4)
        val_loader = data.DataLoader(dataset_val, batch_size, num_workers=4)

        ckpt = ModelCheckpoint(
            monitor="val_loss",
            filename="regressionwavenet-{step:05d}-{val_loss:.4f}",
        )
        lrm = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            gpus=1,
            callbacks=[ckpt, lrm],
            val_check_interval=2 ** 13 / batch_size,
            limit_val_batches=2 ** 11 / batch_size,
        )
        trainer.fit(self, train_dataloader=train_loader, val_dataloaders=val_loader)
        _logger.info(f"Best model on validation {ckpt.best_model_path}")


class NStepPrediction:
    """Predict n-steps using direct model output"""

    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: torch.Tensor, horizon: int) -> torch.Tensor:
        assert horizon <= self.model.forecast_steps
        x = x.view((1, 1, -1))
        t = t.view((1, 1, -1))
        if self.model.use_positional_encoding:
            y = self.model(torch.cat((x, t), 1))
        else:
            y = self.model(x)
        return y[0, :horizon, -1]


class NaiveGeneration:
    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()
        assert not self.model.use_positional_encoding

    @torch.no_grad()
    def predict(self, x: torch.Tensor, t: torch.Tensor, horizon: int) -> torch.Tensor:
        r = self.model.receptive_field
        y = x.new_empty(r + horizon)
        y[:r] = x[-r:]  # Copy last `see` observations
        for h in range(horizon):
            p = self.model(y[h : h + r].view(1, 1, -1))
            y[h + r] = p[0, 0, -1]
        return y[r:]


def train(args):
    logging.basicConfig(level=logging.INFO)

    dataset_train, dataset_val = dataset.create_default_datasets()
    model = RegressionWaveNet(
        in_channels=1,
        forecast_steps=1,
        residual_channels=128,
        skip_channels=64,
        num_blocks=1,
        num_layers=9,
        loss_require_full_receptive_field=True,
    )
    model.fit(dataset_train, dataset_val, batch_size=64)


def eval(args):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    from .dataset import PI, FSeriesIterableDataset

    # torch.random.manual_seed(123)
    net = RegressionWaveNet.load_from_checkpoint(args.ckpt)
    preds = [
        # (NStepPrediction(net), "n-step prediction"),
        (NaiveGeneration(net), "naive n-step generation"),
    ]

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure(figsize=(8.0, 8.0))
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(2, 2),
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # horizon = net.forecast_steps
    # horizon = net.forecast_steps
    horizon = 256

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
        eval(args)


if __name__ == "__main__":
    main()
