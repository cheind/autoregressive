import logging

import pytorch_lightning as pl
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.utils import data

from autoregressive import dataset

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)

# https://github.com/pytorch/pytorch/issues/1333
# https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
# http://infosci.cornell.edu/~koenecke/files/Deep_Learning_for_Time_Series_Tutorial.pdf
# https://arxiv.org/pdf/1609.03499v2.pdf
# https://github.com/ButterscotchVanilla/Wavenet-PyTorch/blob/master/wavenet/models.py
# https://github.com/tomlepaine/fast-wavenet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py#L118
# https://github.com/ibab/tensorflow-wavenet/issues/98
# https://arxiv.org/pdf/1703.04691.pdf
# https://github.com/litanli/wavenet-time-series-forecasting/blob/master/wavenet_pytorch.py
# https://reposhub.com/python/deep-learning/EvilPsyCHo-Deep-Time-Series-Prediction.html
# https://github.com/EvilPsyCHo/Deep-Time-Series-Prediction/blob/f6a6da060bb3f7d07f2a61967ee6007e9821064e/deepseries/nn/cnn.py#L122

from . import layers


class AutoregressiveModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        forecast_steps: int = 1,
        kernel_size: int = 2,
        num_layers: int = 10,
        head_activation: bool = False,
        loss_require_full_receptive_field: bool = True,
    ) -> None:
        super().__init__()

        modules = [
            layers.BasicBlock(in_channels, hidden_channels, kernel_size, dilation=1)
        ]
        for i in range(1, num_layers):
            drate = 2 ** i
            b = layers.BasicBlock(
                hidden_channels, hidden_channels, kernel_size, dilation=drate
            )
            modules.append(b)
        self.features = torch.nn.Sequential(*modules)
        self.head = torch.nn.Conv1d(
            hidden_channels, forecast_steps, 1, stride=1, padding=0
        )

        # Compute the lag (receptive field of this model). Currently
        # the equation below equals 2**n_layers (k=2), but is more generic
        # when we introduce repititions, i.e dilations [1,2,4,8,1,2,4,8]
        self.receptive_field = (kernel_size - 1) * sum(
            [2 ** i for i in range(num_layers)]
        ) + 1
        _logger.info(f"Receptive field of model {self.receptive_field}")
        self.forecast_steps = forecast_steps
        self.head_activation = head_activation
        self.loss_require_full_receptive_field = loss_require_full_receptive_field
        self.save_hyperparameters()

    def forward(self, x):
        h = self.head(self.features(x))
        if self.head_activation:
            h = torch.tanh(h)
        return h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch["x"][..., :-1]
        y = batch["xo"][..., 1:]
        y = y.unfold(-1, self.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        r = 0
        if self.loss_require_full_receptive_field:
            r = self.receptive_field
        y_hat = self(x.unsqueeze(1))  # .squeeze(1)
        loss = F.l1_loss(y_hat[..., r:n], y[..., r:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"][..., :-1]
        y = batch["xo"][..., 1:]
        y = y.unfold(-1, self.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        r = 0
        if self.loss_require_full_receptive_field:
            r = self.receptive_field
        y_hat = self(x.unsqueeze(1))
        loss = F.l1_loss(y_hat[..., r:n], y[..., r:])
        self.log("val_loss", loss, prog_bar=True)
        return loss


class NStepPrediction:
    """Predict n-steps using direct model output"""

    def __init__(self, model: AutoregressiveModel) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        assert horizon <= self.model.forecast_steps
        x = x.view((1, 1, -1))
        y = self.model(x)[0, :horizon, -1]
        return y


class NaiveGeneration:
    def __init__(self, model: AutoregressiveModel) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        r = self.model.receptive_field
        y = x.new_empty(r + horizon)
        y[:r] = x[-r:]  # Copy last `see` observations
        for h in range(horizon):
            p = self.model(y[h : h + r].view(1, 1, -1))
            y[h + r] = p[0, 0, -1]
        return y[r:]


def generate_datasets():
    from .dataset import PI, FSeriesIterableDataset, Noise

    dataset_train = FSeriesIterableDataset(
        num_terms=(3, 5),
        num_tsamples=1000,
        dt=0.02,
        start_trange=0.0,
        period_range=(20.0, 24.0),
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        smoothness=0.75,
        transform=Noise(scale=1e-3, p=0.25),
    )
    # Note, using fixed noise with probability 1, will teach the model
    # to always account for that noise level. When you then turn off the
    # noise in evaluation, the models predictions will be significantly
    # more noisy. Hence, we add noise only once in a while.

    dataset_val = FSeriesIterableDataset(
        num_terms=(3, 5),
        num_tsamples=1000,
        dt=0.02,
        start_trange=0.0,
        period_range=(20.0, 24.0),
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        smoothness=0.75,
    )

    return dataset_train, dataset_val


def train(args):
    import torch.utils.data as data
    from pytorch_lightning.callbacks import ModelCheckpoint

    logging.basicConfig(level=logging.INFO)

    batch_size = 64

    dataset_train, dataset_val = generate_datasets()
    train_loader = data.DataLoader(dataset_train, batch_size, num_workers=0)
    val_loader = data.DataLoader(dataset_val, batch_size, num_workers=0)

    net = AutoregressiveModel(
        in_channels=1,
        hidden_channels=128,
        forecast_steps=128,
        kernel_size=2,
        num_layers=9,
        head_activation=False,
        loss_require_full_receptive_field=True,
    )
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        filename="autoreg-{step:05d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[ckpt],
        val_check_interval=4096 / batch_size,
        limit_val_batches=1024 / batch_size,
    )
    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
    print(ckpt.best_model_path)


def eval(args):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    from .dataset import PI, FSeriesIterableDataset

    # torch.random.manual_seed(123)
    net = AutoregressiveModel.load_from_checkpoint(args.ckpt)
    preds = [
        (NStepPrediction(net), "n-step prediction"),
        # (NaiveGeneration(net), "naive n-step generation"),
    ]

    _, dataset_val = generate_datasets()

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
    horizon = 128

    for ax, s in zip(grid, dataset_val):
        x, xo, t = s["x"], s["xo"], s["t"]

        see = torch.randint(
            net.receptive_field, x.shape[-1] - horizon, size=(1,)
        ).item()

        # see = np.random.randint(0, 300)
        ax.plot(t, xo, c="k", linestyle="--", linewidth=0.5)
        ax.plot(t[:see], x[:see], c="k", linewidth=0.5)
        for p, label in preds:
            y = p.predict(x[:see], horizon)
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
