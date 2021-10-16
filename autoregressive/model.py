import torch
import torch.nn
from torch.nn.modules import module
import torch.optim
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging


_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)

# https://github.com/pytorch/pytorch/issues/1333
# https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
# http://infosci.cornell.edu/~koenecke/files/Deep_Learning_for_Time_Series_Tutorial.pdf
# https://arxiv.org/pdf/1609.03499v2.pdf
# https://github.com/ButterscotchVanilla/Wavenet-PyTorch/blob/master/wavenet/models.py
# https://github.com/tomlepaine/fast-wavenet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py#L118

from . import layers


class AutoregressiveModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        forecast_steps: int = 1,
        kernel_size: int = 2,
        num_layers: int = 10,
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
        self.save_hyperparameters()

    def forward(self, x):
        return self.head(self.features(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        y = y.unfold(-1, self.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        y_hat = self(x.unsqueeze(1))  # .squeeze(1)
        loss = F.smooth_l1_loss(y_hat[..., :n], y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        y = y.unfold(-1, self.forecast_steps, 1).permute(0, 2, 1)
        n = y.shape[-1]
        y_hat = self(x.unsqueeze(1))
        loss = F.smooth_l1_loss(y_hat[..., :n], y)
        self.log("val_loss", loss, prog_bar=True)
        return loss


def train(args):
    import torch.utils.data as data
    from .dataset import FSeriesIterableDataset, PI, Noise
    from pytorch_lightning.callbacks import ModelCheckpoint

    logging.basicConfig(level=logging.INFO)

    batch_size = 64
    dataset_train = FSeriesIterableDataset(
        num_terms=(3, 5),
        num_samples=500,
        period_range=(10.0, 12.0),
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
        num_samples=500,
        period_range=(10.0, 12.0),
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        smoothness=0.75,
    )
    train_loader = data.DataLoader(dataset_train, batch_size, num_workers=0)
    val_loader = data.DataLoader(dataset_val, batch_size, num_workers=0)

    net = AutoregressiveModel(
        in_channels=1,
        hidden_channels=128,
        forecast_steps=128,
        kernel_size=2,
        num_layers=7,
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
    from .dataset import FSeriesIterableDataset, PI, Noise

    # torch.random.manual_seed(123)
    net = AutoregressiveModel.load_from_checkpoint(args.ckpt)
    net.eval()

    dataset_val = FSeriesIterableDataset(
        num_terms=(3, 5),
        num_samples=500,
        period_range=(10.0, 12.0),
        bias_range=0,
        coeff_range=(-1.0, 1.0),
        phase_range=(-PI, PI),
        smoothness=0.75,
    )

    for s in dataset_val:
        x, y, tx, ty = s["x"], s["y"], s["tx"], s["ty"]

        # Assert no data leakage
        # print(net(y[:201].unsqueeze(0).unsqueeze(0))[0, 0, 200])
        # print(net(y.unsqueeze(0).unsqueeze(0))[0, 0, 200])

        # see = np.random.randint(0, 300)
        see = 250  # net.receptive_field
        pred = net.forecast_steps
        yhat = torch.empty(see + pred)
        yhat[:see] = x[:see]
        with torch.no_grad():
            mu = net(yhat[:see].view(1, 1, -1))[0, :, -1]
            yhat[see : see + pred] = mu

        plt.plot(ty, y, c="k")
        plt.plot(tx[:see], y[:see])
        plt.plot(tx[see : see + pred], yhat[see : see + pred])
        plt.ylim(-2, 2)
        plt.show()


# pred = 250
# trials = 1
# yhat = torch.empty((trials, y.shape[0]))
# yhat[:, :see] = x[:see].unsqueeze(0)
# # yhat[:, see - 1] += torch.randn(trials) * 1e-1
# with torch.no_grad():
#     for i in range(see, see + pred):
#         # mu = net(yhat[:, :i].unsqueeze(1))[:, 0, -1]
#         mu = net(y[:i].unsqueeze(0).unsqueeze(0))[:, 0, -1]
#         # p = D.Normal(loc=mu, scale=torch.tensor([1e-2])).sample()
#         yhat[:, i] = mu

# plt.plot(t, y, c="k")
# for tidx in range(trials):
#     plt.plot(t[see : see + pred], yhat[tidx, see : see + pred])
# plt.show()


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