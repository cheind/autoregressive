import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributions as D

# https://github.com/pytorch/pytorch/issues/1333
# https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
# http://infosci.cornell.edu/~koenecke/files/Deep_Learning_for_Time_Series_Tutorial.pdf


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, input):
        return super().forward(F.pad(input, (self.left_pad, 0)))


def basic_block(
    in_channels: int, out_channels: int, kernel_size: int, dilation: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        CausalConv1d(
            in_channels, out_channels, kernel_size, stride=1, dilation=dilation
        ),
        torch.nn.LeakyReLU(),
        torch.nn.BatchNorm1d(out_channels),
    )


class AutoRegressor(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 2,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        channels = [in_channels] + [hidden_channels] * num_blocks
        dilation_depth = [2 ** (i + 1) for i in range(num_blocks + 1)]
        blocks = []
        for cin, cout, d in zip(channels[:-1], channels[1:], dilation_depth):
            blocks.append(basic_block(cin, cout, kernel_size, d))
        self.regress = torch.nn.Sequential(
            *blocks,
            CausalConv1d(
                channels[-1],
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_depth[-1],
            )
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.regress(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.regress(x.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.regress(x.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss


def train():
    import torch.utils.data as data
    from .dataset import FSeriesDataset
    from pytorch_lightning.callbacks import ModelCheckpoint

    batch_size = 16
    blocks = 5
    hdims = 128

    dataset_train = FSeriesDataset(num_curves=4096, num_terms=3, noise=0.0)
    dataset_val = FSeriesDataset(num_curves=4096, num_terms=3, noise=0)
    train_loader = data.DataLoader(dataset_train, batch_size, num_workers=0)
    val_loader = data.DataLoader(dataset_val, batch_size, num_workers=0)

    net = AutoRegressor(num_blocks=blocks, hidden_channels=hdims)
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        filename="autoreg-{epoch:02d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(gpus=1, callbacks=[ckpt])
    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)


def eval():
    import matplotlib.pyplot as plt
    from .dataset import FSeriesDataset

    # torch.random.manual_seed(123)
    net = AutoRegressor.load_from_checkpoint(
        r"C:\dev\autoregressive-damp\lightning_logs\version_15\checkpoints\autoreg-epoch=02-val_loss=0.0007.ckpt"
    )
    net.eval()
    data = FSeriesDataset(num_curves=4096, noise=0.0, num_terms=3)

    for i in range(10):
        x, y = data[i]
        t = torch.linspace(0, 10, 500)

        # Assert no data leakage
        # print(net(y[:201].unsqueeze(0).unsqueeze(0))[0, 0, 200])
        # print(net(y.unsqueeze(0).unsqueeze(0))[0, 0, 200])

        see = 250
        pred = 250
        trials = 1
        yhat = torch.empty((trials, y.shape[0]))
        yhat[:, :see] = x[:see].unsqueeze(0)
        # yhat[:, see - 1] += torch.randn(trials) * 1e-1
        with torch.no_grad():
            for i in range(see, see + pred):
                # mu = net(yhat[:, :i].unsqueeze(1))[:, 0, -1]
                mu = net(y[:i].unsqueeze(0).unsqueeze(0))[:, 0, -1]
                # p = D.Normal(loc=mu, scale=torch.tensor([1e-2])).sample()
                yhat[:, i] = mu

        plt.plot(t, y, c="k")
        for tidx in range(trials):
            plt.plot(t[see : see + pred], yhat[tidx, see : see + pred])
        plt.show()


#  To increase reception field exponentially with linearly increasing number of parameters


if __name__ == "__main__":
    # train()
    eval()
    # x = torch.arange(1, 10).float()
    # n = CausalConv1d(1, 1, 2, dilation=2)
    # n.weight.data.fill_(1.0)
    # n.bias.data.fill_(0.0)
    # print(n(x.view(1, 1, -1)))

    # net = AutoRegressor(1, 1, num_blocks=4)
    # print(net)
