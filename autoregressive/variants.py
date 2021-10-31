import logging
import torch
import torch.nn.functional as F
import torch.distributions as D

from . import wave, losses

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class RegressionWaveNet(wave.WaveNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        forecast_steps: int = 1,
        wave_channels: int = 64,
        num_blocks: int = 1,
        num_layers_per_block: int = 9,
        train_full_receptive_field: bool = True,
        train_exp_decay: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=forecast_steps,
            wave_channels=wave_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
        )
        self.train_full_receptive_field = train_full_receptive_field
        self.train_exp_decay = train_exp_decay
        if self.train_exp_decay and forecast_steps > 1:
            self.register_buffer(
                "l1_weights", self._exp_weights(forecast_steps, 1e-3).view(1, -1, 1)
            )
        else:
            self.l1_weights = 1.0
        super().save_hyperparameters()

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=1, min_lr=1e-7, threshold=1e-7
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
        y: torch.Tensor = batch["xo"][..., 1:]
        y = y.unfold(-1, self.out_channels, 1).permute(0, 2, 1)
        n = y.shape[-1]
        yhat = self(x)
        r = self.receptive_field if self.train_full_receptive_field else 0
        losses = F.l1_loss(yhat[..., r:n], y[..., r:], reduction="none")
        loss = torch.mean(losses * self.l1_weights)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
        y: torch.Tensor = batch["xo"][..., 1:].unsqueeze(1)

        roll_y, _, roll_idx = losses.rolling_nstep(
            self,
            self.create_sampler(),
            x,
            num_generate=64,
            max_rolls=8,
            random_rolls=False,
            skip_partial=self.train_full_receptive_field,
        )
        loss = losses.rolling_nstep_mae(roll_y, roll_idx, y)
        self.log("val_loss", loss, prog_bar=True)

    def _exp_weights(self, T, n):
        """Returns exponential decaying weights for T terms from 1.0 down to n (n>0)"""
        n = torch.as_tensor(n).float()
        x = torch.arange(0, T, 1)
        b = -torch.log(n) / (T - 1)
        return torch.exp(-x * b)

    def create_sampler(self):
        return regression_sampler


def regression_sampler(
    model: RegressionWaveNet, obs: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # The model directly predicts the values for the next timesteps,
    # so this is the identity function
    return x


class QuantizedWaveNet(wave.WaveNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        wave_channels: int = 64,
        num_blocks: int = 1,
        num_layers_per_block: int = 9,
        num_bins: int = 32,
        train_full_receptive_field: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=num_bins,
            wave_channels=wave_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
        )
        self.train_full_receptive_field = train_full_receptive_field
        self.num_bins = num_bins
        self.bin_size = 1.0 / (num_bins - 1)
        super().save_hyperparameters()

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=2, min_lr=1e-7, threshold=1e-7
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
        y: torch.Tensor = batch["b"][..., 1:]
        logits = self(x)
        r = self.receptive_field if self.train_full_receptive_field else 0
        loss = F.cross_entropy(logits[..., r:], y[..., r:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
        y: torch.Tensor = batch["xo"][..., 1:].unsqueeze(1)

        roll_y, _, roll_idx = losses.rolling_nstep(
            self,
            self.create_sampler(),
            x,
            num_generate=64,
            max_rolls=8,
            random_rolls=False,
            skip_partial=self.train_full_receptive_field,
        )
        loss = losses.rolling_nstep_mae(roll_y, roll_idx, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def create_sampler(self):
        def sampler(model, obs, x):
            del model, obs
            bin = D.Categorical(logits=x.permute(0, 2, 1)).sample()
            return (bin.float() * self.bin_size).unsqueeze(1)

        return sampler
