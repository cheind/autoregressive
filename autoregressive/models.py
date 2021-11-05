import dataclasses
import logging
import torch
import torch.nn.functional as F

from . import wave, losses

_logger = logging.getLogger("pytorch_lightning")
_logger.setLevel(logging.INFO)


class RegressionWaveNet(wave.WaveNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        wave_channels: int = 64,
        num_blocks: int = 1,
        num_layers_per_block: int = 9,
        skip_incomplete_receptive_field: bool = True,
        loss_unroll_steps: int = 1,
        loss_margin: float = 0.0,
        lr: float = 1e-3,
        sched_patience: int = 25,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=1,
            wave_channels=wave_channels,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
            use_encoded=False,
        )
        self.skip_incomplete_receptive_field = skip_incomplete_receptive_field
        self.loss_unroll_steps = loss_unroll_steps
        self.loss_margin = loss_margin
        self.lr = lr
        self.sched_patience = sched_patience
        super().save_hyperparameters()

    def configure_optimizers(self):
        import torch.optim.lr_scheduler as sched

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.5,
                    patience=self.sched_patience,
                    min_lr=5e-5,
                    threshold=1e-7,
                ),
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
                "strict": False,
            },
        }

    def training_step(self, batch, batch_idx):
        if self.loss_unroll_steps == 1:
            x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
            y: torch.Tensor = batch["xo"][..., 1:]
            y = y.unfold(-1, self.out_channels, 1).permute(0, 2, 1)
            n = y.shape[-1]
            yhat = self(x)
            r = self.receptive_field if self.skip_incomplete_receptive_field else 0
            loss = F.l1_loss(yhat[..., r:n], y[..., r:])
        else:
            x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
            y: torch.Tensor = batch["x"][..., 1:].unsqueeze(1)
            roll_y, _, roll_idx = losses.rolling_nstep(
                self,
                self.create_sampler(),
                x,
                num_generate=self.loss_unroll_steps,
                max_rolls=1,  # one random roll per batch element
                random_rolls=True,
                skip_partial=self.skip_incomplete_receptive_field,
                detach_sample=False,
            )
            loss = losses.rolling_nstep_mae(
                roll_y, roll_idx, y, margin=self.loss_margin
            )
        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss.detach()}

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["x"][..., :-1].unsqueeze(1)
        y: torch.Tensor = batch["x"][..., 1:].unsqueeze(1)

        roll_y, _, roll_idx = losses.rolling_nstep(
            self,
            self.create_sampler(),
            x,
            num_generate=64,
            max_rolls=8,
            random_rolls=False,
            skip_partial=self.skip_incomplete_receptive_field,
        )
        loss = losses.rolling_nstep_mae(
            roll_y, roll_idx, y, margin=0.0
        )  # don't apply margin here
        return {"val_loss": loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)

    def create_sampler(self):
        def sampler(model: RegressionWaveNet, obs: torch.Tensor, x: torch.Tensor):
            # The model directly predicts the values for the next timesteps,
            # so this is the identity function
            return x

        return sampler
