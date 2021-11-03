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
            self.create_sampler(greedy=True),
            x,
            num_generate=64,
            max_rolls=8,
            random_rolls=False,
            skip_partial=self.train_full_receptive_field,
        )
        loss = losses.rolling_nstep_mae(roll_y, roll_idx, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def create_sampler(self, greedy: bool = False):
        def greedy_sampler(model, obs, x):
            del model, obs
            bin = torch.argmax(x, dim=1)
            return (bin.float() * self.bin_size).unsqueeze(1)

        def stochastic_sampler(model, obs, x):
            del model, obs
            bin = D.Categorical(logits=x.permute(0, 2, 1)).sample()
            return (bin.float() * self.bin_size).unsqueeze(1)

        if greedy:
            return greedy_sampler
        else:
            return stochastic_sampler
