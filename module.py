from typing import Dict, List, Tuple

import torch
import lightning


class LightningWrapper(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        lr: float = 1e-3,
        **kwargs,
    ):
        super.__init__()
        self.save_hyperparameters()
        self.model: torch.nn.Module = model
        self.loss_fn: torch.nn.Module = loss_fn

    def forward(self, *args, **kwargs):
        out: torch.Tensor = self.model(*args, **kwargs)

        return out

    def training_step(self, batch, batch_idx):
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch
        out: torch.Tensor = self.model(x)
        loss: Dict[str, torch.Tensor] = self.loss_fn(out, y)
        self.log("train/loss", loss["total"])

        return loss

    def validation_step(self, batch, batch_idx):
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch
        out: torch.Tensor = self.model(x)
        loss: Dict[str, torch.Tensor] = self.loss_fn(out, y)
        self.log(
            "valid/loss",
            loss["total"],
            prog_bar=True,
            logger=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(step_size=1, gamma=0.5)

        return [
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        ]

    def on_train_epoch_end(self):
        for i, opt in enumerate(self.optimizers()):
            lr = opt.param_groups[0]["lr"]
            self.log(f"lr_opt_{i}", lr)
