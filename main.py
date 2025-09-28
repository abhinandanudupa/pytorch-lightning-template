from typing import Any, Dict, List, Tuple, Union

import torch
import torchvision
import lightning

from loss_module import LossModule
from module import LightningWrapper

if __name__ == "__main__":
    model = Model()
    exp_name = "exp"
    loss_fn = LossModule(losses=[{"mse": {"params": {"reduction": "mean"}, "weight": 1.0}}])
    lmod = LightningWrapper(model=model, loss_fn=loss_fn)
    logger = lightning.pytorch.loggers.TensorBoardLogger(f"logs/{exp_name}", name=exp_name)
    mc = lightning.pytorch.callbacks.ModelChecpoint(monitor="valid/loss", dirpath=f"checkpoints/{exp_name}", filename="checkpoint-{epoch:02d}-{val_acc:.2f}", save_top_k=-1, mode="min", verbose=True)
    es = lightning.pytorch.callbacks.EarlyStopping(monitor="valid/loss", patience=3, verbose=True, mode="min")
    trainer = lightning.Trainer(
        max_epochs=20,
        accelerator="auto",
        callbacks=[mc, es],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        strategy="dpp",
        precison=32,
        enable_checkpointing=True,
        num_sanity_val_steps=2,
        fast_dev_train=False,
        check_val_every_n_epoch=1,
        gradient_clip_val=4.0,
        accumulate_grad_batches=128,
        profiler="simple",
        benchmark=True,
        detect_anomaly=False,
    )
    train_ds = torchvision.datasets.datasets.MNIST("data", train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.MNIST("data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

    trainer.fit(lmod, train_loader, valid_loader)


