import argparse

import pytorch_lightning as pl

from dataloader.discontinuity import DiscontinuityDataModule
from models.jump_detector import LightningJumpDetector
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--time_spacing", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="src/data/synthetic_data.csv")

    return parser.parse_args()


def train_jump_detector(args):
    num_features = args.num_features
    time_spacing = args.time_spacing
    lr = args.lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    val_split = args.val_split
    test_split = args.test_split
    device = args.device
    
    dataframe = pd.read_csv(args.data_dir)
    
    dm = DiscontinuityDataModule(
        dataframe=dataframe,
        batch_size=batch_size,
        num_features=num_features,
        val_split=val_split,
        test_split=test_split,
    )
    dm.setup()

    model = LightningJumpDetector(num_features=num_features, time_spacing=time_spacing, lr=lr)

    wandb_logger = WandbLogger(project="jump-detector")
    
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="jump-detector-{epoch:02d}-{val_loss:.2f}",
            save_top_k=0,
            mode="min",
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator=device,
        callbacks=callbacks,
    )
        
    trainer.fit(model, dm)
    trainer.test(model, dm.test_dataloader())
    
    
if __name__ == "__main__":
    args = parse_args()
    train_jump_detector(args)