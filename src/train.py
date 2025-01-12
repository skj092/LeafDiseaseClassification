from engine import Trainer
from pathlib import Path
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from modules import (setup_logger, seed_everything,
                     CassavaDataset, get_model,
                     get_transform)
from accelerate import Accelerator
from dotenv import load_dotenv
load_dotenv()


# Config Setup
seed_everything(42)
accenerator = Accelerator()
use_external = False
multi_gpu = False
wandb = False
batch_size = 4
num_epochs = 2
device = accenerator.device
path = Path("./data")

# lightning studio specific
df = pd.read_csv(path / "merged.csv")
logger, run = setup_logger(use_wandb=wandb, project_name="cassava-experiment")
logger.info(f"shape of df: {df.shape}")


df2020 = df[df["source"] == 2020]
df2020 = df2020.sample(n=10)
skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42,
)
for fold, (train_index, valid_index) in enumerate(skf.split(df2020, df2020["label"].values)):
    train_df = df2020.iloc[train_index] if not use_external else pd.concat(
        [df2020.iloc[train_index], df[df["source"] == 2019]], axis=0)
    valid_df = df2020.iloc[valid_index]

    # Get train and valid dataset, dataloader
    train_ds = CassavaDataset(
        train_df, path / "train", transforms=get_transform())
    valid_ds = CassavaDataset(
        valid_df, path / "train", transforms=get_transform(is_train=False))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # Load the model
    model = get_model()
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs)

    best_loss = float('inf')
    model_dir = Path(f"{path}/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer, train_dl, valid_dl, scheduler = accenerator.prepare(
        model, optimizer, train_dl, valid_dl, scheduler)
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        accelerator=accenerator,
        logger=logger,
        run=run)
    trainer.fit(num_epochs=2)

if run:
    run.finish()
