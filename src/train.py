from pathlib import Path
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from modules import (setup_logger, seed_everything,
                     CassavaDataset, get_model, train_one_epoch,
                     get_transform, myModel)


# Config Setup
seed_everything(42)
use_external = False
multi_gpu = False
wandb = True
batch_size = 4
num_epochs = 20
arch_name = "vit_base_patch16_384"
pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # resnet34
    # model = get_model()

    # vit model
    model = myModel(arch_name, pretrained=True, img_size=384)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch))

    best_loss = float('inf')
    model_dir = Path(f"{path}/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(num_epochs):
        state = {
            'epoch': epoch,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        logger.info(f"============Epoch: {epoch}/{num_epochs} ============")
        train_loss, valid_loss, train_acc, valid_acc = train_one_epoch(
            model, train_dl, valid_dl, loss_fn, optimizer, scheduler, logger, run
        )

        # save the model
        if valid_loss < best_loss:
            model_file = model_dir / f"model_{fold}.pth"
            print('========New optimal found, saving state==========')
            state['best_loss'] = best_loss = valid_loss
            torch.save(state, model_file)

if run:
    run.finish()
