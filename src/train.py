from pathlib import Path
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import StratifiedKFold
import albumentations as A
import time
from torch.utils.data import DataLoader
from modules import (setup_logger, seed_everything,
                     CassavaDataset, get_model, train_one_epoch,
                     get_transform)


tik = time.time()


logger = setup_logger()
seed_everything(42)
use_external = False
multi_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_path = Path("/home/sonujha/rnd/LeafDiseaseClassification/data/merged")
studio_path = Path('/teamspace/studios/this_studio/LeafDiseaseClassification/data/')

# lightning studio specific
path = studio_path if studio_path.exists() else local_path
df = pd.read_csv(path / "merged.csv")
logger.info(f"shape of df: {df.shape}")


df2020 = df[df["source"] == 2020]
# df2020 = df2020.sample(frac=0.05)
skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42,
)
for train_index, valid_index in skf.split(df2020, df2020["label"].values):
    train_df = df2020.iloc[train_index] if not use_external else pd.concat(
        [df2020.iloc[train_index], df[df["source"] == 2019]], axis=0)
    valid_df = df2020.iloc[valid_index]

    # Get train and valid dataset, dataloader
    train_ds = CassavaDataset(
        train_df, path / "train", transforms=get_transform())
    valid_ds = CassavaDataset(
        valid_df, path / "train", transforms=get_transform(is_train=False))

    train_dl = DataLoader(train_ds, batch_size=196, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=196, shuffle=False)

    # Load the model
    model = get_model()
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # optimizer and loss functions
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        logger.info(f"============Epoch: {epoch}/10 ============")
        train_loss, valid_loss, train_acc, valid_acc = train_one_epoch(
            model, train_dl, valid_dl, loss_fn, optim, logger
        )

    # save the model
    torch.save(model.state_dict(), "model.pth")
    tok = time.time()
    logger.info(f"Total time take {tok-tik:.2f}s")
