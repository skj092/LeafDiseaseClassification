import torch
from dataset import LeafDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import model
import config
import torch.nn as nn
import numpy as np
from engine import train_one_epoch
import argparse
from create_fold import get_fold
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug')
    parser.add_argument('-data_dir', '--data_dir')
    args = parser.parse_args()
    data_path = args.data_dir

    df = get_fold(data_path)

    train_df = df[df.kfold != 0].reset_index(drop=True)
    valid_df = df[df.kfold == 1].reset_index(drop=True)

    train_ds = LeafDataset(train_df, data_path, transforms=transforms.ToTensor())
    valid_ds = LeafDataset(valid_df, data_path, transforms=transforms.ToTensor())

    if args.debug:
        train_ds = Subset(train_ds, np.arange(100))
        valid_ds = Subset(valid_ds, np.arange(24))

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=True)

    model = model.to(config.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.N_EPOCHS):
        train_one_epoch(model, train_dl, valid_dl, loss_fn, optimizer)

    Path(config.MODEL_PATH).mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(Path(config.MODEL_PATH)/"model.pth"))
