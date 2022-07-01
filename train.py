import pandas as pd 
import torch
from dataset import LeafDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import model 
import config 
import torch.nn as nn
import numpy as np 




if __name__=="__main__":
    df = pd.read_csv("data/train_fold.csv")

    train_df = df[df.kfold != 0].reset_index(drop=True)
    valid_df = df[df.kfold == 1].reset_index(drop=True)

    train_ds = LeafDataset(train_df, transforms=transforms.ToTensor())
    valid_ds = LeafDataset(valid_df, transforms=transforms.ToTensor())

    train_ds = Subset(train_ds, np.arange(1024))
    valid_ds = Subset(valid_ds, np.arange(256))

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=True)

    model = model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    


