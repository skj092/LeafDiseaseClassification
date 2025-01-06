import os
import torch
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from tqdm import tqdm
from torch import nn
import numpy as np
import random
import logging
import sys
from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip,
    VerticalFlip, ShiftScaleRotate, Transpose,
)
import cv2
from albumentations.pytorch import ToTensorV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]



def setup_logger(log_file='training_log.log', level=logging.INFO):
    """
    Sets up the logger to log to both a file and the console.

    Args:
    - log_file (str): The name of the log file. Defaults to 'training_log.log'.
    - level (int): The logging level. Defaults to logging.INFO.

    Returns:
    - logger: A logger instance.
    """
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # tf.set_random_seed(seed)


class CassavaDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.tfms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id, label, _ = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.tfms:
            img = self.tfms(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


def get_model():
    model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
    model.fc = nn.Linear(512, 5)
    return model


def train_one_epoch(model, train_dl, valid_dl, loss_fn, optim, logger):
    model.train()
    train_loss, train_acc = 0, 0
    loop = tqdm(train_dl)
    for i, (xb, yb) in enumerate(loop):
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad()
        logit = model(xb)
        loss = loss_fn(logit, yb)
        loss.backward()
        train_loss += loss.item()
        optim.step()
        train_acc += (torch.argmax(logit, dim=1) == yb).float().mean().item()

    # Evaluation on one epoch
    model.eval()
    valid_loss, valid_acc = 0, 0
    loop = tqdm(valid_dl)
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loop):
            xb = xb.to(device)
            yb = yb.to(device)
            logit = model(xb)
            valid_loss += loss_fn(logit, yb).item()
            valid_acc += (torch.argmax(logit, dim=1)
                          == yb).float().mean().item()
    train_acc /= len(train_dl)
    valid_acc /= len(valid_dl)
    train_loss /= len(train_dl)
    valid_loss /= len(valid_dl)
    logger.info(f"train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}")
    logger.info(f"train_acc: {train_acc:.2f}, valid_acc: {valid_acc:.2f}")
    return train_loss, valid_loss, train_acc, valid_acc


def get_transform(is_train=True):
    if is_train:
        return Compose([
            RandomResizedCrop(384, 384),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])
    else:
        return Compose([
            Resize(384, 384),
            Normalize(
                mean=MEAN,
                std=STD,
            ),
            ToTensorV2(),
        ])

