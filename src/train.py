from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import models
from PIL import Image
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import time
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # tf.set_random_seed(seed)


seed_everything(42)
torch.backends.cudnn.benchmark = True


class CassavaDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.tfms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id, label = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, img_id)

        img = Image.open(img_path)
        img = np.array(img)
        if self.tfms:
            img = self.tfms(image=img)["image"].transpose(2, 0, 1)
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)


def get_model():
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model.fc = nn.Linear(512, 5)
    return model


def train_one_epoch(model, train_dl, valid_dl, loss_fn, optim):
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
            valid_acc += (torch.argmax(logit, dim=1) == yb).float().mean().item()
    train_acc /= len(train_dl)
    valid_acc /= len(valid_dl)
    train_loss /= len(train_dl)
    valid_loss /= len(valid_dl)
    print(f"train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}")
    print(f"train_acc: {train_acc:.2f}, valid_acc: {valid_acc:.2f}")
    return train_loss, valid_loss, train_acc, valid_acc


tik = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = Path("/kaggle/input/cassava-leaf-disease-classification")
df = pd.read_csv(path / "train.csv")
# df = df.sample(frac=0.05)
print(f"shape of df: {df.shape}")
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df["label"])

# trand and valid transforms
train_tfms = A.Compose(
    [
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
valid_tfms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Get train and valid dataset, dataloader
train_ds = CassavaDataset(train_df, path / "train_images", transforms=train_tfms)
valid_ds = CassavaDataset(valid_df, path / "train_images", transforms=valid_tfms)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False)

model = get_model()
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
for epoch in range(10):
    print(f"============Epoch: {epoch}/10 ============")
    train_loss, valid_loss, train_acc, valid_acc = train_one_epoch(
        model, train_dl, valid_dl, loss_fn, optim
    )
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

# save the model
torch.save(model.state_dict(), "resnet18.pth")
tok = time.time()
print(f"Total time take {tok-tik:.2f}s")

# Plot the graph
plt.plot(train_losses, color="blue", label="train loss")
plt.plot(valid_losses, color="red", label="valid loss")
plt.xlabel("epochs")
plt.ylabel("score")
plt.legend()
plt.show()

plt.plot(train_accuracies, color="blue", label="train accuracy")
plt.plot(valid_accuracies, color="red", label="valid accuracy")
plt.xlabel("epochs")
plt.ylabel("score")
plt.legend()
plt.show()
