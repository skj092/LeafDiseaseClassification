from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import torch


class LeafDataset(Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

        img_path = os.path.join(self.data_dir, 'train_images', img_name)
        img = Image.open(img_path).resize((224, 224))
        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    ds = LeafDataset(df, transforms=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    for i, (img, label) in enumerate(dl):
        print(img.shape, label.shape)
        if i == 0:
            break
