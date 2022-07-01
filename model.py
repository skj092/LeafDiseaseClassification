from torchvision.models import resnet18
import torch.nn as nn
from dataset import LeafDataset
import pandas as pd 
from torchvision import transforms
from torch.utils.data import DataLoader

model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 4)   

if __name__=="__main__":
    df = pd.read_csv("data/train.csv")
    ds = LeafDataset(df, transforms=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    for i, (img, label) in enumerate(dl):
        print(img.shape, label.shape)
        print(model(img).shape)
        if i == 0:
            break
