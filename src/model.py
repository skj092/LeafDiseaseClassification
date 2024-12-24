from torchvision.models import resnet18
import torch.nn as nn
from dataset import LeafDataset
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score

import sys

model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 5)

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    ds = LeafDataset(df, transforms=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    for i, (img, label) in enumerate(dl):
        print(img.shape, label.shape)
        print('image batch shape', img.shape)
        print('label batch shape', label.shape)
        out = model(img)
        print('model output shape', out.shape)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(out, label)
        print('loss = ', loss)
        output_label = torch.argmax(out, dim=1)
        print('output label ', output_label)
        print('actual label', label)
        acc = accuracy_score(torch.argmax(out, dim=1), label)
        print('accuracy = ', acc)
        sys.exit()
