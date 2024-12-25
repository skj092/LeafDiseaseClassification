from torchvision import models
import torch.nn as nn
from dataset import LeafDataset
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from config import Config

import sys


# model.py

class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        # Example: Return a model based on config parameters
        # Replace with your actual model initialization
        model = models.resnet18(pretrained=False, num_classes=10)
        return model


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    ds = LeafDataset(df, 'data', transforms=transforms.ToTensor())
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
