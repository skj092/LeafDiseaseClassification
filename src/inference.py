# evaluate the test data
from torch.utils.data import Dataset, DataLoader, Subset
import os
import torch
from torchvision import models
import pandas as pd
import torch.nn as nn
from glob import glob
from PIL import Image
from torchvision import transforms
import kagglehub
import numpy as np


class config:
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    def predict_batch(model, test_dl):
        preds = []
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dl):
                img = img.to(config.device)
                output = model(img)
                pred = torch.argmax(output, dim=1).numpy().tolist()
                preds.extend(pred)
        return preds


class LeafDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.images = glob(data_dir + "/*")
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(0)


def predict_batch(model, test_dl):
    preds = []
    with torch.no_grad():
        for i, (img, label) in enumerate(test_dl):
            img = img.to(config.device)
            output = model(img)
            pred = torch.argmax(output, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
    return preds


if __name__ == "__main__":
    data_dir = "/home/sonujha/rnd/LeafDiseaseClassification/data/"
    test_image_path = os.path.join(data_dir, "test_images")
    test_dataset = LeafDataset(test_image_path, transforms=config.valid_tfms)
    test_dl = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)

    # Download latest version
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(512, 5)
    model.to(config.device)

    model_dir = kagglehub.dataset_download("sonujha090/cassava-models")
    models = []
    for model_name in os.listdir(model_dir):
        state = torch.load(os.path.join(model_dir, model_name),
                           weights_only=True, map_location=config.device)
        model.load_state_dict(state['state_dict'])
        model.eval()
        models.append(model)

    # Ensemble Predictions
    model_preds = []
    for i in range(len(models)):
        preds = predict_batch(models[i], test_dl)
        model_preds.append(preds)
    preds = np.stack(model_preds).mean(axis=0)
    # print(preds)

    # submission file
    sample_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    sample_df["image_id"] = os.listdir(test_image_path)
    sample_df["label"] = preds
    df = pd.DataFrame(sample_df)
    df.to_csv("submission.csv", index=False)
    print(df.head())
