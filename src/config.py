from typing import Any
from dataclasses import dataclass
import json
import torch
from torchvision import transforms

batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.001
MODEL_PATH = "models"
N_EPOCHS = 5

train_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
valid_tfms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(),])

# config.py


@dataclass
class Config:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    num_epochs: int = 10
    data_path: str = "default_data/"
    lr: float = 0.001
    MODEL_PATH: str = "/home/sonujha/rnd/LeafDiseaseClassification/models/"
    train_tfms: Any = train_tfms
    valid_tfms: Any = valid_tfms
    device: str = "cpu"
    env: str = "local"  # Added to handle environment-specific overrides
    n_fold: int = 2

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """
        Creates a Config instance from a JSON file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def override_with_env(self, env: str):
        """
        Overrides configuration parameters based on the environment.
        """
        if env == 'local':
            self.batch_size = min(self.batch_size, 4)
            self.num_workers = 0
            self.pin_memory = False
            self.num_epochs = 2
            self.data_path = "data/"
        elif env == 'kaggle':
            self.batch_size = min(self.batch_size, 32)
            self.num_workers = 2
            self.pin_memory = False
            self.num_epochs = 10
            self.data_path = "/kaggle/input/cassava-leaf-disease-classification"
        else:
            raise ValueError(f"Unknown environment: {env}")
