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
