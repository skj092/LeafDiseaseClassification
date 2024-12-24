import torch

batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01
MODEL_PATH = "models/model.pth"
N_EPOCHS = 5
