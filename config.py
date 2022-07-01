import torch 

epochs = 2
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.01