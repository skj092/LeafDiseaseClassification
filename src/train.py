 import torch 
 import dataset
 import engine
 from torch.utils.data import DataLoader
 
 def train(fold):

    bs = 64
    epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    train_ds = LeafData(train_df)
    valid_ds = LeafData(valid_df)

    train_dl = torch.utils.data.DataLoader(train_ds, bs)
    valid_dl = torch.utils.data.DataLoader(valid_ds, bs)

    model = get_model(pretrained=True)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        engine.Train(train_ds, train_dl, model, optimizer, device)
        engine.Evaluate(valid_ds, valid_dl, model, optimizer, device)

if __name__ == "__main__":
    train(0)