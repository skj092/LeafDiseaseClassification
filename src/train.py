import torch
import dataset
import engine
import pandas as pd
from model import get_model, LeafModel
from torch.utils.data import DataLoader


def run(fold):

    bs = 64
    epochs = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("input/train_fold.csv")

    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    train_ds = dataset.LeafData(train_df, dir="input/train_images/")
    valid_ds = dataset.LeafData(valid_df, dir="input/train_images")

    train_dl = torch.utils.data.DataLoader(train_ds, bs)
    valid_dl = torch.utils.data.DataLoader(valid_ds, bs)

    model = get_model()
    # model = LeafModel(df.label.nunique())
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.7, gamma=0.5)

    best_accuracy = 0
    for epoch in range(epochs):
        train_loss = engine.Train(train_ds, train_dl, model, optimizer, device)
        valid_loss, valid_acc = engine.Evaluate(
            valid_ds, valid_dl, model, optimizer, device
        )
        scheduler.step()
        print(
            f"fold = {fold}, epoch={epoch}, train loss = {train_loss}\
        valid_loss = {valid_loss}, accuracy={valid_acc}"
        )
        if valid_acc > best_accuracy:
            torch.save(model.state_dict(), "models/model.pt")
            best_accuracy = valid_acc


if __name__ == "__main__":
    run(0)
