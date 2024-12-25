from tqdm import tqdm
import config
import torch
import wandb


def train_one_epoch(model, train_dl, valid_dl, criterion, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    loop = tqdm(train_dl)
    for i, (img, label) in enumerate(loop):
        img = img.to(config.device)
        label = label.to(config.device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        train_acc += (torch.argmax(output, dim=1) ==
                      label).float().mean().item()
    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for i, (img, label) in enumerate(valid_dl):
            img = img.to(config.device)
            label = label.to(config.device)
            output = model(img)
            valid_loss += criterion(output, label).item()
            valid_acc += (torch.argmax(output, dim=1) ==
                          label).float().mean().item()
    train_loss = train_loss / len(train_dl)
    valid_loss = valid_loss / len(valid_dl)
    train_acc = train_acc / len(train_dl)
    valid_acc = valid_acc / len(valid_dl)
    print(f"train_loss {train_loss:.3f}, valid_loss {valid_loss:.3f}")
    print(f"train_acc {train_acc:.3f}, valid_acc {valid_acc:.3f}")
    wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
              "train_accuracy": train_acc, "valid_accuracy": valid_acc})

    return train_loss, valid_loss, train_acc, valid_acc


def predict_batch(model, test_dl):
    preds = []
    with torch.no_grad():
        for i, (img, label) in enumerate(test_dl):
            img = img.to(config.device)
            output = model(img)
            pred = torch.argmax(output, dim=1).numpy().tolist()
            preds.extend(pred)
    return preds
