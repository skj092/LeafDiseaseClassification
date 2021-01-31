import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np


def Train(dataset, dataloader, model, optimizer, device):
    model.train()
    num_batches = int(len(dataset) / dataloader.batch_size)
    tr_loss = 0
    tk0 = tqdm(dataloader, total=num_batches)
    for step, batch in enumerate(tk0):
        images = batch["image"]
        labels = batch["label"]
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        tr_loss += loss
        loss.backward()
        optimizer.step()
    tk0.close()
    return tr_loss / num_batches


def Evaluate(dataset, dataloader, model, optimizer, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    num_batches = int(len(dataset) / dataloader.batch_size)
    tk0 = tqdm(dataloader, total=num_batches)
    for step, batch in enumerate(tk0):
        images = batch["image"]
        labels = batch["label"]
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, labels)
            val_loss += loss
            pred = torch.argmax(output, dim=1).cpu()
            target = labels.cpu()
            accuracy = metrics.accuracy_score(pred, target)
            val_acc += accuracy
    tk0.close()
    return val_loss / num_batches, val_acc / num_batches


def Predict(dataset, dataloader, model, device):
    model.eval()
    predictions = []
    num_batches = int(len(dataset) / dataloader.batch_size)
    tk0 = tqdm(dataloader, total=num_batches)
    with torch.no_grad():
        for step, batch in enumerate(tk0):
            images = batch["image"]
            labels = batch["label"]
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred = torch.argmax(output).cpu()
            predictions.append(pred)
    tk0.close()
    return np.array(predictions)
