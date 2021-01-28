import torch.nn as nn 
from sklearn import metrics
from tqdm import tqdm


def Train(dataset, dataloader, model, optimizer, device):
    model.train()
    num_batches = int(len(dataset)/dataloader.batch_size)
    tk0 = tqdm(dataloader, total = num_batches)
    for step, batch in enumerate(tk0):
        images = batch['image']
        labels = batch['label']
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        optimizer.step()
    tk0.close()
def Evaluate(dataset,dataloader, model, optimizer, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    num_batches = int(len(dataset)/dataloader.batch_size)
    tk0 = tqdm(dataloader,  total = num_batches)
    for step, batch in enumerate(tk0):
        images = batch['image']
        labels = batch['label']
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
    print({'Validation Loss: ': val_loss/num_batches, 'Accuracy': val_acc/num_batches})

def Predict(dataloader, model, optimizer, device):
    model.eval()
    predictions = []
    tk0 = tqdm(dataloader, total = num_batches)
    with torch.no_grad():
        for step, batch in enumerate(tk0):
            images = batch['image']
            labels = batch['label']
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            pred = torch.argmax(output).cpu()
            predictions.append(pred)
    tk0.close()
    return np.array(predictions)