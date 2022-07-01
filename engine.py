from tqdm import tqdm


def train(model, train_dl, valid_dl):
    model.train()
    for epoch in range(config.epochs):
        loop = tqdm(train_dl)
        for i, (img, label) in enumerate(loop):
            img = img.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch {epoch+1}/{config.epochs} Loss {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for i, (img, label) in enumerate(valid_dl):
                img = img.to(config.device)
                label = label.to(config.device)
                output = model(img)
                valid_loss += criterion(output, label).item()
            valid_loss /= len(valid_dl)
            print(f"Epoch {epoch+1}/{config.epochs} Valid Loss {valid_loss:.4f}")
            model.train()