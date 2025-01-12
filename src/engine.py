import torch
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(
            self,
            model,
            train_dl,
            valid_dl,
            loss_fn,
            optimizer,
            scheduler=None,
            logger=None,
            accelerator=None,
            model_dir='models',
            run=None
    ):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.logger = logger
        self.accelerator = accelerator
        self.model_dir = Path(model_dir)
        self.run = run
        self.model_dir.mkdir(exist_ok=True)
        self.scheduler = scheduler
        self.best_loss = float('inf')

    def train_one_epoch(self):
        self.model.train()
        train_loss, train_acc = 0, 0
        loop = tqdm(self.train_dl)
        for i, (xb, yb) in enumerate(loop):
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            self.optimizer.zero_grad()
            logit = self.model(xb)
            loss = self.loss_fn(logit, yb)
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss += loss.item()
            train_acc += (torch.argmax(logit, dim=1)
                          == yb).float().mean().item()
        return train_loss / len(self.train_dl), train_acc / len(self.train_dl)

    def validate_one_epoch(self):
        self.model.eval()
        valid_loss, valid_acc = 0, 0
        loop = tqdm(self.valid_dl)
        with torch.no_grad():
            for i, (xb, yb) in enumerate(loop):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logit = self.model(xb)
                valid_loss += self.loss_fn(logit, yb).item()
                valid_acc += (torch.argmax(logit, dim=1)
                              == yb).float().mean().item()

        return valid_loss / len(self.valid_dl), valid_acc / len(self.valid_dl)

    def fit(self, num_epochs, fold=0):
        self.device = self.accelerator.device if self.accelerator else 'cpu'
        for epoch in range(num_epochs):
            if self.logger:
                self.logger.info(
                    f"============Epoch: {epoch}/{num_epochs}============")

            # Training phase
            train_loss, train_acc = self.train_one_epoch()

            # Validation phase
            valid_loss, valid_acc = self.validate_one_epoch()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.logger:
                self.logger.info(
                    f"train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}")
                self.logger.info(
                    f"train_acc: {train_acc:.2f}, valid_acc: {valid_acc:.2f}")
                self.logger.info(f"current_lr: {current_lr:.6f}")

            if self.run:
                self.run.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'learning_rate': current_lr
                })

            # Save best model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                model_file = self.model_dir / f"model_{fold}.pth"
                print('========New optimal found, saving state==========')

                state = {
                    'epoch': epoch,
                    'best_loss': self.best_loss,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(state, model_file)
