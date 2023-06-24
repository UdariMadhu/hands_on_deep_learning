import torch


class VanillaTrainer:
    """
    A basic trainer class to train a model.
    """

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 criterion,
                 lrs,
                 device,
                 print_freq=50,
                 verbose=False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lrs = lrs
        self.print_freq = print_freq
        self.verbose = verbose

    def train_epoch(self):
        # must call model.train() before training, and model.eval() before inference as some modules behave differently in
        # training and evaluation mode. E.g., dropout and batch normalization layers behave differently in training and
        # inference mode. Dropout is turned off in eval mode, and batch normalization uses running statistics in eval mode, i.e.,
        # it uses the mean and variance of the entire training set instead of the mean and variance of the current batch.
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % self.print_freq == 0 and self.verbose:
                print(
                    f'Batch {batch_idx}/{len(self.train_loader)}: train_loss '
                    +
                    f'{running_loss / (batch_idx + 1):.4f}, train_acc {100.0 * correct / total:.2f}, lr: {self.lrs.get_lr()[0]:.5f}'
                )
        self.lrs.step()
        return running_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        # must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Crucial to add no_grad() when evaluating the model, otherwise pytorch will try to maintain
        # the computation graph for backpropagation and will run out of memory
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if ((batch_idx + 1) % self.print_freq == 0 or
                    (batch_idx + 1) == len(self.val_loader)) and self.verbose:
                    print(
                        f'Batch {batch_idx}/{len(self.val_loader)}: val_loss '
                        +
                        f'{running_loss / (batch_idx + 1):.4f}, val_acc {100.0 * correct / total:.2f}'
                    )

        return running_loss / len(self.val_loader), 100.0 * correct / total

    def train_all_epochs(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            if self.verbose:
                print(
                    f'Epoch {epoch + 1}/{epochs}: train_loss {train_loss:.4f}, train_acc {train_acc:.2f}, '
                    + f'val_loss {val_loss:.4f}, val_acc {val_acc:.2f}')
