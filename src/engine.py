import torch
import torch.nn as nn
from tqdm import tqdm


class Engine:
    def __init__(self, optimizer, model, device):
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def loss_fn(self, outputs, targets):
        # Binary Cross Entropy loss combines with Sigmoid layer
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def train_fn(self, train_loader, scheduler):
        """Loop over our training set and feed tensors inputs to BERT model and optimize
        Args:
            train_loader: iterable over a training dataset
            scheduler: adjust the learning rate based on the number of epochs
        Returns:
            [type]: final_targets and final_predictions values
        """
        # set training mode for training data
        self.model.train()
        final_targets, final_predictions = [], []
        for data in tqdm(train_loader):
            # fetch values from cutom dataset and convert to tensors
            ids = data["ids"].to(self.device, torch.long)
            masks = data["mask"].to(self.device, torch.long)
            token_type_ids = data["token_type_ids"].to(self.device, torch.long)
            targets = data["sentiment"].to(self.device, torch.float)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize + scheduler
            outputs = self.model(ids, masks, token_type_ids)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            # append to empty list
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_predictions.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_predictions

    def eval_fn(self, test_loader):
        """Loop over our testing set and feed tensors inputs to BERT model and optimize
        Args:
            test_loader: iterable over a testing dataset
        Returns:
            [list]: final_targets and final_predictions values
        """
        # set evaluation mode for testing set
        self.model.eval()
        final_targets, final_predictions = [], []
        # disables gradient calculation
        with torch.no_grad():
            for data in tqdm(test_loader):
                # fetch values from cutom dataset and convert to tensors
                ids = data["ids"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                mask = data["mask"].to(self.device, dtype=torch.long)
                targets = data["sentiment"].to(self.device, dtype=torch.float)
                # feed parameters to BERT model
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                # append to empty lists
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_predictions.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )
        return final_targets, final_predictions
