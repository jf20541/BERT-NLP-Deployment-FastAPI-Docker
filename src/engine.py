import torch
import torch.nn as nn


class Engine:
    def __init__(self, optimizer, model, device):
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def loss_fn(self, outputs, targets):
        return nn.BCELoss()(outputs, targets.view(-1, 1))

    def train_fn(self, train_loader, scheduler):
        self.model.train()
        # final_targets, final_predictions = [], []
        for bi, d in enumerate(train_loader, len(train_loader)):
            ids = d["ids"].to(self.device, torch.long)
            masks = d["mask"].to(self.device, torch.long)
            token_type_ids = d["token_type_ids"].to(self.device, torch.long)
            targets = d["sentiment"].to(self.device, torch.float)
            self.optimizer.zero_grad()
            outputs = self.model(ids, masks, token_type_ids)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

    def eval_fn(self, test_loader):
        self.model.eval()
        final_targets, final_predictions = [], []
        with torch.no_grad():
            for bi, d in enumerate(test_loader, len(test_loader)):
                ids = d["ids"].to(self.device, torch.long)
                masks = d["mask"].to(self.device, torch.long)
                token_type_ids = d["token_type_ids"].to(self.device, torch.long)
                targets = d["sentiment"].to(self.device, torch.float)
                outputs = self.model(ids, masks, token_type_ids)
                final_targets.extend(targets.cpu().deatch().numpy().list())
                final_predictions.extend(outputs.cpu().deatch().numpy().list())
        return final_targets, final_predictions
