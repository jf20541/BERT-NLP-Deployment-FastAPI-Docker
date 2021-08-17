import torch
import torch.nn as nn
from tqdm import tqdm
import config


class Engine:
    def __init__(self, optimizer, model, device):
        self.optimizer = optimizer
        self.model = model
        self.device = config.DEVICE

    def loss_fn(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def train_fn(self, train_loader, scheduler):
        self.model.train()
        final_targets, final_predictions = [], []
        for bi, d in tqdm(enumerate(train_loader), total=len(train_loader)):
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

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_predictions.extend(outputs.cpu().detach().numpy().tolist())
        return final_targets, final_predictions

    def eval_fn(self, test_loader):
        self.model.eval()
        final_targets, final_predictions = [], []
        with torch.no_grad():
            for bi, d in tqdm(enumerate(test_loader), total=len(test_loader)):
                ids = d["ids"].to(self.device, dtype=torch.long)
                token_type_ids = d["token_type_ids"].to(self.device, dtype=torch.long)
                mask = d["mask"].to(self.device, dtype=torch.long)
                targets = d["sentiment"].to(self.device, dtype=torch.float)
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_predictions.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )
        return final_targets, final_predictions
