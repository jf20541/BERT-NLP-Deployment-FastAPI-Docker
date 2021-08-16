import torch


class IMDBDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx, :], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }
