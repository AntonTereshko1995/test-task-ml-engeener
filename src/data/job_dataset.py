import torch

class JobDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, features, device):
        self.encodings = encodings
        self.features = torch.tensor(features, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.features[idx]
        return item