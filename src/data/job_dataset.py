
import torch


class JobDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["features"] = self.labels[idx]
        return item