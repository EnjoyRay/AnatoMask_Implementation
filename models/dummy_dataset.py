import torch
from torch.utils.data import Dataset

class Dummy3DMedicalDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(64, 64, 64), channels=1):
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.channels, *self.image_size)
        return x
