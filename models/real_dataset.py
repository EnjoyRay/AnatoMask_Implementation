import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random

class Real3DMedicalDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        patch = np.load(file_path)  # shape: (1, D, H, W)
        return torch.from_numpy(patch).float()


class SegmentationPatchDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".npy")
        ])
        assert len(self.image_paths) == len(self.label_paths), "Image/Label count mismatch."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])  # shape: (1, D, H, W)
        label = np.load(self.label_paths[idx])  # shape: (D, H, W)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label



class RareClassPatchDataset(Dataset):
    def __init__(self, image_dir, label_dir, focus_foreground=True):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".npy")
        ])
        assert len(self.image_paths) == len(self.label_paths)

        self.focus_foreground = focus_foreground
        if self.focus_foreground:
            # 过滤出至少包含一个前景像素的 patch
            self.fg_indices = []
            for i, path in enumerate(self.label_paths):
                label = np.load(path)
                if (label > 0).any():  # 存在非背景类
                    self.fg_indices.append(i)

    def __len__(self):
        return len(self.fg_indices) if self.focus_foreground else len(self.image_paths)

    def __getitem__(self, idx):
        true_idx = self.fg_indices[idx] if self.focus_foreground else idx
        image = np.load(self.image_paths[true_idx])     # (1, D, H, W)
        label = np.load(self.label_paths[true_idx])     # (D, H, W)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label
