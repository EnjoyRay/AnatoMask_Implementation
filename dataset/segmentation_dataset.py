import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, target_size=(8, 8, 8)):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        lbl_path = os.path.join(self.lbl_dir, self.img_files[idx])

        img = np.load(img_path).astype(np.float32)  # [1, 64, 64, 64]
        lbl = np.load(lbl_path).astype(np.int64)    # [1, 64, 64, 64]

        # To tensor
        img = torch.from_numpy(img)  # [1, D, H, W]
        lbl = torch.from_numpy(lbl).float()

        # Downsample label to match encoder output
        lbl_down = F.interpolate(lbl.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0).squeeze(0).long()

        return img, lbl_down
