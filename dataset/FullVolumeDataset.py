import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class FullVolumeDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        lbl_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

        # 取文件名不带扩展名
        img_basenames = set([f.replace(".nii.gz", "") for f in img_files])
        lbl_basenames = set([f.replace(".nii.gz", "") for f in lbl_files])
        common_basenames = sorted(img_basenames & lbl_basenames)

        self.image_paths = [os.path.join(image_dir, f"{name}.nii.gz") for name in common_basenames]
        self.label_paths = [os.path.join(label_dir, f"{name}.nii.gz") for name in common_basenames]

        self.transform = transform
        print(f"[FullVolumeDataset] Loaded {len(self.image_paths)} matched samples.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.int64)

        image = np.expand_dims(image, axis=0)  # [1, H, W, D]

        if self.transform:
            image, label = self.transform(image, label)

        return torch.from_numpy(image), torch.from_numpy(label)
