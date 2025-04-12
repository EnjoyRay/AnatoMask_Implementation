import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize(volume):
    """Z-score normalization"""
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / (std + 1e-5)

def extract_patches(volume, patch_size=(64, 64, 64), stride=(64, 64, 64)):
    """Extract non-overlapping or sliding patches from 3D volume"""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for z in range(0, D - pd + 1, sd):
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patch = volume[z:z+pd, y:y+ph, x:x+pw]
                if np.sum(patch) > 1e-5:  # skip blank patches
                    patches.append(patch)

    return patches

def preprocess_amos_nii_to_npy(input_dir, output_dir, patch_size=(64, 64, 64)):
    os.makedirs(output_dir, exist_ok=True)
    nii_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nii.gz")])

    for fname in tqdm(nii_files, desc="Preprocessing AMOS"):
        path = os.path.join(input_dir, fname)
        try:
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.float32)
    
            data = normalize(data)
            patches = extract_patches(data, patch_size=patch_size)
    
            base_name = os.path.splitext(os.path.splitext(fname)[0])[0]
            for i, patch in enumerate(patches):
                patch = np.expand_dims(patch, axis=0)
                save_name = f"{base_name}_patch{i:03d}.npy"
                np.save(os.path.join(output_dir, save_name), patch)
    
        except Exception as e:
            print(f"[❌ ERROR] Skipping {fname}: {e}")

