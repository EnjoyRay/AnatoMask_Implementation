import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / (std + 1e-5)

def extract_valid_patches(image, label, patch_size=(64,64,64), stride=(64,64,64)):
    D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches_img = []
    patches_lbl = []

    for z in range(0, D - pd + 1, sd):
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                img_patch = image[z:z+pd, y:y+ph, x:x+pw]
                lbl_patch = label[z:z+pd, y:y+ph, x:x+pw]
                if np.sum(lbl_patch) > 0:  # 非全背景
                    patches_img.append(np.expand_dims(img_patch, axis=0))  # [1, D, H, W]
                    patches_lbl.append(np.expand_dims(lbl_patch, axis=0))  # [1, D, H, W]

    return patches_img, patches_lbl

def preprocess_segmentation_amos(img_dir, lbl_dir, out_img_dir, out_lbl_dir, patch_size=(64,64,64)):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    total_classes = set()
    total_patches = 0

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])

    for fname in tqdm(files, desc="Generating patches"):
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname)

        img = nib.load(img_path).get_fdata().astype(np.float32)
        lbl = nib.load(lbl_path).get_fdata().astype(np.int64)

        img = normalize(img)

        patches_img, patches_lbl = extract_valid_patches(img, lbl, patch_size=patch_size)

        base_name = os.path.splitext(os.path.splitext(fname)[0])[0]
        for i, (pi, pl) in enumerate(zip(patches_img, patches_lbl)):
            np.save(os.path.join(out_img_dir, f"{base_name}_patch{i:03d}.npy"), pi.astype(np.float32))
            np.save(os.path.join(out_lbl_dir, f"{base_name}_patch{i:03d}.npy"), pl.astype(np.int64))
            total_classes.update(np.unique(pl))
            total_patches += 1

    print(f"\n✅ 总共生成 patch 数量: {total_patches}")
    print(f"✅ 所有标签中出现过的类别: {sorted(list(total_classes))}")

