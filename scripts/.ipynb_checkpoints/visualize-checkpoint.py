import matplotlib.pyplot as plt
import torch

def show_slice_comparison(original, masked, reconstructed, slice_idx=None, axis=2):
    """
    original/masked/reconstructed: tensors with shape [1, 1, D, H, W]
    axis: 0-Z, 1-Y, 2-X
    """
    original = original.cpu().squeeze().numpy()
    masked = masked.cpu().squeeze().numpy()
    reconstructed = reconstructed.cpu().squeeze().numpy()

    if slice_idx is None:
        slice_idx = original.shape[axis] // 2

    if axis == 0:
        ori_slice = original[slice_idx, :, :]
        msk_slice = masked[slice_idx, :, :]
        rec_slice = reconstructed[slice_idx, :, :]
    elif axis == 1:
        ori_slice = original[:, slice_idx, :]
        msk_slice = masked[:, slice_idx, :]
        rec_slice = reconstructed[:, slice_idx, :]
    else:
        ori_slice = original[:, :, slice_idx]
        msk_slice = masked[:, :, slice_idx]
        rec_slice = reconstructed[:, :, slice_idx]

    plt.figure(figsize=(12, 4))
    for i, (title, img) in enumerate(zip(
        ["Original", "Masked Input", "Reconstructed"],
        [ori_slice, msk_slice, rec_slice]
    )):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_slice_with_error(original, masked, reconstructed, slice_idx=None, axis=2):
    original = original.cpu().squeeze().numpy()
    masked = masked.cpu().squeeze().numpy()
    reconstructed = reconstructed.cpu().squeeze().numpy()
    error = abs(original - reconstructed)

    if slice_idx is None:
        slice_idx = original.shape[axis] // 2

    if axis == 0:
        slices = [arr[slice_idx, :, :] for arr in [original, masked, reconstructed, error]]
    elif axis == 1:
        slices = [arr[:, slice_idx, :] for arr in [original, masked, reconstructed, error]]
    else:
        slices = [arr[:, :, slice_idx] for arr in [original, masked, reconstructed, error]]

    titles = ["Original", "Masked Input", "Reconstructed", "Reconstruction Error"]

    plt.figure(figsize=(14, 6))
    for i, (title, img) in enumerate(zip(titles, slices)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
