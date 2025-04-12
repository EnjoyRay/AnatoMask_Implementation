import torch.nn.functional as F
import torch.nn as nn
import torch

def dice_loss(pred, target, smooth=1.):
    """
    pred: [B, C, D, H, W]  - logits
    target: [B, D, H, W]   - int64 labels
    """
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]

    # One-hot encode ground truth
    target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

    intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

def segmentation_loss(pred, target):
    ce = nn.CrossEntropyLoss()(pred, target)
    dsc = dice_loss(pred, target)
    return ce + dsc

