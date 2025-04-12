import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)  # [B, H, W, D, C]
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, H, W, D]

    intersect = (pred * one_hot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + one_hot.sum(dim=(2, 3, 4))
    dice = (2 * intersect + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_full_segmentation(model, dataloader, optimizer, device, epochs=20,
                            loss_fn="dice_ce", save_path="checkpoints/segmentation_finetune.pth"):
    model.train()
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")

    # ✅ [新增] 计算类别权重，缓解类别不均衡
    label_dir = "/root/lanyun-tmp/amos_dataset/segmentation_npy/labels"
    cls_hist = np.zeros(16)
    for f in os.listdir(label_dir):
        y = np.load(os.path.join(label_dir, f))
        cls_hist += np.bincount(y.flatten(), minlength=16)

    cls_weights = 1.0 / (cls_hist + 1e-6)           # 反比权重
    cls_weights = cls_weights / cls_weights.sum()   # 归一化
    cls_weights = torch.tensor(cls_weights, dtype=torch.float32).to(device)
    loss_ce_weighted = nn.CrossEntropyLoss(weight=cls_weights)

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for images, labels in loop:
            images = images.to(device)                # [B, 1, H, W, D]
            labels = labels.to(device).squeeze(1)     # [B, H, W, D]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images)                 # [B, C, H, W, D]

                loss = 0
                if loss_fn == "dice_ce":
                    # ✅ [替换] 使用加权交叉熵
                    loss += loss_ce_weighted(preds, labels)
                    loss += dice_loss(preds, labels)
                elif loss_fn == "ce":
                    loss += loss_ce_weighted(preds, labels)
                elif loss_fn == "dice":
                    loss += dice_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")

