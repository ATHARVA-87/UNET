# src/utils/metrics.py

import torch


def dice_score(preds, targets, smooth=1e-6):
    """Dice Coefficient"""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(preds, targets, smooth=1e-6):
    """IoU (Jaccard Index)"""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def accuracy(preds, targets):
    """Binary pixel-wise accuracy"""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()
