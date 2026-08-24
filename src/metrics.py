"""Evaluation metrics (mIoU, precision, recall, F1, Boundary IoU), taken
line-for-line from the final training notebook
(notebooks/train_paper_models_seeds42-44.ipynb)."""

import cv2
import numpy as np
import torch

ROAD_CLASS_INDEX = 4

def _safe_div(num, den): return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)

def _confusion_update(conf_mat, pred, target, num_classes=9, ignore_index=255):
    valid = target != ignore_index
    if not np.any(valid): return
    encoded = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    conf_mat += np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)

def _mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(h*h + w*w))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=dilation)[1:h+1, 1:w+1]
    return (mask - eroded).astype(np.uint8)

def _boundary_update(boundary_inter, boundary_union, pred, target, num_classes=9, ignore_index=255):
    valid = target != ignore_index
    if not np.any(valid): return
    pred, target = pred.copy(), target.copy()
    pred[~valid], target[~valid] = 0, 0
    for cls in range(num_classes):
        gt_bin = ((target == cls) & valid).astype(np.uint8)
        pred_bin = ((pred == cls) & valid).astype(np.uint8)
        if gt_bin.sum() == 0 and pred_bin.sum() == 0: continue
        gt_b, pred_b = _mask_to_boundary(gt_bin), _mask_to_boundary(pred_bin)
        boundary_inter[cls] += np.logical_and(gt_b, pred_b).sum()
        boundary_union[cls] += np.logical_or(gt_b, pred_b).sum()

def _scores_from_stats(conf_mat, boundary_inter, boundary_union):
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    iou_per_class = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_per_class = _safe_div(2 * precision * recall, precision + recall)
    biou_per_class = _safe_div(boundary_inter, boundary_union)
    return {
        'miou': float(np.nanmean(iou_per_class)),
        'precision': float(np.nanmean(precision)),
        'recall': float(np.nanmean(recall)),
        'f1': float(np.nanmean(f1_per_class)),
        'bmiou': float(np.nanmean(biou_per_class)),
        'road_iou': float(iou_per_class[ROAD_CLASS_INDEX]) if len(iou_per_class) > ROAD_CLASS_INDEX else 0.0,
        'per_class_iou': iou_per_class.tolist()
    }

@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=9, ignore_index=255, is_dual=False):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    b_inter = np.zeros(num_classes, dtype=np.float64)
    b_union = np.zeros(num_classes, dtype=np.float64)
    for batch in dataloader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].long().cpu().numpy()
        logits_oem = model(images)[0] if is_dual else model(images)
        preds = torch.argmax(logits_oem, dim=1).cpu().numpy()
        for p, t in zip(preds, labels):
            _confusion_update(conf_mat, p, t, num_classes, ignore_index)
            _boundary_update(b_inter, b_union, p, t, num_classes, ignore_index)
    return _scores_from_stats(conf_mat, b_inter, b_union)
