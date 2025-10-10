import numpy as np
import pandas as pd

def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Parameters
    ----------
    box1 : (N, 4)
        Predicted boxes [x1, y1, x2, y2].
    box2 : (M, 4)
        Ground-truth boxes [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray : (N, M)
        IoU matrix, IoU[i, j] = IoU(box1[i], box2[j])
    """
    if box1.size == 0 or box2.size == 0:
        return np.zeros((len(box1), len(box2)))

    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area + 1e-8

    return inter_area / union

def match_image_predictions(df_pred_img: pd.DataFrame, row_gt_img: pd.DataFrame, iou_thres: float = 0.5) -> np.ndarray:
    """
    Match predictions to ground truth boxes for a single image.

    Parameters
    ----------
    df_pred_img : pd.DataFrame
        Predicted boxes for one image (must have columns ['x1','y1','x2','y2']).
    row_gt_img : pd.DataFrame
        Ground-truth boxes for that image (must have columns ['x1','y1','x2','y2']).
    iou_thres : float
        IoU threshold to consider a match (default=0.5).

    Returns
    -------
    np.ndarray
        Binary array (len = len(df_pred_img)): 1 for TP, 0 for FP.
    """
    if row_gt_img.empty or df_pred_img.empty:
        return np.zeros(len(df_pred_img), dtype=int)

    boxes_pred = df_pred_img[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 4) # N, 4
    boxes_gt = row_gt_img[["x1", "y1", "x2", "y2"]].to_numpy().reshape(-1, 4) # 1, 4

    ious = box_iou(boxes_pred, boxes_gt).ravel() # N
    tp_flags = np.zeros(len(df_pred_img), dtype=int)
    ok = ious >= iou_thres

    if ok.any():
        first_hit = np.flatnonzero(ok)[0]
        tp_flags[first_hit] = 1

    return tp_flags

def compute_tp_flags(df_pred: pd.DataFrame, df_gt: pd.DataFrame, iou_thres: float = 0.5):
    """
    Compute binary TP flags for all predictions in df_pred
    by matching with df_gt (IoU ≥ threshold).

    Parameters
    ----------
    df_pred : pd.DataFrame
        Predicted boxes, columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    df_gt : pd.DataFrame
        Ground-truth boxes, columns ['img', 'cls', 'x1', 'y1', 'x2', 'y2'].
    iou_thres : float
        IoU threshold (default=0.5).
    """
    if "img" not in df_gt.columns or "img" not in df_pred.columns:
        raise ValueError("Both df_gt and df_pred must contain column 'img'.")

    # --- Primary key validation ---
    duplicated = df_gt["img"].duplicated(keep=False)
    if duplicated.any():
        dup_imgs = df_gt.loc[duplicated, "img"].unique()
        raise ValueError(
            f"❌ Error: 'img' must be unique in df_gt, but found duplicates for: {dup_imgs.tolist()}"
        )
    if len(df_gt) == 0:
        raise ValueError("Ground truth is empty")
    

    # --- Match per image ---
    df_pred = df_pred.copy()

    for img, df_pred_img in df_pred.groupby("img", sort=False):
        row_gt_img = df_gt[df_gt["img"] == img]
        tp_img = match_image_predictions(df_pred_img, row_gt_img, iou_thres)
        df_pred.loc[df_pred_img.index, "tp"] = tp_img
    
    df_pred["tp"] = df_pred["tp"].astype(int)

    return df_pred

def compute_map(df_pred: pd.DataFrame, df_gt: pd.DataFrame, iou_thres: float = 0.5):
    """
    Compute mean Average Precision at IoU=0.5 (mAP@50)
    for a single-class detector (e.g. heart valve detection).

    Parameters
    ----------
    df_pred : pd.DataFrame
        Predicted boxes, columns ['img','cls','conf','x1','y1','x2','y2'].
    df_gt : pd.DataFrame
        Ground-truth boxes, columns ['img','cls','x1','y1','x2','y2'].
    iou_thres : float
        IoU threshold for a match (default=0.5).

    Returns
    -------
    float
        mAP@50 value.
    """
    # --- Sort predictions by confidence ---
    df_pred = df_pred.sort_values("conf", ascending=False).reset_index(drop=True)
    # --- Compute TP flags ---
    df_pred = compute_tp_flags(df_pred, df_gt, iou_thres=iou_thres)
    n_gt = len(df_gt)

    tp = df_pred["tp"]
    fp = 1 - tp

    # --- Cumulative counts ---
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # --- Precision & Recall ---
    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / (n_gt + 1e-8)

    # --- Monotonic precision (enforce non-increasing) ---
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # --- Integrate precision–recall curve (trapezoid rule) ---
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    ap = np.trapezoid(mpre, mrec)

    print(f"📊 mAP@{int(round(iou_thres*100))} = {ap:.4f}")
    return ap