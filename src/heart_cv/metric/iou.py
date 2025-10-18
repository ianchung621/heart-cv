import pandas as pd
import numpy as np

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

def compute_iou(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-prediction IoU with the corresponding ground-truth box (same 'img').

    Parameters
    ----------
    df_pred : pd.DataFrame
        Predicted boxes. Columns: ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    df_gt : pd.DataFrame
        Ground-truth boxes. Columns: ['img', 'cls', 'x1', 'y1', 'x2', 'y2'].
        Must have unique 'img' values (one GT box per image).

    Returns
    -------
    pd.DataFrame
        Copy of df_pred with an additional column:
            - 'iou' : float, IoU between each predicted box and the GT box of that image.
    """
    # --- Validation ---
    if "img" not in df_pred.columns or "img" not in df_gt.columns:
        raise ValueError("Both df_pred and df_gt must contain column 'img'.")
    if df_gt["img"].duplicated(keep=False).any():
        dup_imgs = df_gt.loc[df_gt["img"].duplicated(keep=False), "img"].unique()
        raise ValueError(
            f"❌ Error: 'img' must be unique in df_gt, but found duplicates for: {dup_imgs.tolist()}"
        )
    if len(df_gt) == 0:
        raise ValueError("Ground truth is empty.")
    
    # --- Copy to avoid mutating input ---
    df_pred = df_pred.copy()
    df_pred["iou"] = 0.0

    # --- Compute IoUs per image ---
    for img, df_pred_img in df_pred.groupby("img", sort=False):
        row_gt_img = df_gt[df_gt["img"] == img]
        if row_gt_img.empty:
            continue  # no GT box for this image → IoU stays 0

        boxes_pred = df_pred_img[["x1", "y1", "x2", "y2"]].to_numpy()
        boxes_gt = row_gt_img[["x1", "y1", "x2", "y2"]].to_numpy()

        # Compute IoU matrix (N_pred x N_gt)
        ious = box_iou(boxes_pred, boxes_gt)

        # Take max IoU per prediction (in case there are multiple GT boxes)
        max_ious = ious.max(axis=1) if ious.size > 0 else np.zeros(len(df_pred_img))
        df_pred.loc[df_pred_img.index, "iou"] = max_ious

    return df_pred