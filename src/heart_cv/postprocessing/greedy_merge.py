import numpy as np
import pandas as pd
from warnings import warn
from numba import njit

import numpy as np
from numba import njit

@njit
def box_area(b):
    w = max(0.0, b[2] - b[0])
    h = max(0.0, b[3] - b[1])
    return w * h

@njit
def intersection_area(A, B):
    x1 = max(A[0], B[0])
    y1 = max(A[1], B[1])
    x2 = min(A[2], B[2])
    y2 = min(A[3], B[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

@njit
def iou(A, B):
    inter = intersection_area(A, B)
    if inter == 0.0:
        return 0.0
    areaA = box_area(A)
    areaB = box_area(B)
    return inter / (areaA + areaB - inter)

@njit
def merge_linear(A, B, t):
    """Coordinate-wise interpolation: M = (1-t)B + tA"""
    M = np.empty(4, dtype=np.float64)
    for i in range(4):
        M[i] = (1 - t) * B[i] + t * A[i]
    return M

@njit
def maximin_merge(A, B, eps=1e-5):
    """
    Jit finction. Find M = (1-t)B + tA maximizing min(IoU(A,M), IoU(B,M)).
    Returns (M, min_iou).

    Parameters
    ----------
    A, B : np.ndarray[4]
        Boxes [x1,y1,x2,y2].
    eps : float
        Search tolerance.

    Returns
    -------
    M : np.ndarray[4]
        Merged box.
    iou_min : float
        Maximum of min(IoU(A,M), IoU(B,M)).
    """
    t_lo, t_hi = 0.0, 1.0
    f_lo = iou(A, merge_linear(A, B, t_lo)) - iou(B, merge_linear(A, B, t_lo))
    f_hi = iou(A, merge_linear(A, B, t_hi)) - iou(B, merge_linear(A, B, t_hi))

    # If both IoUs trend same way, pick midpoint
    if f_lo * f_hi > 0:
        t_star = 0.5
    else:
        for _ in range(64):
            t_mid = 0.5 * (t_lo + t_hi)
            M_mid = merge_linear(A, B, t_mid)
            f_mid = iou(A, M_mid) - iou(B, M_mid)
            if abs(f_mid) < eps:
                t_star = t_mid
                break
            if f_lo * f_mid < 0:
                t_hi = t_mid
                f_hi = f_mid
            else:
                t_lo = t_mid
                f_lo = f_mid
            t_star = 0.5 * (t_lo + t_hi)

    M_star = merge_linear(A, B, t_star)
    iouA = iou(A, M_star)
    iouB = iou(B, M_star)
    return M_star, min(iouA, iouB)

def apply_greedy_merge(
    df_pred: pd.DataFrame,
    min_area_thres = 0.25,
    min_iou_thres = 0.6,
    min_Lbox_area = 1000,
    eps = 1e-5,
) -> pd.DataFrame:
    """
    Greedy merge for images with exactly 2 boxes.

    For each image:
      - If there are exactly 2 boxes (A,B)
      - Find which one is larger by area
      - If small box area > thres * (small_box_area inside large box)
        (i.e. small box is mostly contained within large box)
      - Replace both with a merged box M(A,B) via merge_box_maximin()

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must include columns ['img', 'x1','y1','x2','y2'].
    thres : float, default=1
        Containment ratio threshold.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with merged boxes.
    """

    df_out = []
    merged_img = []

    for img, g in df_pred.groupby("img", group_keys=False):
        if len(g) != 2:
            df_out.append(g)
            continue

        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        areas = np.array([(b[2]-b[0]) * (b[3]-b[1]) for b in boxes])
        i_large = int(np.argmax(areas))
        i_small = 1 - i_large
        if areas[i_small] < min_area_thres * areas[i_large]:
            df_out.append(g)
            continue

        if areas[i_large] < min_Lbox_area:
            df_out.append(g)
            continue

        A = boxes[i_large]
        B = boxes[i_small]

        try:
            M, iou_min = maximin_merge(A, B, eps)
            if iou_min < min_iou_thres:
                df_out.append(g)
                continue
            
        except ValueError as e:
            warn(f"[{img}] merge failed, Skipping: {e}")
            df_out.append(g)
            continue

        # build merged row: take the larger box’s metadata
        merged_row = g.iloc[i_large].copy()
        merged_row[["x1", "y1", "x2", "y2"]] = M
        merged_row["conf"] = g["conf"].max()  # optional
        merged_img.append(merged_row["img"])
        df_out.append(pd.DataFrame([merged_row]))

    df_merged = pd.concat(df_out, ignore_index=True)
    print(f"merged {len(df_pred) - len(df_merged)} boxes: {merged_img}")
    return df_merged