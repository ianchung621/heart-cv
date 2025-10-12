import numpy as np
import pandas as pd

def apply_nms(df_pred: pd.DataFrame, iou_thres: float = 0.5) -> pd.DataFrame:
    """
    Apply Non-Maximum Suppression (NMS) per image.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    iou_thres : float
        IoU threshold for suppression (default 0.5).

    Returns
    -------
    pd.DataFrame
        Filtered predictions after NMS.
    """
    def iou(box, boxes):
        # box: (x1, y1, x2, y2)
        # boxes: (N, 4)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter
        return inter / (union + 1e-9)

    kept = []
    for img, g in df_pred.groupby("img"):
        g = g.sort_values("conf", ascending=False).to_numpy()
        keep_idx = []
        while len(g):
            keep_idx.append(g[0])
            if len(g) == 1:
                break
            ious = iou(g[0, 3:7].astype(float), g[1:, 3:7].astype(float))
            g = g[1:][ious < iou_thres]
        kept.append(pd.DataFrame(keep_idx, columns=df_pred.columns))
    return pd.concat(kept, ignore_index=True)