from typing import Literal
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

def apply_wbf(
    df_pred_list:list[pd.DataFrame],
    image_size=(512, 512),
    iou_thr=0.5,
    skip_box_thr=0.001,
    weights=None,
    conf_type: Literal['avg','max','box_and_model_avg','absent_model_aware_avg']='avg'):
    """
    Apply Weighted Box Fusion (WBF) on multiple YOLO prediction DataFrames.

    Parameters
    ----------
    df_pred_list : list[pd.DataFrame]
        List of prediction DataFrames from different YOLO models.
        Each must have columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    image_size : tuple[int, int]
        (W, H) image size for normalization (default 512x512).
    iou_thr : float
        IoU threshold for fusion.
    skip_box_thr : float
        Boxes below this confidence are skipped.
    weights : list[float] or None
        Relative model weights for fusion.

    Returns
    -------
    pd.DataFrame
        New ensembled df_pred with columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    """
    W, H = image_size
    out_rows = []

    imgs = sorted(set().union(*[df['img'].unique() for df in df_pred_list]))
    for img in imgs:
        boxes_list, scores_list, labels_list = [], [], []
        for df in df_pred_list:
            d: pd.DataFrame = df[df['img'] == img]
            if d.empty:
                continue
            boxes = boxes = d[['x1','y1','x2','y2']].values
            # normalize to 0-1 range
            boxes = boxes / [W, H, W, H]
            boxes = boxes.clip(0, 1)

            boxes_list.append(boxes.tolist())
            scores_list.append(d['conf'].tolist())
            labels_list.append(d['cls'].tolist())

        if not boxes_list:
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type
        )

        # rescale back
        boxes = boxes * [W, H, W, H]
        for b, s, l in zip(boxes, scores, labels):
            out_rows.append([img, int(l), s, *b])

    df_out = pd.DataFrame(out_rows, columns=['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'])
    return df_out