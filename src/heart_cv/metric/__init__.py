import pandas as pd

from .map import compute_map
from .iou import compute_iou
from .recall import compute_recall
from ..postprocessing import add_pid_z_paths

def report_metrics(df_pred: pd.DataFrame, df_gt: pd.DataFrame, exclude_pid: list[int]|int = None):
    """
    Compute and print evaluation metrics (mAP@50, recall),
    optionally excluding specific patients.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Prediction DataFrame with at least 'pid' column.
    df_gt : pd.DataFrame
        Ground-truth DataFrame with at least 'pid' column.
    exclude_pid : list[int], optional
        List of patient IDs to exclude before computing metrics.
    """
    add_pid_z_paths(df_pred)
    add_pid_z_paths(df_gt)
    if isinstance(exclude_pid, int):
        exclude_pid = [exclude_pid]
    if isinstance(exclude_pid, list):
        df_pred = df_pred[~df_pred["pid"].isin(exclude_pid)].copy()
        df_gt = df_gt[~df_gt["pid"].isin(exclude_pid)].copy()

    map50 = compute_map(df_pred, df_gt)
    recall = compute_recall(df_pred, df_gt)

    print(f"map@50: {map50:.4f}     recall: {recall:.4f}")