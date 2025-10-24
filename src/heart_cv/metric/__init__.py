import pandas as pd

from .map import compute_map
from .iou import compute_iou
from .recall import compute_recall

def report_metrics(df_pred: pd.DataFrame, df_gt: pd.DataFrame):
    print(f"map@50: {compute_map(df_pred, df_gt)}")
    print(f"recall: {compute_recall(df_pred, df_gt)}")