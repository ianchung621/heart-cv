import pandas as pd
from .iou import compute_iou

def compute_recall(df_pred: pd.DataFrame, df_gt: pd.DataFrame):
    df_pred = compute_iou(df_pred, df_gt)
    return (df_pred.groupby('img')['iou'].max() > 0.5).sum()/len(df_gt)