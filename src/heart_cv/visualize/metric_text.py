import pandas as pd

from ..metric import compute_iou

_FLOAT_COLS = ['conf',"iou",'area']
_OUTPUT_COLS = _FLOAT_COLS

def compute_metric(df_pred: pd.DataFrame, df_gt: pd.DataFrame):
    out_cols = _OUTPUT_COLS + ["z"]
    df = compute_iou(df_pred, df_gt).sort_values('conf', ascending=False)
    df['area'] = (df.x2 - df.x1) * (df.y2 - df.y1)
    df.index = (df['iou'] > 0.5).map({True:"✅", False:"❌"})
    df.index.name = None
    df[_FLOAT_COLS] = df[_FLOAT_COLS].round(2)
    return df[out_cols]

def display_text(df_metric: pd.DataFrame, z_val: int, df_gt: pd.DataFrame):
    print(df_metric.loc[df_metric['z'] == z_val, _OUTPUT_COLS])