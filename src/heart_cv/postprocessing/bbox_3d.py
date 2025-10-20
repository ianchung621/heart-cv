import pandas as pd
import numpy as np
from .utils import add_pid_z_paths

def get_bounding_box(
    df_pred_x: pd.DataFrame,
    df_pred_y: pd.DataFrame,
    df_pred_z: pd.DataFrame,
    conf_thres: float = 0.8,
    paddings: tuple[float, float, float] = (0.05, 0.05, 0.),
) -> pd.DataFrame:
    """
    Combine 3-view predictions (x, y, z) into unified 3D bounding boxes.

    Each df must contain: ['pid', 'z', 'x1', 'y1', 'x2', 'y2'] (pixel coordinates).
    Padding values are *fractions* of each patient’s box size along (x, y, z).

    Returns
    -------
    bounding_box_df : pd.DataFrame
        Columns: ['pid','x_min','x_max','y_min','y_max','z_min','z_max']
    """
    def ensure_pid_z(df: pd.DataFrame):
        if not {"pid", "z"}.issubset(df.columns):
            df = add_pid_z_paths(df)
        return df
    def get_boundary(df: pd.DataFrame):
        return (
            df[df['conf'] > conf_thres].groupby("pid")
            .agg({
                "z": ["min", "max"],
                "x1": "min",
                "x2": "max",
                "y1": "min",
                "y2": "max"
            })
        )
    df_pred_x, df_pred_y, df_pred_z = map(ensure_pid_z, (df_pred_x, df_pred_y, df_pred_z))

    pred_x_range = get_boundary(df_pred_x)
    pred_y_range = get_boundary(df_pred_y)
    pred_z_range = get_boundary(df_pred_z)
    pred_x_range.columns = ["y_min","y_max","x_min","x_max","z_min","z_max"]
    pred_y_range.columns = ["x_min","x_max","y_min","y_max","z_min","z_max"]
    pred_z_range.columns = ["z_min","z_max","x_min","x_max","y_min","y_max"]
    
    # --- merge across perspectives ---
    df = pd.DataFrame(index=pred_z_range.index)
    df["pid"] = df.index

    # xy from z-view
    df["x_min"] = pred_z_range["x_min"]
    df["x_max"] = pred_z_range["x_max"]
    df["y_min"] = pred_z_range["y_min"]
    df["y_max"] = pred_z_range["y_max"]

    # z from x/y views
    df["z_min"] = pd.concat([pred_x_range["z_min"], pred_y_range["z_min"]], axis=1).min(axis=1)
    df["z_max"] = pd.concat([pred_x_range["z_max"], pred_y_range["z_max"]], axis=1).max(axis=1)

    # --- compute per-axis box sizes ---
    df["x_len"] = df["x_max"] - df["x_min"]
    df["y_len"] = df["y_max"] - df["y_min"]
    df["z_len"] = df["z_max"] - df["z_min"]

    # --- apply symmetric fractional padding ---
    pad_x, pad_y, pad_z = paddings
    df["x_min"] -= df["x_len"] * pad_x / 2
    df["x_max"] += df["x_len"] * pad_x / 2
    df["y_min"] -= df["y_len"] * pad_y / 2
    df["y_max"] += df["y_len"] * pad_y / 2
    df["z_min"] -= df["z_len"] * pad_z / 2
    df["z_max"] += df["z_len"] * pad_z / 2

    return df[["pid","x_min","x_max","y_min","y_max","z_min","z_max"]].reset_index(drop=True)

def trim_by_bbox3d(df_pred: pd.DataFrame, bbox_df: pd.DataFrame, inbox_thres: float = 0.5) -> pd.DataFrame:
    """
    Trim predictions that fall mostly outside the 3D bounding box (per patient).

    Steps:
      1. Ensure `pid`, `z` exist in df_pred (infer if missing).
      2. Drop boxes with z outside [z_min, z_max].
      3. Compute intersection area ratio of each [x1, y1, x2, y2] with bbox [x_min, y_min, x_max, y_max].
      4. Keep boxes whose ratio >= inbox_thres.

    Returns
    -------
    pd.DataFrame : filtered copy of df_pred
    """
    # --- ensure pid, z exist ---
    if not {"pid", "z"}.issubset(df_pred.columns):
        df_pred = add_pid_z_paths(df_pred)

    # --- merge bbox info ---
    merged = df_pred.merge(bbox_df, on="pid", how="left")

    # --- remove z outside [z_min, z_max] ---
    z_mask = (merged["z"] >= merged["z_min"]) & (merged["z"] <= merged["z_max"])
    merged = merged[z_mask].copy()

    # --- compute intersection area ratio ---
    x_left   = np.maximum(merged["x1"], merged["x_min"])
    y_top    = np.maximum(merged["y1"], merged["y_min"])
    x_right  = np.minimum(merged["x2"], merged["x_max"])
    y_bottom = np.minimum(merged["y2"], merged["y_max"])

    inter_w = np.maximum(0, x_right - x_left)
    inter_h = np.maximum(0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    row_area = (merged["x2"] - merged["x1"]) * (merged["y2"] - merged["y1"])
    ratio = np.divide(inter_area, row_area, out=np.zeros_like(inter_area), where=row_area > 0)

    # --- apply threshold ---
    keep_mask = ratio >= inbox_thres
    trimmed = merged[keep_mask].copy()

    return trimmed.reset_index(drop=True)