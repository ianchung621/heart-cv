import numpy as np
import pandas as pd

from .utils import add_pid_z_paths

def expand_xz_boxes_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Expand [x1,z1,x2,z2,y,conf,pid] → [x1,x2,y,z,conf,pid]."""
    z1 = df["z1"].astype(int).to_numpy()
    z2 = df["z2"].astype(int).to_numpy()
    counts = z2 - z1 + 1

    repeated_idx = np.repeat(np.arange(len(df)), counts)
    x1   = df["x1"].to_numpy()[repeated_idx]
    x2   = df["x2"].to_numpy()[repeated_idx]
    y    = df["y"].to_numpy()[repeated_idx]
    conf = df["conf"].to_numpy()[repeated_idx]
    pid  = df["pid"].to_numpy(dtype=int)[repeated_idx]
    z    = np.concatenate([np.arange(a, b + 1) for a, b in zip(z1, z2)])

    return pd.DataFrame({
        "pid": pid, "z": z, "y": y,
        "x1": x1, "x2": x2, "conf": conf
    })


def expand_yz_boxes_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Expand [y1,z1,y2,z2,x,conf,pid] → [y1,y2,x,z,conf,pid]."""
    z1 = df["z1"].astype(int).to_numpy()
    z2 = df["z2"].astype(int).to_numpy()
    counts = z2 - z1 + 1

    repeated_idx = np.repeat(np.arange(len(df)), counts)
    y1   = df["y1"].to_numpy()[repeated_idx]
    y2   = df["y2"].to_numpy()[repeated_idx]
    x    = df["x"].to_numpy()[repeated_idx]
    conf = df["conf"].to_numpy()[repeated_idx]
    pid  = df["pid"].to_numpy(dtype=int)[repeated_idx]
    z    = np.concatenate([np.arange(a, b + 1) for a, b in zip(z1, z2)])

    return pd.DataFrame({
        "pid": pid, "z": z, "x": x,
        "y1": y1, "y2": y2, "conf": conf
    })

def get_z_projections(df_pred_x: pd.DataFrame, df_pred_y: pd.DataFrame, conf_thres: float = 0.8):
    """
    Convert x- and y-view predictions into per-z stripes for 3D reconstruction.

    Parameters
    ----------
    df_pred_x : pd.DataFrame
        Columns ['pid','z','x1','y1','x2','y2','conf'] for x-view (xz boxes)
    df_pred_y : pd.DataFrame
        Columns ['pid','z','x1','y1','x2','y2','conf'] for y-view (yz boxes)
    conf_thres : float
        Minimum confidence to keep

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_x, df_y) expanded to stripes with columns
        ['pid','z','x1','x2','y','conf','img'], ['pid','z','y1','y2','x','conf','img']
    """
    df_x = add_pid_z_paths(df_pred_x)
    df_y = add_pid_z_paths(df_pred_y)

    # --- x projection ---
    df_x = df_x.rename(columns={
        "x1": "x1", "x2": "x2",
        "y1": "z1", "y2": "z2",
        "z": "y"
    })
    df_x = df_x[df_x["conf"] > conf_thres]
    df_x = expand_xz_boxes_fast(df_x)
    df_x["z"] += 1
    df_x["y"] -= 1
    df_x["img"] = (
        "patient" +
        df_x["pid"].astype(str).str.zfill(4) + "_" +
        df_x["z"].astype(str).str.zfill(4)
    )

    # --- y projection ---
    df_y = df_y.rename(columns={
        "x1": "y1", "x2": "y2",
        "y1": "z1", "y2": "z2",
        "z": "x"
    })
    df_y = df_y[df_y["conf"] > conf_thres]
    df_y = expand_yz_boxes_fast(df_y)
    df_y["z"] += 1
    df_y["x"] -= 1
    df_y["img"] = (
        "patient" +
        df_y["pid"].astype(str).str.zfill(4) + "_" +
        df_y["z"].astype(str).str.zfill(4)
    )

    return df_x, df_y

def bbox_to_orig_coords(df_pred_x: pd.DataFrame, df_pred_y: pd.DataFrame, conf_thres: float = 0.8):
    """
    Convert x- and y-view predictions into per-z stripes for 3D reconstruction.

    Parameters
    ----------
    df_pred_x : pd.DataFrame
        Columns ['pid','z','x1','y1','x2','y2','conf'] for x-view (xz boxes)
    df_pred_y : pd.DataFrame
        Columns ['pid','z','x1','y1','x2','y2','conf'] for y-view (yz boxes)
    conf_thres : float
        Minimum confidence to keep

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_x, df_y) expanded to stripes with columns
        ['pid','z','x1','x2','y','conf','img'], ['pid','z','y1','y2','x','conf','img']
    """
    df_x = add_pid_z_paths(df_pred_x)
    df_y = add_pid_z_paths(df_pred_y)

    # --- x projection ---
    df_x = df_x.rename(columns={
        "x1": "x1", "x2": "x2",
        "y1": "z1", "y2": "z2",
        "z": "y"
    })
    df_x = df_x[df_x["conf"] > conf_thres]
    df_x["z1"] += 1
    df_x["z2"] += 1
    df_x["y"] -= 1

    # --- y projection ---
    df_y = df_y.rename(columns={
        "x1": "y1", "x2": "y2",
        "y1": "z1", "y2": "z2",
        "z": "x"
    })
    df_y = df_y[df_y["conf"] > conf_thres]
    df_y["z1"] += 1
    df_y["z2"] += 1
    df_y["x"] -= 1

    return df_x, df_y