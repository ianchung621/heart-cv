from pathlib import Path
import pandas as pd

def add_pid_z_paths(df: pd.DataFrame, col: str = "img", overwrite: bool = False) -> pd.DataFrame:
    """
    Extract pid, z from image name (e.g. 'patient0001_0174'),
    infer dataset split (training/testing) from pid range,
    and build full image/label paths.

    Rules:
        pid 1–50   → training set (with labels)
        pid 51–100 → testing set (no labels)
    """
    if not overwrite and {"pid","z","img_path"}.issubset(df.columns):
        return df

    extracted = df[col].astype(str).str.extract(r"patient(\d+)_(\d+)")
    df["pid"] = extracted[0].astype(int)
    df["z"] = extracted[1].astype(int)

    # Ensure all belong to same split
    pids = df["pid"]
    if (1 <= pids).all() and (pids <= 50).all():
        split = "training"
    elif (51 <= pids).all() and (pids <= 100).all():
        split = "testing"
    else:
        raise ValueError("Mixed or invalid pid range (must be all 1–50 or all 51–100).")

    # Construct image paths
    df["img_path"] = df.apply(
        lambda r: Path(f"dataset/{split}_image") /
                  f"patient{r.pid:04d}" /
                  f"patient{r.pid:04d}_{r.z:04d}.png",
        axis=1
    )

    # Construct label paths (training only)
    if split == "training":
        df["label_path"] = df.apply(
            lambda r: Path(f"dataset/training_label") /
                      f"patient{r.pid:04d}" /
                      f"patient{r.pid:04d}_{r.z:04d}.txt",
            axis=1
        )

    return df

def drop_low_conf(df: pd.DataFrame, ratio: float = 0.1) -> pd.DataFrame:
    """
    Drop rows whose conf < ratio * conf_max within each img.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['img', 'conf'].
    ratio : float, default 0.1
        Threshold ratio relative to the maximum conf per image.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df = df.reset_index(drop=True)
    conf_max = df.groupby("img")["conf"].transform("max")
    return df[df["conf"] >= ratio * conf_max].reset_index(drop=True)

def keep_topk_per_img(df_pred: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Keep only the top-K confidence boxes per image.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must contain ['img', 'conf'] plus other columns.
    top_k : int, default 3
        Number of top boxes to keep per image.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with top-K boxes per image.
    """
    df_sorted = df_pred.sort_values(["img", "conf"], ascending=[True, False])
    df_topk = df_sorted.groupby("img", group_keys=False).head(top_k)
    return df_topk.reset_index(drop=True)