from pathlib import Path
import pandas as pd

def add_pid_z_paths(df: pd.DataFrame, col: str = "img") -> pd.DataFrame:
    """
    Extract pid, z from image name (e.g. 'patient0001_0174'),
    infer dataset split (training/testing) from pid range,
    and build full image/label paths.

    Rules:
        pid 1–50   → training set (with labels)
        pid 51–100 → testing set (no labels)
    """
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