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

def trim_inconfident_head(df_pred: pd.DataFrame, conf_thres: float = 0.5) -> pd.DataFrame:
    """
    Iteratively trim low-confidence boxes from the head (lowest z)
    for each patient until no boxes with confidence < conf_thres remain.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must include columns ['pid', 'z', 'conf'].
    conf_thres : float, default=0.5
        Confidence threshold. Boxes below this are trimmed from the head.

    Returns
    -------
    pd.DataFrame
        Trimmed DataFrame across all patients.
    """
    out: list[pd.DataFrame] = []

    for pid, df_pid in df_pred.groupby("pid"):
        df_pid = df_pid.sort_values("z").reset_index(drop=True)

        z_values = sorted(df_pid["z"].unique())
        collected: list[pd.DataFrame] = []
        trimmed = 0
        stop_z: int | None = None

        # head iteration
        for z in z_values:
            df_z = df_pid[df_pid["z"] == z]
            weak_mask = df_z["conf"] < conf_thres
            n_weak = weak_mask.sum()
            n_total = len(df_z)

            if n_weak == n_total:
                # all weak: drop entire slice
                trimmed += n_total
                continue

            if n_weak == 0:
                # all strong: collect entire slice and stop trimming
                collected.append(df_z)
                stop_z = z
                break

            # mixed: keep strong, continue trimming
            strong = df_z[~weak_mask]
            trimmed += n_weak
            collected.append(strong)

        if stop_z is None:
            # never encountered a fully-strong slice → only use collected
            if collected:
                print(f"pid {pid}: trimmed {trimmed} boxes")
                out.append(pd.concat(collected, ignore_index=True))
            else:
                print(f"pid {pid}: trimmed {trimmed} boxes (all dropped)")
            continue

        # append remaining slices untouched
        tail = df_pid[df_pid["z"] > stop_z]
        if not tail.empty:
            collected.append(tail)

        if trimmed > 0:
            print(f"pid {pid}: trimmed {trimmed} boxes from head")
        out.append(pd.concat(collected, ignore_index=True))

    if not out:
        return pd.DataFrame(columns=df_pred.columns)

    return pd.concat(out, ignore_index=True)

def trim_inconfident_tail(
    df_pred: pd.DataFrame,
    conf_thres: float = 0.25,
) -> pd.DataFrame:
    """
    Tail-trim by z until reaching a fully-strong slice.
    For each z (descending):
      - all weak   -> drop entire z
      - mixed      -> keep only strong boxes
      - all strong -> stop trimming; keep remaining slices above untouched
    """

    out: list[pd.DataFrame] = []

    for pid, df_pid in df_pred.groupby("pid"):
        df_pid = df_pid.sort_values("z").reset_index(drop=True)

        z_values = sorted(df_pid["z"].unique(), reverse=True)
        collected: list[pd.DataFrame] = []
        trimmed = 0
        stop_z: int | None = None

        # iterate from tail downward
        for z in z_values:
            df_z = df_pid[df_pid["z"] == z]
            weak_mask = df_z["conf"] < conf_thres
            n_weak = weak_mask.sum()
            n_total = len(df_z)

            if n_weak == n_total:
                # all weak: drop whole slice
                trimmed += n_total
                continue

            if n_weak == 0:
                # all strong: collect and stop trimming
                collected.append(df_z)
                stop_z = z
                break

            # mixed: keep strong, keep trimming
            strong = df_z[~weak_mask]
            trimmed += n_weak
            collected.append(strong)

        if stop_z is None:
            # never found any fully-strong slice (tail all weak/mixed)
            if collected:
                df_keep = pd.concat(collected, ignore_index=True)
                print(f"pid {pid}: trimmed {trimmed} boxes")
                out.append(df_keep)
            else:
                print(f"pid {pid}: trimmed {trimmed} boxes (all dropped)")
            continue

        # append the remaining head slices untouched
        head = df_pid[df_pid["z"] < stop_z]
        if not head.empty:
            collected.append(head)

        if trimmed > 0:
            print(f"pid {pid}: trimmed {trimmed} boxes")
        result = pd.concat(collected, ignore_index=True)
        out.append(result)

    if not out:
        return pd.DataFrame(columns=df_pred.columns)

    return pd.concat(out, ignore_index=True)