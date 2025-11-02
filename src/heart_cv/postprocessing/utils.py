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

def pruning_small_side_tubes(
    df_tube: pd.DataFrame,
    keeping_tube_length: int = 10,
    conf_thres: float = 0.5,
) -> pd.DataFrame:
    """
    Prune short or side tubes in a tube DataFrame while preserving strong lone boxes.

    Parameters
    ----------
    df_tube : pd.DataFrame
        Must contain ['img','cls','conf','x1','y1','x2','y2','pid','z','tube_id'].
    keeping_tube_length : int, default=10
        Minimum number of slices (tube length) to keep a tube intact.
    conf_thres : float, default=0.5
        Confidence threshold for retaining lone boxes from pruned tubes.

    Returns
    -------
    pd.DataFrame
        Pruned DataFrame with main and retained boxes.
    """

    if df_tube.empty:
        return df_tube.copy()

    # --- compute per-tube stats --- #
    tube_stats = (
        df_tube.groupby("tube_id")
        .agg(
            n_z=("z", "nunique"),
            z_min=("z", "min"),
            z_max=("z", "max"),
        )
        .reset_index()
    )
    tube_stats["z_span"] = tube_stats["z_max"] - tube_stats["z_min"] + 1

    # --- find main tube --- #
    main_tube_id = tube_stats.loc[tube_stats["n_z"].idxmax(), "tube_id"]
    z_min_main = tube_stats.loc[tube_stats["tube_id"] == main_tube_id, "z_min"].item()
    z_max_main = tube_stats.loc[tube_stats["tube_id"] == main_tube_id, "z_max"].item()

    keep_rows = []

    for tube_id, group in df_tube.groupby("tube_id"):
        n_z = group["z"].nunique()
        z_min = group["z"].min()
        z_max = group["z"].max()

        if tube_id == main_tube_id:
            # always keep main tube
            keep_rows.append(group)
        elif n_z > keeping_tube_length:
            # long tubes always kept
            keep_rows.append(group)
        else:
            # check overlap relative to main tube
            fully_inside = (z_min >= z_min_main) and (z_max <= z_max_main)
            if fully_inside:
                # drop fully side tube, but keep strong boxes
                strong_boxes = group[group["conf"] > conf_thres]
                if not strong_boxes.empty:
                    keep_rows.append(strong_boxes)
            else:
                # extends beyond -> keep
                keep_rows.append(group)


    if not keep_rows:
        return pd.DataFrame(columns=df_tube.columns)

    df_pruned = pd.concat(keep_rows, ignore_index=True)
    return df_pruned.sort_values(["tube_id", "z"]).reset_index(drop=True)

def pruning_recursive_side_tubes(
    df_tube: pd.DataFrame,
    keeping_tube_length: int = 10,
    conf_thres: float = 0.5,
) -> pd.DataFrame:
    """
    Prune side tubes using strict containment on z-spans (no recursion needed).

    Strictly-inside uses *open* bounds:
        A inside B  <=>  z_min_B < z_min_A  AND  z_max_A < z_max_B
    so boundary-touching counts as "exceeding" and is kept.

    Always keep:
      - main tube (max n_z)
      - tubes with length > keeping_tube_length
      - any tube NOT strictly inside any already-kept tube

    From pruned tubes, retain only boxes with conf > conf_thres.

    Parameters
    ----------
    df_tube : DataFrame with columns:
        ['img','cls','conf','x1','y1','x2','y2','pid','z','tube_id']
    keeping_tube_length : int
        Threshold to mark tubes as protected (always kept).
    conf_thres : float
        Confidence threshold to salvage lone boxes from pruned tubes.

    Returns
    -------
    DataFrame: pruned df_tube with the same columns.
    """
    if df_tube.empty:
        return df_tube.copy()

    # Per-tube stats
    stats = (
        df_tube.groupby("tube_id")
        .agg(n_z=("z", "nunique"), z_min=("z", "min"), z_max=("z", "max"))
        .reset_index()
    )
    stats["z_span"] = stats["z_max"] - stats["z_min"] + 1

    # Main tube = max length
    main_tube_id = stats.loc[stats["n_z"].idxmax(), "tube_id"]

    # Protected tubes (always keep)
    protected = set(stats.loc[stats["n_z"] >= keeping_tube_length, "tube_id"].tolist())
    protected.add(main_tube_id)

    # Sort by span (wide first), then by length, so big parents are considered earlier
    order = stats.sort_values(["z_span", "n_z"], ascending=False).reset_index(drop=True)

    kept = set()
    kept_spans: list[tuple[int, int, int]] = []  # (z_min, z_max, tube_id)

    def strictly_inside(a_min, a_max, b_min, b_max) -> bool:
        # open bounds => touching boundary is NOT inside
        return (b_min < a_min) and (a_max < b_max)

    for _, row in order.iterrows():
        tid = int(row.tube_id)
        a_min, a_max = int(row.z_min), int(row.z_max)

        if tid in protected:
            kept.add(tid)
            kept_spans.append((a_min, a_max, tid))
            continue

        # If strictly inside ANY already-kept span => prunable
        inside_any = any(strictly_inside(a_min, a_max, b_min, b_max) for b_min, b_max, _ in kept_spans)

        if not inside_any:
            # Not strictly inside any kept span → keep it
            kept.add(tid)
            kept_spans.append((a_min, a_max, tid))
        # else: prunable

    # Build final dataframe: keep full kept tubes; salvage high-conf boxes from pruned tubes
    keep_rows = []
    for tid, grp in df_tube.groupby("tube_id"):
        if tid in kept:
            keep_rows.append(grp)
        else:
            salvage = grp[grp["conf"] >= conf_thres]
            if not salvage.empty:
                keep_rows.append(salvage)

    if not keep_rows:
        return pd.DataFrame(columns=df_tube.columns)

    out = pd.concat(keep_rows, ignore_index=True)
    return out.sort_values(["tube_id", "z"]).reset_index(drop=True)