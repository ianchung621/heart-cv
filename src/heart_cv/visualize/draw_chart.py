import pandas as pd
import matplotlib.pyplot as plt

def compute_chart_df(df_metric: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-slice metrics into a chart-friendly DataFrame.

    For each z:
      - Select the prediction with the highest confidence (max_conf)
      - Take its corresponding IoU
      - Compute ground-truth valve area from df_gt

    Parameters
    ----------
    df_metric : pd.DataFrame
        Must include columns ['z', 'conf', 'iou','area'].
    df_gt : pd.DataFrame
        Must include columns ['z', 'x1','y1','x2','y2'] (one GT box per z).

    Returns
    -------
    pd.DataFrame
        Columns: ['z', 'max_conf', 'iou', 'valve_area']
    """
    if df_metric.empty or df_gt.empty:
        return pd.DataFrame(columns=["z", "max_conf", "iou", "valve_area"])

    # collect attrs
    map50 = getattr(df_metric, "map50", None)
    recall = getattr(df_metric, "recall", None)
    # --- (1) max confidence + corresponding IoU ---
    df_metric = df_metric.reset_index()
    df_metric["weighted_area"] = df_metric["conf"] * df_metric["area"]

    df_area = (
        df_metric.groupby("z", as_index=False)
        .agg(weighted_valve_area=("weighted_area", "sum"),
             sum_conf=("conf", "sum"))
    )
    df_area["weighted_valve_area"] = (
        df_area["weighted_valve_area"] / df_area["sum_conf"]
    )

    idx = df_metric.groupby("z")["conf"].idxmax()
    df_top = df_metric.loc[idx, ["z", "conf", "iou"]].rename(columns={"conf": "max_conf"})

    # --- (2) valve area from GT boxes ---
    df_gt = df_gt.copy()
    pid = int(df_gt["pid"].iloc[0])
    df_gt["valve_area"] = (df_gt["x2"] - df_gt["x1"]) * (df_gt["y2"] - df_gt["y1"])
    df_gt = df_gt[["z", "valve_area"]]

    # --- (3) merge and return ---
    df_chart = pd.merge(df_top, df_area[["z", "weighted_valve_area"]], on="z", how="left")
    df_chart = pd.merge(df_chart, df_gt, on="z", how="left").sort_values("z").reset_index(drop=True)
    df_chart.pid = pid
    df_chart.map50 = map50
    df_chart.recall = recall
    return df_chart

def draw_chart(df_chart: pd.DataFrame, z_val: int, df_gt: pd.DataFrame):
    """
    Plot per-slice confidence, IoU, and valve area with dual y-axes.
    Highlights the current z_val and draws GT valve region boundaries.

    Parameters
    ----------
    df_chart : pd.DataFrame
        Columns: ['z', 'max_conf', 'iou', 'valve_area']
    z_val : int
        Current slice index to highlight.
    df_gt : pd.DataFrame
        Ground truth boxes (with 'z' column) to determine valve region.
    """
    if df_chart.empty:
        print("No chart data available.")
        return

    fig, ax1 = plt.subplots(figsize=(6, 3))

    # left y-axis: confidence + IoU
    ax1.plot(df_chart["z"], df_chart["max_conf"], label="Max Conf", color="tab:green", lw=1.5)
    ax1.plot(df_chart["z"], df_chart["iou"], label="IoU", color="tab:red", lw=1.5)

    ax1.set_xlabel("Slice (z)")
    ax1.set_ylabel("Confidence / IoU", color="tab:gray")
    ax1.set_ylim(0, 1)

    # right y-axis: valve area
    ax2 = ax1.twinx()
    ax2.plot(df_chart["z"], df_chart["valve_area"], label="GT Area", color="tab:blue", lw=1.5)
    ax2.plot(df_chart["z"], df_chart["weighted_valve_area"], label="Pred Area", color="blue", lw=1.5)
    ax2.set_ylabel("Valve Area (px²)", color="tab:blue")

    # highlight current z
    if z_val in df_chart["z"].values:
        row = df_chart.loc[df_chart["z"] == z_val].iloc[0]
        ax1.scatter([row["z"]], [row["max_conf"]], color="tab:green", s=50, zorder=5)
        ax1.scatter([row["z"]], [row["iou"]], color="tab:red", s=50, zorder=5)
        ax2.scatter([row["z"]], [row["valve_area"]], color="tab:blue", s=50, zorder=5)
        ax2.text(row["z"], row["valve_area"], f"{row["valve_area"]:.0f}", color="tab:blue")
        ax2.scatter([row["z"]], [row["weighted_valve_area"]], color="blue", s=50, zorder=5)

    # draw vertical lines for valve boundaries
    z_min, z_max = df_gt["z"].min(), df_gt["z"].max()
    ax1.axvline(z_min, color="gray", ls="--", lw=0.8)
    ax1.axvline(z_max, color="gray", ls="--", lw=0.8)

    # legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left',
    bbox_to_anchor=(1.05, 0.5))
    
    # title
    pid = getattr(df_chart, "pid", None)
    map50 = getattr(df_chart, "map50", None)
    recall = getattr(df_chart, "recall", None)
    ax1.set_title(f"Patient {pid}\n map@50 = {map50:.4f} recall = {recall:.4f}")

    fig.tight_layout()
    plt.show()