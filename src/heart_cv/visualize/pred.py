from pathlib import Path
import pandas as pd

from ipywidgets import Output
from IPython.display import display

from .interactive import make_slice_navigator
from .draw_slice import draw_slice_image
from .metric_text import display_text, compute_metric
from .draw_chart import compute_chart_df, draw_chart

from ..postprocessing import add_pid_z_paths
from ..dataset import load_yolo_labels

def prepare_pid_predictions(
    df_pred: pd.DataFrame,
    pid: int,
) -> tuple[pd.DataFrame, list[int], pd.DataFrame | None]:
    """
    Filter and prepare prediction DataFrame for one patient.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Predicted boxes, must contain ['pid', 'z', 'conf', 'img_path'].
    pid : int
        Patient ID to visualize or evaluate.

    Returns
    -------
    df_pid : pd.DataFrame
        Predictions for given pid, with a new 'label_path' column.
    z_values : list[int]
        Sorted unique z values for the patient.
    df_gt : pd.DataFrame or None
        Ground-truth boxes (same structure as df_pred)
    """
    df_pred = add_pid_z_paths(df_pred)
    df_pid = df_pred[df_pred["pid"] == pid].copy()

    if df_pid.empty:
        raise ValueError(f"No predictions found for pid={pid}")

    # sort + z list
    df_pid = df_pid.sort_values(["z", "conf"])
    z_values = sorted(df_pid["z"].unique())

    df_gt = None
    if "label_path" in df_pid.columns and df_pid["label_path"].notna().any():
        label_paths = df_pid["label_path"].unique()
        first_path = Path(label_paths[0])
        label_dir = first_path.parent

        # strict sanity check
        assert all(str(Path(p).parent) == str(label_dir) for p in label_paths), \
            "Inconsistent label_dir detected in df_pid['label_path']"

        if label_dir.exists():
            df_gt = add_pid_z_paths(load_yolo_labels(label_dir))
            df_gt = df_gt[df_gt["pid"] == pid]

    return df_pid, z_values, df_gt

def animate_predictions(df_pred: pd.DataFrame, pid: int, metric: bool = True):
    """
    Interactive viewer for visualizing predicted boxes slice-by-slice.
    Each slice (z) is shown only once, even if multiple boxes exist.
    """
    df_pid, z_values, df_gt = prepare_pid_predictions(df_pred, pid)
    if metric:
        df_metric = compute_metric(df_pid, df_gt)
        df_chart = compute_chart_df(df_metric, df_gt)

    def display_slice(z_idx: int, out_img: Output, out_text: Output, out_chart: Output):
        z_val = z_values[z_idx]
        with out_img:
            out_img.clear_output(wait=True)
            draw_slice_image(df_pid, z_val, df_gt)
        if metric:
            with out_text:
                out_text.clear_output(wait=True)
                display_text(df_metric, z_val, df_gt)
            with out_chart:
                out_chart.clear_output(wait=True)
                draw_chart(df_chart, z_val, df_gt)

    navigator = make_slice_navigator(display_slice, num_slices=len(z_values))
    display(navigator)