
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

from .utils import make_slice_navigator
from ..postprocessing import add_pid_z_paths

def animate_predictions(df_pred: pd.DataFrame, pid: int, show_label: bool = True):
    """
    Interactive viewer for visualizing predicted boxes slice-by-slice.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Columns must include ['pid', 'z', 'x1', 'y1', 'x2', 'y2', 'img'] or can infer via add_pid_z_paths().
    pid : int
        Patient ID to visualize.
    show_label : bool, default=True
        Show ground-truth boxes if label files exist.
    """
    # ensure path columns
    df_pred = add_pid_z_paths(df_pred)
    df_pid = df_pred[df_pred["pid"] == pid].copy()

    if df_pid.empty:
        raise ValueError(f"No predictions found for pid={pid}")

    # sort by z for smooth navigation
    df_pid = df_pid.sort_values(["z","conf"])

    # prepare image/label paths
    z_values = df_pid["z"].tolist()

    def draw_slice(z_idx: int):
        row = df_pid.iloc[z_idx]
        img = cv2.imread(str(row["img_path"]))
        if img is None:
            print(f"Image not found: {row['img_path']}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- draw prediction boxes ---
        df_z = df_pid[df_pid["z"] == row["z"]]
        
        if not df_z.empty:
            # find the box with maximum confidence
            idx_max = df_z["conf"].idxmax() if "conf" in df_z.columns else None

            for idx, r in df_z.iterrows():
                x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])
                color = (0, 0, 255)  # blue for normal boxes
                thickness = 2
                font_scale = 0.45
                text_thickness = 1

                # highlight the max-confidence box
                if idx == idx_max:
                    color = (0, 255, 50)  # green
                    thickness = 3
                    font_scale = 0.6
                    text_thickness = 2

                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if "conf" in r:
                    cv2.putText(
                        img, f"{r.conf:.2f}", (x1, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

        # --- draw ground-truth boxes if available ---
        if show_label and row.get("label_path") and Path(row["label_path"]).exists():
            with open(row["label_path"]) as f:
                for line in f:
                    _, xc, yc, w, h = map(float, line.split())
                    H, W = img.shape[:2]
                    x1 = int((xc - w / 2) * W)
                    x2 = int((xc + w / 2) * W)
                    y1 = int((yc - h / 2) * H)
                    y2 = int((yc + h / 2) * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Patient {pid}, Slice {row['z']}")
        plt.show()

    #interact(
    #    draw_slice,
    #    z_idx=IntSlider(min=0, max=len(z_values) - 1, step=1, value=0,
    #                    description="z-index", continuous_update=False),
    #)
    make_slice_navigator(draw_slice, num_slices=len(z_values))