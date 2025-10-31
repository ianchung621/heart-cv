from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_image(path: Path):
    """Read an image and convert to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"Image not found: {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_pred_boxes(img, df_z: pd.DataFrame):
    """Draw predicted boxes with confidence values."""
    if df_z.empty:
        return img

    idx_max = df_z["conf"].idxmax() if "conf" in df_z.columns else None

    for idx, r in df_z.iterrows():
        x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])
        color = (0, 255, 50) if idx == idx_max else (0, 0, 255)
        thickness = 3 if idx == idx_max else 2
        font_scale = 0.6 if idx == idx_max else 0.45
        text_thickness = 2 if idx == idx_max else 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if "conf" in r:
            cv2.putText(
                img, f"{r.conf:.2f}",
                (x1, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thickness,
                cv2.LINE_AA,
            )
    return img

def draw_label_boxes(img, df_gt_z: pd.DataFrame):
    """Draw a single ground-truth box (YOLO-style, preloaded df)."""
    if df_gt_z is None or df_gt_z.empty:
        return img

    assert len(df_gt_z) == 1, f"Expected exactly one GT box for this slice, got {len(df_gt_z)}"

    r = df_gt_z.iloc[0]
    x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img

def draw_x_stripes(img, df_x_z: pd.DataFrame, alpha: float = 0.4):
    """Lightweight transparent horizontal stripes."""
    if df_x_z is None or df_x_z.empty:
        return img

    color = np.array([255, 0, 255], dtype=float)
    for _, r in df_x_z.iterrows():
        x1, x2, y = int(r.x1), int(r.x2), int(r.y)
        img[y, x1:x2] = (1 - alpha) * img[y, x1:x2] + alpha * color
    return img


def draw_y_stripes(img, df_y_z: pd.DataFrame, alpha: float = 0.4):
    """Lightweight transparent vertical stripes."""
    if df_y_z is None or df_y_z.empty:
        return img

    color = np.array([0, 255, 255], dtype=float)
    for _, r in df_y_z.iterrows():
        y1, y2, x = int(r.y1), int(r.y2), int(r.x)
        img[y1:y2, x] = (1 - alpha) * img[y1:y2, x] + alpha * color
    return img

def draw_slice_image(
    df_pid: pd.DataFrame,
    z_val: int,
    df_gt: pd.DataFrame | None = None,
    df_x: pd.DataFrame | None = None,
    df_y: pd.DataFrame | None = None
):
    """Render prediction and label boxes for one slice."""
    df_z = df_pid[df_pid["z"] == z_val]
    if df_z.empty:
        return

    row = df_z.iloc[0]
    img = load_image(row["img_path"])
    if img is None:
        return

    img = draw_pred_boxes(img, df_z)

    if df_gt is not None:
        df_gt_z = df_gt[df_gt["z"] == z_val]
        if not df_gt_z.empty:
            img = draw_label_boxes(img, df_gt_z)
    
    # Draw projection stripes
    if df_x is not None:
        df_x_z = df_x[df_x["z"] == z_val]
        img = draw_x_stripes(img, df_x_z)

    if df_y is not None:
        df_y_z = df_y[df_y["z"] == z_val]
        img = draw_y_stripes(img, df_y_z)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Slice {z_val}")
    plt.show()