import pandas as pd
import numpy as np
from pathlib import Path

def load_yolo_labels(label_dir: str | Path, image_size: int = 512) -> pd.DataFrame:
    """
    Load YOLO-format labels into a unified DataFrame.

    Parameters
    ----------
    label_dir : str or Path
        Directory containing YOLO .txt files (cls xc yc w h [conf]).
    image_size : int
        Image width/height in pixels.

    Returns
    -------
    pd.DataFrame
        Columns: ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']
    """
    label_dir = Path(label_dir)
    rows = []

    txt_files = sorted(label_dir.rglob("*.txt"))
    if not txt_files:
        print(f"⚠️ No .txt files found in {label_dir}")
        return pd.DataFrame(columns=["img", "cls", "conf", "x1", "y1", "x2", "y2"])

    for txt_path in txt_files:
        img_name = txt_path.stem  # e.g. patient0049_0299
        data = np.loadtxt(txt_path, ndmin=2)

        # Handle empty files
        if data.size == 0:
            continue

        for row in data:
            cls, xc, yc, w, h = row[:5]
            conf = row[5] if len(row) > 5 else np.nan

            x1 = (xc - w / 2) * image_size
            y1 = (yc - h / 2) * image_size
            x2 = (xc + w / 2) * image_size
            y2 = (yc + h / 2) * image_size

            rows.append([img_name, int(cls), conf, x1, y1, x2, y2])

    df = pd.DataFrame(rows, columns=["img", "cls", "conf", "x1", "y1", "x2", "y2"])
    return df