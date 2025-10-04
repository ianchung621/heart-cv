from pathlib import Path
from typing import Literal
import cv2
import numpy as np

def convert_to_rgb(src_path: Path, dst_path: Path, method: Literal["plain"]):
    """Convert grayscale to pseudo RGB and save."""
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {src_path}")
    
    if method == "plain":
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(str(dst_path), img)