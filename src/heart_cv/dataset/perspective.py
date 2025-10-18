from pathlib import Path
from typing import Literal
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from tqdm import tqdm

from .load_label import load_yolo_labels
from .image_processing import load_volume


def create_view(
    patient_img_dir: Path,
    patient_label_dir: Path | None,
    perspective: Literal['x', 'y'],
    final_shape: tuple[int, int, int] = (512, 512, 512),
):
    """
    Construct a 3D volume and reorient to x- or y-view.
    Pads to (512,512,512) with zeros unsymmetrically (pad only on max end).

    Parameters
    ----------
    patient_img_dir : Path
        Directory containing *.png slices along z.
    patient_label_dir : Path | None
        Directory containing YOLO label *.txt files.
    perspective : Literal['x','y']
        Orientation of the generated view.
    final_shape : tuple[int,int,int]
        Target 3D shape (H,W,Z).
    """
    volume = load_volume(patient_img_dir)  # (H, W, Z)
    H, W, Z = volume.shape
    Hf, Wf, Zf = final_shape

    # pad unsymmetrically to target shape
    pad_h, pad_w, pad_z = max(0, Hf - H), max(0, Wf - W), max(0, Zf - Z)
    volume = np.pad(volume, ((0, pad_h), (0, pad_w), (0, pad_z)))
    H, W, Z = volume.shape  # update to padded sizes

    mask = None
    if patient_label_dir is not None:
        label_df = load_yolo_labels(patient_label_dir)  # ['img','cls','conf','x1','y1','x2','y2']
        label_df = label_df.sort_values('img')
        mask = np.zeros_like(volume, dtype=np.uint8)
        for _, row in label_df.iterrows():
            # patient0001_0174 -> z_idx = 173 (0-based)
            z_idx = int(str(row['img']).split('_')[-1]) - 1
            x1, y1, x2, y2 = map(int, np.round([row.x1, row.y1, row.x2, row.y2]))
            # clamp to image bounds (robustness)
            x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H))
            if x2 > x1 and y2 > y1 and 0 <= z_idx < Z:
                mask[y1:y2, x1:x2, z_idx] = 1

    if perspective == 'x':
        # slices are fixed x -> slice plane (Z, W)
        final_volume = np.transpose(volume, (0, 2, 1))  # (H, Z, W)
        N, H2, W2 = final_volume.shape[0], Z, W
        boxes_per_slice: list[None | np.ndarray] = [None] * N
        if mask is not None:
            mask_x = np.transpose(mask, (0, 2, 1))  # (H, Z, W)
            for i in range(N):  # x-index
                m = mask_x[i]            # (Z, W)
                zs, ws = np.where(m > 0)
                if zs.size == 0:
                    continue
                z1, z2 = zs.min(), zs.max()
                w1, w2 = ws.min(), ws.max()
                # YOLO in slice coords: (xc,yc,w,h) → here axes = (Z, W)
                zc = (z1 + z2) / 2 / Z
                wc = (w1 + w2) / 2 / W
                h = (z2 - z1) / Z
                w_ = (w2 - w1) / W
                boxes_per_slice[i] = np.array([[0, wc, zc, w_, h]], dtype=float)

        # reorder to (N, H2, W2)
        final_volume = final_volume  # already (H, Z, W) with N=H as first axis

    elif perspective == 'y':
        # slices are fixed y -> slice plane (Z, H)
        final_volume = np.transpose(volume, (1, 2, 0))  # (W, Z, H)
        N, H2, W2 = final_volume.shape[0], Z, H  # here "H2" is Z; "W2" is H for the slice frame
        boxes_per_slice: list[None | np.ndarray] = [None] * N
        if mask is not None:
            mask_y = np.transpose(mask, (1, 2, 0))  # (W, Z, H)
            for i in range(N):  # y-index
                m = mask_y[i]            # (Z, H)
                zs, hs = np.where(m > 0)
                if zs.size == 0:
                    continue
                z1, z2 = zs.min(), zs.max()
                h1, h2 = hs.min(), hs.max()
                # YOLO in slice coords: axes = (Z, H)
                zc = (z1 + z2) / 2 / Z
                hc = (h1 + h2) / 2 / H
                h = (z2 - z1) / Z
                w_ = (h2 - h1) / H
                boxes_per_slice[i] = np.array([[0, hc, zc, w_, h]], dtype=float)

        # already (N, H2, W2) = (W, Z, H)

    else:
        raise ValueError("perspective must be 'x' or 'y'")

    return final_volume, boxes_per_slice

def prepare_perspective_datasets(
    img_src: Path,
    label_src: Path | None,
    img_dst: Path,
    label_dst: Path | None,
    perspective: Literal["x", "y"],
):
    """
    Generate perspective-specific YOLO datasets (x or y view).

    Processes each patient sequentially (with tqdm progress bar),
    but writes slices (Z dimension) in parallel.
    """
    if img_dst.exists():
        print(f"⚠️ Removing existing dataset at {img_dst} ...")
        shutil.rmtree(img_dst)
    if label_dst is not None and label_dst.exists():
        print(f"⚠️ Removing existing dataset at {label_dst} ...")
        shutil.rmtree(label_dst)

    img_dst.mkdir(parents=True, exist_ok=True)
    if label_dst is not None:
        label_dst.mkdir(parents=True, exist_ok=True)

    patient_img_dirs = sorted([p for p in img_src.glob("*") if p.is_dir()])

    for patient_dir in tqdm(patient_img_dirs, desc=f"Preparing {perspective}-view"):
        pid = patient_dir.name
        patient_label_dir = (label_src / pid) if label_src is not None else None

        volume, boxes_per_slice = create_view(patient_dir, patient_label_dir, perspective)

        # per-patient subfolders
        (img_dst / pid).mkdir(parents=True, exist_ok=True)
        if label_dst is not None:
            (label_dst / pid).mkdir(parents=True, exist_ok=True)

        N = volume.shape[0]

        def save_slice(i: int):
            img_name = f"{pid}_{i+1:04d}.png"
            dst_img_path = img_dst / pid / img_name
            cv2.imwrite(str(dst_img_path), volume[i])

            if label_dst is not None:
                bxs = boxes_per_slice[i]
                if bxs is not None and len(bxs) > 0:
                    dst_lbl_path = label_dst / pid / img_name.replace(".png", ".txt")
                    with open(dst_lbl_path, "w") as f:
                        for (cls, xc, yc, w, h) in bxs:
                            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                # else: no label file for this slice

        with ThreadPoolExecutor() as ex:
            list(ex.map(save_slice, range(N)))

    print(f"✅ Finished preparing {perspective}-view dataset at: {img_dst}")

