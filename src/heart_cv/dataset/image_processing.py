from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np
from numba import njit, prange
from tqdm import tqdm

@lru_cache(maxsize=None)
def _exp_kernel(diffusion_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (backward, forward) exponential kernels, normalized separately."""
    half_width = int(2 * diffusion_length)
    z = np.arange(half_width, dtype=np.float32)
    k = np.exp(-z / diffusion_length)
    k /= k.sum()
    # backward weights (z-), forward weights (z+)
    return k[::-1], k  # reversed and normal

@njit(parallel = True)
def _jit_convolve_3D(volume: np.ndarray, w_fwd: np.ndarray, w_bwd: np.ndarray):
    H, W, Z = volume.shape
    Kf = w_fwd.shape[0]
    Kb = w_bwd.shape[0]
    vol_rgb = np.empty((H, W, Z, 3), dtype=np.float32)

    # --- For each slice z
    for z in prange(Z):
        for y in range(H):
            for x in range(W):
                # forward
                fwd_sum = 0.0
                for k in range(Kf):
                    fwd_sum += volume[y, x, min(Z - 1, z + k)] * w_fwd[k]
                # backward
                bwd_sum = 0.0
                for k in range(Kb):
                    idx = z - k
                    if idx < 0:
                        idx = 0
                    bwd_sum += volume[y, x, idx] * w_bwd[k]
                # assign RGB
                vol_rgb[y, x, z, 0] = bwd_sum
                vol_rgb[y, x, z, 1] = volume[y, x, z]
                vol_rgb[y, x, z, 2] = fwd_sum
    return vol_rgb

def load_volume(patient_dir: Path) -> np.ndarray:
    """Load all PNG slices from one patient into (H, W, Z) volume."""
    slice_paths = sorted(patient_dir.glob("*.png"))
    def load_slice(p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img.ndim == 3 and img.shape[2] == 1: # some system reads (H,W,1)
            img = img[:, :, 0] # ensure (H, W)
        return img
    
    with ThreadPoolExecutor() as ex:
        imgs = list(ex.map(load_slice, slice_paths))

    imgs = [img for img in imgs if img is not None]
    volume = np.stack(imgs, axis=-1)
    return volume


def construct_rgb_volume(volume: np.ndarray, method: str = "plain", **kwargs) -> np.ndarray:
    """Construct RGB volume (H, W, Z, 3) based on the selected method."""
    H, W, Z = volume.shape

    if method == "plain":
        vol_rgb = np.repeat(volume[..., None], 3, axis=-1)
        return vol_rgb

    elif method == "nn-stack":
        # Pad along z-axis to handle z-1, z+1 boundaries
        vol_padded = np.pad(volume, ((0, 0), (0, 0), (1, 1)), mode="edge")  # (H, W, Z+2)
        # Stack shifted versions: (H, W, Z, 3)
        z_prev = vol_padded[:, :, :-2]
        z_curr = vol_padded[:, :, 1:-1]
        z_next = vol_padded[:, :, 2:]
        vol_rgb = np.stack([z_prev, z_curr, z_next], axis=-1)
        return vol_rgb
    
    elif method == "diffusion":
        diffusion_length = kwargs.get("diffusion_length", 20)  # typical size is 55.74 +- 9.66
        w_fwd, w_bwd = _exp_kernel(diffusion_length)
        vol_rgb = _jit_convolve_3D(volume, w_fwd, w_bwd)
        return np.clip(vol_rgb, 0, 255).astype(np.uint8)
        

    else:
        raise ValueError(f"Unknown RGB method: {method}")

def save_rgb_slices(
    volume_rgb: np.ndarray,
    patient_dir: Path,
    split: str,
    image_dst: Path,
    label_src: Path,
    label_dst: Path,
    only_label: bool = False
):
    """Save each slice of the RGB volume to YOLO dataset folder."""
    patient_id = patient_dir.name
    Z = volume_rgb.shape[2]

    def save_one(z):
        img_name = f"{patient_id}_{z+1:04d}.png"
        dst_img = image_dst / split / img_name
        dst_lbl = label_dst / split / img_name.replace(".png", ".txt")
        cv2.imwrite(str(dst_img), volume_rgb[..., z, :])
        label_path = label_src / patient_id / dst_lbl.name
        if label_path.exists():
            shutil.copy(label_path, dst_lbl)
    
    if only_label:
        label_paths = list((label_src / patient_id).glob("*.txt"))
        z_idxs = [int(p.stem.split("_")[-1]) - 1 for p in label_paths]  # 0-based indices
    else:
        z_idxs = range(Z)

    with ThreadPoolExecutor() as ex:
        ex.map(save_one, z_idxs)

# ---------------------------
# Public Entry Function
# ---------------------------

def convert_to_rgb(
    patient_dir: Path,
    split: str,
    image_dst: Path,
    label_src: Path,
    label_dst: Path,
    method: str = "plain",
    only_label: bool = False,
    **kwargs
):
    """Load full (H, W, Z), transform into RGB volume, and save per-slice images."""

    volume = load_volume(patient_dir)
    volume_rgb = construct_rgb_volume(volume, method=method, **kwargs)
    save_rgb_slices(volume_rgb, patient_dir, split, image_dst, label_src, label_dst, only_label)

def prepare_yolo_test_images(
    img_src: Path,
    img_dst: Path,
    method: str = "plain",
    **kwargs
):
    """
    Prepare RGB test images for YOLO inference.
    All resulting PNGs are saved flat in img_dst (no patient subfolders).

    Parameters
    ----------
    img_src : Path
        Root directory containing patient subfolders with PNG slices.
    img_dst : Path
        Destination directory for RGB-constructed PNG images.
    method : str
        RGB construction method ("plain", "nn-stack", "diffusion").
    **kwargs :
        Extra keyword arguments passed to construct_rgb_volume().
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted([p for p in img_src.glob("*") if p.is_dir()])

    for patient_dir in tqdm(patient_dirs, desc="processing patients"):
        patient_id = patient_dir.name

        volume = load_volume(patient_dir)
        volume_rgb = construct_rgb_volume(volume, method=method, **kwargs)
        Z = volume_rgb.shape[2]

        def save_one(z):
            img_name = f"{patient_id}_{z+1:04d}.png"
            dst_img = img_dst / img_name
            cv2.imwrite(str(dst_img), volume_rgb[..., z, :])

        with ThreadPoolExecutor() as ex:
            ex.map(save_one, range(Z))

    print(f"✅ Finished preparing test images -> {img_dst}")