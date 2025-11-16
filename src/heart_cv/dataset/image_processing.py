from pathlib import Path
import shutil
import re
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

@njit(parallel = True)
def _jit_convolve_3D_1(volume: np.ndarray, w_fwd: np.ndarray, w_bwd: np.ndarray):
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
                    k += 1
                    fwd_sum += volume[y, x, min(Z - 1, z + k)] * w_fwd[k]
                # backward
                bwd_sum = 0.0
                for k in range(Kb):
                    k += 1
                    idx = z - k
                    if idx < 0:
                        idx = 0
                    bwd_sum += volume[y, x, idx] * w_bwd[k]
                # assign RGB
                vol_rgb[y, x, z, 0] = bwd_sum
                vol_rgb[y, x, z, 1] = volume[y, x, z]
                vol_rgb[y, x, z, 2] = fwd_sum
    return vol_rgb

def normalize_volume_uint8(
    volume: np.ndarray,
    p_low: float = 5.0,
    p_high: float = 95.0,
    return_uint8: bool = True,
) -> np.ndarray:
    """
    Normalize a CT volume (H,W,D) in uint8 using patient-specific dynamic range.

    Steps:
      1. Estimate robust low/high bounds using percentiles.
      2. Clip to [lo, hi].
      3. Min–max normalize to [0,1].
      4. Optionally rescale back to uint8.

    Parameters
    ----------
    volume : np.ndarray
        uint8 array of shape (H, W, D).
    p_low : float
        Lower percentile for dynamic range (default 5).
    p_high : float
        Upper percentile for dynamic range (default 95).
    return_uint8 : bool
        If True → output uint8 in [0,255].
        If False → output float32 in [0,1].

    Returns
    -------
    np.ndarray
        Normalized volume, either float32 or uint8.
    """
    # Percentile-based patient-specific bounds
    lo = float(np.percentile(volume, p_low))
    hi = float(np.percentile(volume, p_high))
    if hi <= lo:
        hi = lo + 1.0

    # Min–max normalization to [0,1]
    vol_f = volume.astype(np.float32)
    vol_f = np.clip(vol_f, lo, hi)
    vol_f = (vol_f - lo) / (hi - lo)

    if return_uint8:
        return (vol_f * 255.0).astype(np.uint8)
    return vol_f

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

    elif method == "nn-stack":
        # Pad along z-axis to handle z-1, z+1 boundaries
        vol_padded = np.pad(volume, ((0, 0), (0, 0), (1, 1)), mode="edge")  # (H, W, Z+2)
        # Stack shifted versions: (H, W, Z, 3)
        z_prev = vol_padded[:, :, :-2]
        z_curr = vol_padded[:, :, 1:-1]
        z_next = vol_padded[:, :, 2:]
        vol_rgb = np.stack([z_prev, z_curr, z_next], axis=-1)
    
    elif method == "diffusion":
        diffusion_length = kwargs.get("diffusion_length", 20)  # typical size is 55.74 +- 9.66
        w_fwd, w_bwd = _exp_kernel(diffusion_length)
        vol_rgb = _jit_convolve_3D(volume, w_fwd, w_bwd)
        vol_rgb =  np.clip(vol_rgb, 0, 255).astype(np.uint8)
    
    elif method == "diff1":
        diffusion_length = kwargs.get("diffusion_length", 20)  # typical size is 55.74 +- 9.66
        w_fwd, w_bwd = _exp_kernel(diffusion_length)
        vol_rgb = _jit_convolve_3D_1(volume, w_fwd, w_bwd)
        vol_rgb = np.clip(vol_rgb, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown RGB method: {method}")
    
    normalize = kwargs.get("norm", False)
    if normalize:
        vol_rgb = normalize_volume_uint8(vol_rgb)
    
    return vol_rgb

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
    patient_id = patient_dir.name  # e.g. "patient0001"

    # --- detect starting z index from existing image names --- #
    existing_imgs = list(patient_dir.glob(f"{patient_id}_*.png"))
    if not existing_imgs:
        raise FileNotFoundError(f"No slice images found under {patient_dir}")
    # Extract z indices using regex pattern
    z_indices = []
    for p in existing_imgs:
        m = re.search(r"_(\d{4})\.png$", p.name)
        if m:
            z_indices.append(int(m.group(1)))
    z_start = min(z_indices)  # actual starting slice number
    Z = volume_rgb.shape[2]

    def save_one(local_z: int):
        z_global = z_start + local_z  # convert to dataset z index
        img_name = f"{patient_id}_{z_global:04d}.png"
        dst_img = image_dst / split / img_name
        dst_lbl = label_dst / split / img_name.replace(".png", ".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(dst_img), volume_rgb[..., local_z, :])

        # copy label if exists
        label_path = label_src / patient_id / dst_lbl.name
        if label_path.exists():
            shutil.copy(label_path, dst_lbl)

    if only_label:
        label_paths = list((label_src / patient_id).glob("*.txt"))
        z_idxs = [int(p.stem.split("_")[-1]) - 1 for p in label_paths]
    else:
        z_idxs = range(Z)

    with ThreadPoolExecutor() as ex:
        list(ex.map(save_one, z_idxs))

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

        # --- detect z_start from filenames --- #
        existing_imgs = list(patient_dir.glob(f"{patient_id}_*.png"))
        if not existing_imgs:
            print(f"⚠️ No slices found for {patient_id}")
            continue
        z_indices = []
        for p in existing_imgs:
            m = re.search(r"_(\d{4})\.png$", p.name)
            if m:
                z_indices.append(int(m.group(1)))
        z_start = min(z_indices)

        volume = load_volume(patient_dir)
        volume_rgb = construct_rgb_volume(volume, method=method, **kwargs)
        Z = volume_rgb.shape[2]

        def save_one(local_z: int):
            z_global = z_start + local_z
            img_name = f"{patient_id}_{z_global:04d}.png"
            dst_img = img_dst / img_name
            cv2.imwrite(str(dst_img), volume_rgb[..., local_z, :])

        with ThreadPoolExecutor() as ex:
            list(ex.map(save_one, range(Z)))


    print(f"✅ Finished preparing test images -> {img_dst}")