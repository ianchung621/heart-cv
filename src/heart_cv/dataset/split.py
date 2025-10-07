import numpy as np
from pathlib import Path

def split_patients(patient_dirs: list[Path], ratio=(0.7, 0.2, 0.1), seed: int = 42) -> dict[str, list[Path]]:
    """Deterministically split patient directories into train/val/test sets across all machines."""
    rng = np.random.default_rng(seed)
    patient_dirs = np.array(patient_dirs)
    rng.shuffle(patient_dirs)

    n = len(patient_dirs)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    split_dict = {
        "train": list(patient_dirs[:n_train]),
        "val": list(patient_dirs[n_train:n_train + n_val]),
        "test": list(patient_dirs[n_train + n_val:])
    }

    print("✅ Patient split:")
    for k, v in split_dict.items():
        print(f"  {k:<5}: {len(v)} patients")
    return split_dict