import random
from pathlib import Path

def split_patients(patient_dirs: list[Path], ratio=(0.7, 0.2, 0.1), seed: int = 42) -> dict[str, list[Path]]:
    """Randomly split patient directories by ratio (train, val, test)."""

    rnd = random.Random(seed)
    patient_dirs = list(patient_dirs)
    rnd.shuffle(patient_dirs)

    n = len(patient_dirs)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    split_dict = {
        "train": patient_dirs[:n_train],
        "val": patient_dirs[n_train:n_train+n_val],
        "test": patient_dirs[n_train+n_val:]
    }

    print("✅ Patient split:")
    for k, v in split_dict.items():
        print(f"  {k:<5}: {len(v)} patients")
    return split_dict