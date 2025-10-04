from pathlib import Path
from typing import Literal
import shutil

from tqdm import tqdm

from .split import split_patients
from .image_processing import convert_to_rgb

def collect_image_label_pairs(patients: list[Path], label_src: Path) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) pairs for given patients containg png images."""
    pairs = []
    for patient_dir in patients:
        for img_path in patient_dir.glob("*.png"):
            label_path = label_src / patient_dir.name / (img_path.stem + ".txt")
            if label_path.exists():
                pairs.append((img_path, label_path))
    return pairs

def write_dataset_yaml(yolo_dataset: Path):
    yaml_text = f"""# Auto-generated YOLO dataset config
path: {yolo_dataset}
train: images/train
val: images/val
test: images/test

nc: 1
names: ["valve"]
"""
    with open(yolo_dataset / "dataset.yaml", "w") as f:
        f.write(yaml_text)
    print(f"✅ Wrote {yolo_dataset}/dataset.yaml")

def prepare_yolo_dataset(
    image_src: Path,
    label_src: Path,
    yolo_dataset: Path,
    splits: tuple[str] = ("train", "val", "test"),
    split_ratio: tuple[float] = (0.7, 0.2, 0.1),
    method: Literal["plain"] = "plain",
    seed: int = 42
):
    print("Preparing YOLO dataset...")

    if yolo_dataset.exists():
        print(f"⚠️ Removing existing dataset at {yolo_dataset} ...")
        shutil.rmtree(yolo_dataset)

    image_dst = yolo_dataset / "images"
    label_dst = yolo_dataset / "labels"

    for split in splits:
        (image_dst / split).mkdir(parents=True, exist_ok=True)
        (label_dst / split).mkdir(parents=True, exist_ok=True)

    # split by patient
    patient_dirs = [d for d in image_src.iterdir() if d.is_dir()]
    split_dict = split_patients(patient_dirs, split_ratio, seed)

    # process each split
    for split, patient_subset in split_dict.items():
        pairs = collect_image_label_pairs(patient_subset, label_src)
        for img_path, label_path in tqdm(pairs, desc = f"Preparing {split} images and labels"):
            dst_img = image_dst / split / img_path.name
            dst_lbl = label_dst / split / label_path.name

            convert_to_rgb(img_path, dst_img, method)
            shutil.copy(label_path, dst_lbl)
    
    write_dataset_yaml(yolo_dataset)

    print("✅ Dataset prepared at:", yolo_dataset)