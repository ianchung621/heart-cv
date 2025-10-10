from pathlib import Path
from typing import Literal
import shutil

from tqdm import tqdm

from .split import split_patients
from .image_processing import convert_to_rgb

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
    split_patient_dict: dict[str, list] | None = None,
    method: Literal["plain", "nn-stack", "diffusion"] = "plain",
    seed: int = 42,
    only_label: bool = False,
    ** kwargs
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
    if split_patient_dict:
        split_dict = {split: [image_src/p for p in patients] for split, patients in split_patient_dict.items()}
    else:
        patient_dirs = [d for d in image_src.iterdir() if d.is_dir()]
        split_dict = split_patients(sorted(patient_dirs), split_ratio, seed)

    # process each split
    for split, patient_subset in split_dict.items():
        for patient_dir in tqdm(patient_subset, desc=f"Preparing {split} patients"):
            convert_to_rgb(
                patient_dir=patient_dir,
                split=split,
                image_dst=image_dst,
                label_src=label_src,
                label_dst=label_dst,
                method=method,
                only_label=only_label,
                **kwargs
            )
    
    write_dataset_yaml(yolo_dataset)

    print("✅ Dataset prepared at:", yolo_dataset)