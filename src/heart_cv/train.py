import os
import shutil

from ultralytics.models import YOLO

from .config import *

def build_no_background_dataset(base_dataset: Path) -> Path:
    """
    Build a temporary dataset directory with only labeled images using *absolute-path symlinks*.
    Returns the path to the filtered dataset root.
    """

    yaml_dir = base_dataset.parent / f"{base_dataset.name}_no_bg"
    yaml_base = str(base_dataset)
    yaml_filtered = str(yaml_dir)

    base_dataset = base_dataset.resolve()  # ensure absolute
    filtered_dir = yaml_dir.resolve()

    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)
    filtered_dir.mkdir(parents=True)

    # Copy dataset.yaml template
    yaml_src = base_dataset / "dataset.yaml"
    yaml_dst = filtered_dir / "dataset.yaml"

    for subset in ["train", "val", "test"]:
        labels_dir = base_dataset / "labels" / subset
        images_dir = base_dataset / "images" / subset
        if not labels_dir.exists() or not images_dir.exists():
            continue

        new_labels_dir = filtered_dir / "labels" / subset
        new_images_dir = filtered_dir / "images" / subset
        new_labels_dir.mkdir(parents=True, exist_ok=True)
        new_images_dir.mkdir(parents=True, exist_ok=True)

        # Only include samples that have corresponding label files
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size == 0:
                continue  # skip empty labels

            img_name = label_file.stem
            for ext in (".jpg", ".jpeg", ".png"):
                img_file = images_dir / f"{img_name}{ext}"
                if img_file.exists():
                    img_link = new_images_dir / img_file.name
                    label_link = new_labels_dir / label_file.name

                    # Create absolute-path symlinks
                    os.symlink(img_file.resolve(), img_link)
                    os.symlink(label_file.resolve(), label_link)

    # Rewrite YAML with absolute paths
    yaml_text = yaml_src.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace(yaml_base, yaml_filtered)
    yaml_dst.write_text(yaml_text, encoding="utf-8")

    return yaml_dir

def train_yolo_model(
    yolo_dataset: Path,
    no_background: bool = False,
    model_name: str | None = None
) -> str:
    """
    Train YOLO model on the given dataset.
    If `no_background=True`, builds a filtered symlink dataset excluding empty labels.
    """
    dataset_path = build_no_background_dataset(yolo_dataset) if no_background else yolo_dataset

    model = YOLO(YOLO_MODEL_NAME)
    results = model.train(
        data=str(dataset_path / "dataset.yaml"),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH,
        verbose=False,
        lr0 = YOLO_LR0,
        cos_lr=True,
        amp=True,
        optimizer='adamw',
        seed=SEED,
        deterministic=True,
        workers = YOLO_WORKERS,
        # regularization and data-augumention
        weight_decay=YOLO_WEIGHT_DECAY,
        mixup=YOLO_MIXUP,
        mosaic=YOLO_MOSAIC,
        degrees=YOLO_DEGREES,
        translate=YOLO_TRANSLATE,
        scale=YOLO_SCALE,
        shear=YOLO_SHEAR,
        perspective=YOLO_PERSPECTIVE,
        flipud=YOLO_FLIPUD,
        fliplr=YOLO_FLIPLR,
        close_mosaic=YOLO_CLOSE_MOSAIC_LAST,
        val=True
    )
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    print(f"[YOLO] best: {best_model_path}")

    if model_name:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        final_path = models_dir / f"{model_name}.pt"
        shutil.copy2(best_model_path, final_path)
        print(f"[YOLO] saved to: {final_path}")
        return str(final_path)