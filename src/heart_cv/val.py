import os
from pathlib import Path

from ultralytics.models import YOLO
from heart_cv.config import VAL_DIR, YOLO_IMGSZ, YOLO_WORKERS
from heart_cv.dataset import load_yolo_labels, Cropper

def val_yolo_model(yolo_dataset: Path, model_path: Path, csv_name: str):

    model = YOLO(model_path)

    results = model.val(
        data=str(yolo_dataset / "dataset.yaml"),
        imgsz = YOLO_IMGSZ,
        workers = YOLO_WORKERS,
        save_txt = True,
        save_conf = True,
        plots = True
    )

    label_dir = os.path.join(results.save_dir, 'labels')
    csv_path = VAL_DIR / f'{csv_name}.csv'
    load_yolo_labels(label_dir).to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to {csv_path}")

def val_cropped_yolo_model(yolo_dataset: Path, model_path: Path, csv_name: str):

    model = YOLO(model_path)

    results = model.val(
        data=str(yolo_dataset / "dataset.yaml"),
        imgsz = 256,
        workers = YOLO_WORKERS,
        save_txt = True,
        save_conf = True,
        plots = True
    )

    label_dir = os.path.join(results.save_dir, 'labels')
    csv_path = VAL_DIR / f'{csv_name}.csv'
    Cropper().load_cropped_label_dir(label_dir).to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to {csv_path}")