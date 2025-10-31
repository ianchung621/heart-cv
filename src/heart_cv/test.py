from pathlib import Path
import os

from ultralytics.models import YOLO

from .config import TEST_DIR, YOLO_IMGSZ
from .dataset import load_yolo_labels

def test_yolo_model(image_dir: Path, model_path: Path, csv_name: str, csv_dir: Path = TEST_DIR):
    """Run YOLO inference and save results as CSV."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔍 Running inference: {csv_name}")

    model = YOLO(model_path)

    results_gen = model.predict(
        source=image_dir,
        imgsz=YOLO_IMGSZ,
        workers=1,
        conf = 0.001,
        batch = 16,
        save_conf = True,
        save_txt = True,
        half = True,
        stream = True
    )
    
    first = next(results_gen)
    save_dir = first.save_dir
    for _ in results_gen:
        pass

    label_dir = os.path.join(save_dir, "labels")
    df_pred = load_yolo_labels(label_dir)
    csv_path = csv_dir / f"{csv_name}.csv"
    df_pred.to_csv(csv_path, index=False)
    print(f"✅ Saved predictions to {csv_path}")
