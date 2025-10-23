from pathlib import Path

from heart_cv.config import MODEL_DIR
from heart_cv.val import val_yolo_model

YOLO_DATASET_X = Path("dataset/yolo_datasets/nn_x")
YOLO_DATASET_Y = Path("dataset/yolo_datasets/nn_y")

if __name__ == "__main__":
    val_yolo_model(YOLO_DATASET_X, MODEL_DIR / "nn_x.pt", "nn_x")
    val_yolo_model(YOLO_DATASET_Y, MODEL_DIR / "nn_y.pt", "nn_y")