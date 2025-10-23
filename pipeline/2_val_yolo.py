from pathlib import Path

from heart_cv.config import MODEL_DIR
from heart_cv.val import val_yolo_model

DATASET_DIR = Path("dataset")
YOLO_DATASETS = DATASET_DIR / "yolo_datasets"

NN = YOLO_DATASETS/ "nn"
DIFFUSION_3 = YOLO_DATASETS/ "diffusion_3"

if __name__ == "__main__":
    val_yolo_model(NN, MODEL_DIR / "nn.pt", "nn")
    val_yolo_model(DIFFUSION_3, MODEL_DIR / "diffusion_3.pt", "diffusion_3")