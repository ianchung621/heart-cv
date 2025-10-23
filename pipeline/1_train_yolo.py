from pathlib import Path

from heart_cv.dataset import prepare_yolo_dataset
from heart_cv.train import train_yolo_model
from heart_cv.config import SPLIT_PATIENT_DICT

DATASET_DIR = Path("dataset")
IMAGES_SRC = DATASET_DIR / "training_image"
LABELS_SRC = DATASET_DIR / "training_label"

YOLO_DATASETS = DATASET_DIR / "yolo_datasets"
NN = YOLO_DATASETS/ "nn"
DIFFUSION_3 = YOLO_DATASETS/ "diffusion_3"

def prepare():
    prepare_yolo_dataset(IMAGES_SRC, LABELS_SRC, NN, 
        split_patient_dict = SPLIT_PATIENT_DICT, method = "nn-stack")
    prepare_yolo_dataset(IMAGES_SRC, LABELS_SRC, DIFFUSION_3,
        split_patient_dict = SPLIT_PATIENT_DICT, method = "diffusion", diffusion_length = 3)

def train():
    train_yolo_model(NN, no_background=True, model_name='nn')
    train_yolo_model(DIFFUSION_3, no_background=True, model_name='diffusion_3')

def main():
    prepare()
    train()

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"\nDone in {(time.time()-start)/60:.2f} min")