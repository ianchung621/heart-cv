from pathlib import Path

from heart_cv.dataset import prepare_perspective_datasets, prepare_yolo_dataset
from heart_cv.config import SPLIT_PATIENT_DICT


# ==============================
# 📁 Constants (root folders)
# ==============================

# Raw data
IMG_SRC = Path("dataset/training_image")
LABEL_SRC = Path("dataset/training_label")
TEST_IMG_SRC = Path("dataset/testing_image")

# Perspective datasets
PERSPECTIVE_X_IMG = Path("dataset/perspectives/x/training_image")
PERSPECTIVE_X_LABEL = Path("dataset/perspectives/x/training_label")
PERSPECTIVE_Y_IMG = Path("dataset/perspectives/y/training_image")
PERSPECTIVE_Y_LABEL = Path("dataset/perspectives/y/training_label")
PERSPECTIVE_X_TEST = Path("dataset/perspectives/x/testing_image")
PERSPECTIVE_Y_TEST = Path("dataset/perspectives/y/testing_image")

# YOLO datasets
YOLO_DATASET_X = Path("dataset/yolo_datasets/nn_x")
YOLO_DATASET_Y = Path("dataset/yolo_datasets/nn_y")


# ==============================
# 🧩 Stage 1: Prepare perspectives
# ==============================

def prepare_perspective():
    print("\n=== Stage 1: Generating perspective datasets ===")

    # --- Training: x and y views ---
    prepare_perspective_datasets(
        IMG_SRC,
        LABEL_SRC,
        PERSPECTIVE_X_IMG,
        PERSPECTIVE_X_LABEL,
        "x",
    )
    prepare_perspective_datasets(
        IMG_SRC,
        LABEL_SRC,
        PERSPECTIVE_Y_IMG,
        PERSPECTIVE_Y_LABEL,
        "y",
    )

    # --- Testing: x and y views ---
    prepare_perspective_datasets(
        TEST_IMG_SRC,
        None,
        PERSPECTIVE_X_TEST,
        None,
        "x",
    )
    prepare_perspective_datasets(
        TEST_IMG_SRC,
        None,
        PERSPECTIVE_Y_TEST,
        None,
        "y",
    )


# ==============================
# 🧩 Stage 2: Prepare YOLO datasets
# ==============================

def prepare_yolo():
    print("\n=== Stage 2: Building YOLO training datasets ===")

    prepare_yolo_dataset(
        PERSPECTIVE_X_IMG,
        PERSPECTIVE_X_LABEL,
        YOLO_DATASET_X,
        split_patient_dict=SPLIT_PATIENT_DICT,
        method="nn-stack",
    )

    prepare_yolo_dataset(
        PERSPECTIVE_Y_IMG,
        PERSPECTIVE_Y_LABEL,
        YOLO_DATASET_Y,
        split_patient_dict=SPLIT_PATIENT_DICT,
        method="nn-stack",
    )


# ==============================
# 🚀 Entry point
# ==============================

def main():
    prepare_perspective()
    prepare_yolo()
    print("\n✅ All perspective and YOLO datasets prepared successfully!")


if __name__ == "__main__":
    main()