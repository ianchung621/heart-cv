from pathlib import Path
from heart_cv.dataset import prepare_perspective_datasets

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

def prepare_run_dir():
    print("\n=== Stage 2: Generating run dirs ===")
    from heart_cv.config import TRAIN_DIR, VAL_DIR, TEST_DIR
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Finished preparing run dirs at {TRAIN_DIR, VAL_DIR, TEST_DIR}")

def main():
    prepare_perspective()
    prepare_run_dir()

if __name__ == "__main__":
    main()