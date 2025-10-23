from pathlib import Path

from heart_cv.test import test_yolo_model
from heart_cv.dataset import prepare_yolo_test_images
from heart_cv.config import MODEL_DIR

model_names = ["nn_x", "nn_y"]
PERSPECTIVE_X_TEST = Path("dataset/perspectives/x/testing_image")
PERSPECTIVE_Y_TEST = Path("dataset/perspectives/y/testing_image")
IMG_DST = Path("dataset/yolo_datasets/test")

def prepare():
    IMG_DST.mkdir(parents=True, exist_ok=True)
    print("Preparing test image variants...")

    prepare_yolo_test_images(PERSPECTIVE_X_TEST, IMG_DST / model_names[0], "nn-stack")
    prepare_yolo_test_images(PERSPECTIVE_Y_TEST, IMG_DST / model_names[1], "nn-stack")

    print("✅ Finished preparing test datasets.")


def inference():
    img_dirs = [PERSPECTIVE_X_TEST, PERSPECTIVE_Y_TEST]
    for img_dir, mn in zip(img_dirs, model_names):
        test_yolo_model(IMG_DST / mn, MODEL_DIR/f"{mn}.pt", mn)

def main():
    prepare()
    inference()

if __name__ == "__main__":
    main()