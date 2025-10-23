from pathlib import Path

from heart_cv.config import MODEL_DIR
from heart_cv.dataset import prepare_yolo_test_images
from heart_cv.test import test_yolo_model


IMG_SRC = Path("dataset/testing_image")
IMG_DST = Path("dataset/yolo_datasets/test")

model_names = ["nn", "diffusion_3"]


def prepare():
    """Prepare YOLO test images in flat structure for all RGB construction methods."""
    IMG_DST.mkdir(parents=True, exist_ok=True)
    print("Preparing test image variants...")

    prepare_yolo_test_images(IMG_SRC, IMG_DST / model_names[0], "nn-stack")
    prepare_yolo_test_images(IMG_SRC, IMG_DST / model_names[1], "diffusion", diffusion_length=3)

    print("✅ Finished preparing test datasets.")

def inference():
    """Run YOLO inference for all models."""
    for name in model_names:
        test_yolo_model(IMG_DST / name, MODEL_DIR / name, name)


def main():
    prepare()
    inference()


if __name__ == "__main__":
    main()