import argparse
import logging
import time
from typing import Literal
from pathlib import Path

import pandas as pd

from heart_cv.dataset import prepare_yolo_dataset, prepare_yolo_test_images, Cropper
from heart_cv.train import train_yolo_model, train_yolo_model
from heart_cv.val import val_yolo_model, val_cropped_yolo_model
from heart_cv.test import test_yolo_model, test_cropped_yolo_model
from heart_cv.config import CV_SPLIT_DICTS, MODEL_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
from heart_cv.postprocessing import get_bounding_box

fold_idx = 0  # default, will be replaced by CLI

def build_parser():
    p = argparse.ArgumentParser(description="Heart-CV pipeline")
    p.add_argument(
        "--fold_idx",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
    )
    return p

def timed(fn):
    """Minimal timing wrapper without renaming your function."""
    def wrapped(*args, **kwargs):
        logging.info(f"[{fn.__name__}] start")
        t0 = time.time()
        out = fn(*args, **kwargs)
        logging.info(f"[{fn.__name__}] done in {time.time() - t0:.2f}s")
        return out
    return wrapped

ModelName = Literal[
    "nn",
    "diff",
    "nn_x",
    "nn_y",
    "nn_cropped",
    "diff_cropped",
    "nn_cropped_norm",
    "diff_cropped_norm",
]

MODEL_KEYS: list[ModelName] = [
    "nn",
    "diff",
    "nn_x",
    "nn_y",
    "nn_cropped",
    "diff_cropped",
    "nn_cropped_norm",
    "diff_cropped_norm",
]

MODEL_NAMES: dict[ModelName, str] = {
    key: f"fold{fold_idx}_{key}"
    for key in MODEL_KEYS
}

MODELS: dict[ModelName, Path] = {
    key: MODEL_DIR / f"{MODEL_NAMES[key]}.pt"
    for key in MODEL_KEYS
}

DATASET_DIR = Path("dataset")
IMAGES_SRC = DATASET_DIR / "training_image"
LABELS_SRC = DATASET_DIR / "training_label"
TEST_SRC = DATASET_DIR / "testing_image"

YOLO_DATASETS = DATASET_DIR / "yolo_datasets"
YOLO_TESTSETS = DATASET_DIR / "yolo_testsets"

SPLIT_DICT = CV_SPLIT_DICTS[fold_idx]

PERSPECTIVE_X_IMG = DATASET_DIR / "perspectives/x/training_image"
PERSPECTIVE_X_LABEL = DATASET_DIR / "perspectives/x/training_label"
PERSPECTIVE_Y_IMG = DATASET_DIR / "perspectives/y/training_image"
PERSPECTIVE_Y_LABEL = DATASET_DIR / "perspectives/y/training_label"
PERSPECTIVE_X_TEST = DATASET_DIR / "perspectives/x/testing_image"
PERSPECTIVE_Y_TEST = DATASET_DIR / "perspectives/y/testing_image"

CROPPED_DIR = DATASET_DIR / f"fold{fold_idx}_cropped"
CROPPED_IMAGES_SRC = CROPPED_DIR / "training_image"
CROPPED_LABELS_SRC = CROPPED_DIR / "training_label"
CROPPED_TEST_SRC = CROPPED_DIR / "testing_image"

def build_config(fold_idx: int):
    MODEL_NAMES = {key: f"fold{fold_idx}_{key}" for key in MODEL_KEYS}
    MODELS = {key: MODEL_DIR / f"{MODEL_NAMES[key]}.pt" for key in MODEL_KEYS}

    CROPPED_DIR = DATASET_DIR / f"fold{fold_idx}_cropped"
    CROPPED_IMAGES_SRC = CROPPED_DIR / "training_image"
    CROPPED_LABELS_SRC = CROPPED_DIR / "training_label"
    CROPPED_TEST_SRC = CROPPED_DIR / "testing_image"
    SPLIT_DICT = CV_SPLIT_DICTS[fold_idx]

    return MODEL_NAMES, MODELS, CROPPED_DIR, CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, CROPPED_TEST_SRC, SPLIT_DICT

@timed
def prepare():
    prepare_yolo_dataset(IMAGES_SRC, LABELS_SRC, YOLO_DATASETS/ MODEL_NAMES["nn"] , 
        split_patient_dict = SPLIT_DICT, method = "nn-stack")
    prepare_yolo_dataset(IMAGES_SRC, LABELS_SRC, YOLO_DATASETS/ MODEL_NAMES["diff"],
        split_patient_dict = SPLIT_DICT, method = "diffusion", diffusion_length = 3)
    prepare_yolo_test_images(TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["nn"], "nn-stack")
    prepare_yolo_test_images(TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["diff"], "diffusion", diffusion_length = 3)

@timed
def train():
    train_yolo_model(YOLO_DATASETS/ MODEL_NAMES["nn"], 
        no_background=True, model_name=MODEL_NAMES["nn"])
    logging.info("nn done")
    train_yolo_model(YOLO_DATASETS/ MODEL_NAMES["diff"],
        no_background=True, model_name=MODEL_NAMES["diff"])
    logging.info("diff done")

@timed
def val():
    val_yolo_model(YOLO_DATASETS/ MODEL_NAMES["nn"] , MODELS["nn"], MODEL_NAMES["nn"])
    val_yolo_model(YOLO_DATASETS/ MODEL_NAMES["diff"] , MODELS["diff"], MODEL_NAMES["diff"])

@timed
def inference():
    test_yolo_model(YOLO_TESTSETS / MODEL_NAMES["nn"], MODELS["nn"], MODEL_NAMES["nn"])
    test_yolo_model(YOLO_TESTSETS / MODEL_NAMES["diff"], MODELS["diff"], MODEL_NAMES["diff"])

@timed
def prepare_perspective():
    prepare_yolo_dataset(PERSPECTIVE_X_IMG, PERSPECTIVE_X_LABEL, YOLO_DATASETS / MODEL_NAMES["nn_x"],
        split_patient_dict=SPLIT_DICT, method="nn-stack")
    prepare_yolo_dataset(PERSPECTIVE_Y_IMG, PERSPECTIVE_Y_LABEL, YOLO_DATASETS / MODEL_NAMES["nn_y"],
        split_patient_dict=SPLIT_DICT, method="nn-stack")
    prepare_yolo_test_images(PERSPECTIVE_X_TEST, YOLO_TESTSETS / MODEL_NAMES["nn_x"], "nn-stack")
    prepare_yolo_test_images(PERSPECTIVE_Y_TEST, YOLO_TESTSETS / MODEL_NAMES["nn_y"], "nn-stack")

@timed
def train_perspective():
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_x"], no_background=True, model_name=MODEL_NAMES["nn_x"])
    logging.info("nn_x done.")
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_y"], no_background=True, model_name=MODEL_NAMES["nn_y"])
    logging.info("nn_y done.")

@timed
def val_perspective():
    val_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_x"], MODELS["nn_x"], MODEL_NAMES["nn_x"])
    val_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_y"], MODELS["nn_y"], MODEL_NAMES["nn_y"])

@timed
def test_perspective():
    test_yolo_model(YOLO_TESTSETS / MODEL_NAMES["nn_x"], MODELS["nn_x"], MODEL_NAMES["nn_x"])
    test_yolo_model(YOLO_TESTSETS / MODEL_NAMES["nn_y"], MODELS["nn_y"], MODEL_NAMES["nn_y"])

@timed
def inference_train_perspective():
    model_names = ["nn", "nn_x", "nn_y"]
    for mn in model_names:
        train_dir = YOLO_DATASETS/ f"{MODEL_NAMES[mn]}/images/train"
        test_yolo_model(train_dir, MODELS[mn], MODEL_NAMES[mn],  TRAIN_DIR)

@timed
def prepare_cropped(paddings = (0.5,0.5,0.2)):
    csv_names = {mn: f"{MODEL_NAMES[mn]}.csv" for mn in ["nn", "nn_x", "nn_y"]}
    df_pred_train = pd.read_csv(TRAIN_DIR / csv_names["nn"])
    df_pred_x_train = pd.read_csv(TRAIN_DIR / csv_names["nn_x"])
    df_pred_y_train = pd.read_csv(TRAIN_DIR / csv_names["nn_y"])

    df_pred_val = pd.read_csv(VAL_DIR / csv_names["nn"])
    df_pred_x_val = pd.read_csv(VAL_DIR / csv_names["nn_x"])
    df_pred_y_val = pd.read_csv(VAL_DIR / csv_names["nn_y"])

    df_pred_test = pd.read_csv(TEST_DIR / csv_names["nn"])
    df_pred_x_test = pd.read_csv(TEST_DIR / csv_names["nn_x"])
    df_pred_y_test = pd.read_csv(TEST_DIR / csv_names["nn_y"])

    bbox3d_df_train = get_bounding_box(df_pred_x_train, df_pred_y_train, df_pred_train, paddings=paddings)
    bbox3d_df_val = get_bounding_box(df_pred_x_val, df_pred_y_val, df_pred_val, paddings=paddings)
    bbox3d_df_test = get_bounding_box(df_pred_x_test, df_pred_y_test, df_pred_test, paddings=paddings)

    bbox3d_df = pd.concat([bbox3d_df_train, bbox3d_df_val, bbox3d_df_test], ignore_index=True)
    cropper = Cropper(bbox3d_df, CROPPED_DIR)
    cropper.crop_by_bbox3d()

@timed
def prepare_cropped_yolo():
    prepare_yolo_dataset(CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, YOLO_DATASETS / MODEL_NAMES["nn_cropped"], 
        split_patient_dict=SPLIT_DICT, method="nn-stack")
    prepare_yolo_dataset(CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, YOLO_DATASETS / MODEL_NAMES["diff_cropped"], 
        split_patient_dict=SPLIT_DICT, method="diff1", diffusion_length = 3)
    prepare_yolo_dataset(CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, YOLO_DATASETS / MODEL_NAMES["nn_cropped_norm"], 
        split_patient_dict=SPLIT_DICT, method="nn-stack", norm=True)
    prepare_yolo_dataset(CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, YOLO_DATASETS / MODEL_NAMES["diff_cropped_norm"], 
        split_patient_dict=SPLIT_DICT, method="diff1", diffusion_length = 3, norm=True)
    

    prepare_yolo_test_images(CROPPED_TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["nn_cropped"],
        method="nn-stack")
    prepare_yolo_test_images(CROPPED_TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["diff_cropped"],
        method="diff1", diffusion_length = 3)
    prepare_yolo_test_images(CROPPED_TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["nn_cropped_norm"],
        method="nn-stack", norm=True)
    prepare_yolo_test_images(CROPPED_TEST_SRC, YOLO_TESTSETS / MODEL_NAMES["diff_cropped_norm"],
        method="diff1", diffusion_length = 3, norm=True)

@timed
def train_cropped():
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_cropped"], no_background=True, model_name=MODEL_NAMES["nn_cropped"], cropped=True)
    logging.info("nn_cropped done.")
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["diff_cropped"], no_background=True, model_name=MODEL_NAMES["diff_cropped"], cropped=True)
    logging.info("diff_cropped done.")
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_cropped_norm"], no_background=True, model_name=MODEL_NAMES["nn_cropped_norm"], cropped=True)
    logging.info("nn_cropped_norm done.")
    train_yolo_model(YOLO_DATASETS / MODEL_NAMES["diff_cropped_norm"], no_background=True, model_name=MODEL_NAMES["diff_cropped_norm"], cropped=True)
    logging.info("diff_cropped_norm done.")

@timed
def val_cropped():
    val_cropped_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_cropped"], MODELS["nn_cropped"], MODEL_NAMES["nn_cropped"], CROPPED_DIR)
    val_cropped_yolo_model(YOLO_DATASETS / MODEL_NAMES["diff_cropped"], MODELS["diff_cropped"], MODEL_NAMES["diff_cropped"], CROPPED_DIR)
    val_cropped_yolo_model(YOLO_DATASETS / MODEL_NAMES["nn_cropped_norm"], MODELS["nn_cropped_norm"], MODEL_NAMES["nn_cropped_norm"], CROPPED_DIR)
    val_cropped_yolo_model(YOLO_DATASETS / MODEL_NAMES["diff_cropped_norm"], MODELS["diff_cropped_norm"], MODEL_NAMES["diff_cropped_norm"], CROPPED_DIR)


@timed
def inference_cropped():
    test_cropped_yolo_model(YOLO_TESTSETS / MODEL_NAMES["nn_cropped"], MODELS["nn_cropped"], MODEL_NAMES["nn_cropped"], TEST_DIR, CROPPED_DIR)
    test_cropped_yolo_model(YOLO_TESTSETS / MODEL_NAMES["diff_cropped"], MODELS["diff_cropped"], MODEL_NAMES["diff_cropped"], TEST_DIR, CROPPED_DIR)
    test_cropped_yolo_model(YOLO_TESTSETS / MODEL_NAMES["nn_cropped_norm"], MODELS["nn_cropped_norm"], MODEL_NAMES["nn_cropped_norm"], TEST_DIR, CROPPED_DIR)
    test_cropped_yolo_model(YOLO_TESTSETS / MODEL_NAMES["diff_cropped_norm"], MODELS["diff_cropped_norm"], MODEL_NAMES["diff_cropped_norm"], TEST_DIR, CROPPED_DIR)

def main():
    from logging import FileHandler

    log_file = "log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),     # console
            FileHandler(log_file, mode="a")  # append to log.txt
        ]
    )

    logging.info("=== Logging started ===")

    args = build_parser().parse_args()

    fold_idx = args.fold_idx
    logging.info(f"[fold_idx = {fold_idx}]")

    global MODEL_NAMES, MODELS, CROPPED_DIR, CROPPED_IMAGES_SRC, CROPPED_LABELS_SRC, CROPPED_TEST_SRC, SPLIT_DICT

    (
        MODEL_NAMES,
        MODELS,
        CROPPED_DIR,
        CROPPED_IMAGES_SRC,
        CROPPED_LABELS_SRC,
        CROPPED_TEST_SRC,
        SPLIT_DICT
    ) = build_config(fold_idx)

    prepare()
    train()
    val()
    inference()

    prepare_perspective()
    train_perspective()
    val_perspective()
    test_perspective()
    inference_train_perspective()

    prepare_cropped()
    prepare_cropped_yolo()
    train_cropped()
    val_cropped()
    inference_cropped()


if __name__ == "__main__":
    main()