from pathlib import Path
from heart_cv.train import train_yolo_model

YOLO_DATASET_X = Path("dataset/yolo_datasets/nn_x")
YOLO_DATASET_Y = Path("dataset/yolo_datasets/nn_y")

def main():
    train_yolo_model(YOLO_DATASET_X, no_background=True, model_name='nn_x')
    train_yolo_model(YOLO_DATASET_Y, no_background=True, model_name='nn_y')

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"\nDone in {(time.time()-start)/60:.2f} min")