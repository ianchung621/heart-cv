from pathlib import Path

SEED = 42

DATASET_DIR = Path("dataset")
IMAGES_SRC = DATASET_DIR / "training_image"
LABELS_SRC = DATASET_DIR / "training_label"

# --- splitting ---
SPLIT_RATIO = (0.7, 0.2, 0.1)  # (train, val, test)
SPLITS = ["train", "val", "test"]

SPLIT_PATIENT_DICT = {
    'train': ['patient0004', 'patient0005', 'patient0006', 'patient0008', 'patient0010', 'patient0011', 'patient0016', 'patient0017', 'patient0018', 'patient0019', 'patient0021', 'patient0022', 'patient0024', 'patient0025', 'patient0026', 'patient0027', 'patient0028', 'patient0029', 'patient0030', 'patient0031', 'patient0032', 'patient0033', 'patient0035', 'patient0038', 'patient0039', 'patient0040', 'patient0041', 'patient0042', 'patient0043', 'patient0044', 'patient0045', 'patient0046', 'patient0047', 'patient0048', 'patient0050'],
    'val': ['patient0001', 'patient0003', 'patient0007', 'patient0012', 'patient0013', 'patient0015', 'patient0020', 'patient0023', 'patient0036', 'patient0049'],
    'test': ['patient0002', 'patient0009', 'patient0014', 'patient0034', 'patient0037']
    }

# --- YOLO Hyperparameters ---
YOLO_MODEL_NAME    = 'yolo11x.pt'
YOLO_EPOCHS        = 50
YOLO_IMGSZ         = 512
YOLO_BATCH         = 8
YOLO_LR0           = 0.001
YOLO_WEIGHT_DECAY  = 5e-4
YOLO_MIXUP         = 0.15
YOLO_MOSAIC        = 1.0
YOLO_DEGREES       = 15
YOLO_TRANSLATE     = 0.10
YOLO_SCALE         = 0.50
YOLO_SHEAR         = 2.0
YOLO_PERSPECTIVE   = 0.0
YOLO_FLIPUD        = 0.0
YOLO_FLIPLR        = 0.5
YOLO_CLOSE_MOSAIC_LAST = 20