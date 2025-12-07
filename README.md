# AI CUP 2025 - Aortic Valve Detection

This repository implements a full cross-validation pipeline for the heart valve detection task, based on a YOLO-style object-detection model.  
The main entrypoints are in `pipeline/`:

- `pipeline/prepare.py` – prepare data
- `pipeline/run_fold.py` – train/validate/inference one CV fold
- `pipeline/post_process.py` – aggregate predictions and build the final submission file  

---

## Repository Structure

```text
heart-valve-detection/
├── pyproject.toml
├── src/
│   └── heart_cv/
├── dataset/
│   ├── training_image/
│   │   └── patient0001/
│   │       └── patient0001_0001.png
│   └── training_label/
│       └── patient0001/
│           └── patient0001_0203.txt   # YOLO label: class xc yc w h
└── pipeline/
    ├── prepare.py
    ├── run_fold.py
    └── post_process.py
├── run_all.sh
```

## Setup
1.	Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

2.	Install dependencies:
```
pip install -e .
```

## Execute

One-shot script (recommended)

```
./run_all.sh
```

Manual

```
python pipeline/prepare.py
python pipeline/run_fold.py --fold_idx 0
python pipeline/run_fold.py --fold_idx 1
python pipeline/run_fold.py --fold_idx 2
python pipeline/run_fold.py --fold_idx 3
python pipeline/run_fold.py --fold_idx 4
python pipeline/post_process.py
```