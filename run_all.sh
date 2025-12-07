#!/usr/bin/env bash

python pipeline/prepare.py

for fold_idx in 0 1 2 3 4; do
    python pipeline/run_fold.py --fold_idx "${fold_idx}"
done

python pipeline/post_process.py