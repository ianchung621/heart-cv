import pandas as pd
from pathlib import Path
from heart_cv.config import TEST_DIR
from heart_cv.postprocessing import (
    get_bounding_box,
    trim_by_bbox3d,
    apply_wbf,
    propagate_box,
    apply_nms,
    BBoxSectioner
)
from heart_cv.submission import write_submission_txt

MODEL_NAMES = ["nn", "diffusion_3"]


# --- Stage 1: Load predictions ---
def load_predictions(test_dir: Path, model_names: list[str]):
    print("📂 Loading predictions...")
    df_preds = [pd.read_csv(test_dir / f"{mn}.csv") for mn in model_names]
    df_pred_x = pd.read_csv(test_dir / "nn_x.csv")
    df_pred_y = pd.read_csv(test_dir / "nn_y.csv")
    print(f"✅ Loaded {len(df_preds)} model predictions + x/y views")
    return df_preds, df_pred_x, df_pred_y


# --- Stage 2: Compute 3D bounding boxes ---
def build_bbox3d(df_pred_x, df_pred_y, ref_df):
    print("📦 Building 3D bounding boxes...")
    bbox3d_df = get_bounding_box(df_pred_x, df_pred_y, ref_df, paddings=(0.2, 0.2, 0))
    print(f"✅ Bounding boxes built for {bbox3d_df['pid'].nunique()} patients")
    return bbox3d_df


# --- Stage 3: Trim predictions by 3D bounding boxes ---
def trim_predictions(df_preds, bbox3d_df):
    print("✂️  Trimming predictions by 3D bounding boxes...")
    dfs_trimmed = [trim_by_bbox3d(df, bbox3d_df) for df in df_preds]
    print("✅ Trimmed predictions")
    return dfs_trimmed


# --- Stage 4: Weighted Box Fusion ---
def fuse_predictions(dfs_trimmed):
    print("🔗 Applying Weighted Box Fusion...")
    df_wbf = apply_wbf(dfs_trimmed, iou_thr=0.4, conf_type="max")
    print(f"✅ Fused {len(dfs_trimmed)} models into {len(df_wbf)} predictions")
    return df_wbf


# --- Stage 5: Section-wise postprocessing ---
def section_postprocess(df_wbf, bbox3d_df):
    print("🧩 Applying section-wise propagation...")
    sectioner = BBoxSectioner()
    df_sectioned = sectioner.apply_by_section(
        df_wbf,
        bbox3d_df,
        {
            "head": lambda df: propagate_box(df, 5, direction="up", conf_decay=0.5), #4
            "tail": lambda df: propagate_box(df, 5, direction="down", conf_decay=0.5), #2
        },
    )
    print("✅ Section-wise propagation complete")
    return df_sectioned


# --- Stage 6: Final NMS and submission ---
def finalize_submission(df_sectioned):
    print("🧹 Applying final NMS...")
    df_final = apply_nms(df_sectioned)
    print(f"✅ Final predictions: {len(df_final)} boxes")
    write_submission_txt(df_final)
    print("🏁 Submission file written.")


# --- Main pipeline ---
def main():
    df_preds, df_pred_x, df_pred_y = load_predictions(TEST_DIR, MODEL_NAMES)
    bbox3d_df = build_bbox3d(df_pred_x, df_pred_y, df_preds[0])
    dfs_trimmed = trim_predictions(df_preds, bbox3d_df)
    df_wbf = fuse_predictions(dfs_trimmed)
    df_sectioned = section_postprocess(df_wbf, bbox3d_df)
    finalize_submission(df_sectioned)


if __name__ == "__main__":
    main()