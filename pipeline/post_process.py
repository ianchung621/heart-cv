import pandas as pd

from heart_cv.config import TEST_DIR
from heart_cv.postprocessing import (
    get_bounding_box,
    trim_by_bbox3d,
    apply_wbf,
    propagate_box,
    apply_nms,
    build_patient_tube_df,
    aggregate_tube_df,
    pruning_recursive_side_tubes,
    trim_inconfident_head,
    trim_inconfident_tail,
    select_best_connected_path,
    BBoxSectioner,
    TubeSectionor,
    apply_isotonic
)
from heart_cv.submission import write_submission_txt


def main():

    # 0. load predictions from cv folds
    model_keys = ["nn","diff","nn_cropped","diff_cropped","nn_cropped_norm","diff_cropped_norm"]
    df_preds = [pd.concat(
                    [pd.read_csv(TEST_DIR / f"fold{i}_{mn}.csv") for i in range(5)],
                    ignore_index=True
                ) for mn in model_keys]
    df_pred_x = pd.concat([pd.read_csv(TEST_DIR / f"fold{i}_nn_x.csv") for i in range(5)], ignore_index=True)
    df_pred_y = pd.concat([pd.read_csv(TEST_DIR / f"fold{i}_nn_y.csv") for i in range(5)], ignore_index=True)

    # 1. trim by 3D bbox + wbf
    bbox3d_df = get_bounding_box(df_pred_x, df_pred_y, df_preds[0], paddings=(0.2,0.2,5), strict_z_range=False)
    dfs_trimmed = [trim_by_bbox3d(df, bbox3d_df) for df in df_preds]
    df_wbf = apply_wbf(dfs_trimmed, iou_thr=0.4, conf_type='max', skip_box_thr=0.1, weights=[3,3,1,1,2,2])

    # 2. build tube and keep only head/main/tail tube
    df_tube = build_patient_tube_df(df_wbf, min_weight=0.6)
    df_pruned = aggregate_tube_df(df_tube, pruning_recursive_side_tubes) # remove all small tubes besides head/main/tail tube
    df_pruned = trim_inconfident_head(df_pruned, 0.5)  # remove box from head where conf < 0.5
    df_bestpath = aggregate_tube_df(df_pruned, select_best_connected_path) # for each tube, use DP to select best path where [conf + 0.1 iou] is best

    # 3. prior injection by isotonic regression
    ts = TubeSectionor()
    df_rescored = ts.apply_by_section(
        df_bestpath, 
        {
            "tail_o" : lambda df: apply_isotonic(df, True), # give tail overlap increasing conf
            "main_o": lambda df: apply_isotonic(df, False), # give main overlap decreasing conf
            "tail_no": lambda df: apply_isotonic(df, False) # give tail decreasing conf
        }
    )
    df_rescored = trim_inconfident_tail(df_rescored, 0.3)  # remove box from tail where conf < 0.3
    df_rescored = aggregate_tube_df(df_rescored, pruning_recursive_side_tubes)

    # 4. propagate box + nms
    sectioner = BBoxSectioner((0.3,0.5))
    df_sectioned = sectioner.apply_by_section(
        df_rescored,
        bbox3d_df,
        {
            "tail": lambda df: propagate_box(df, 5)
        }
    )
    df_nms = apply_nms(df_sectioned, iou_thres=0.7)

    write_submission_txt(df_nms)

if __name__ == "__main__":
    main()