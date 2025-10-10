import pandas as pd
import numpy as np

def write_submission_txt(
    df: pd.DataFrame,
    output_path: str = "submission.txt",
    include_conf: bool = True,
):
    """
    Write competition submission file from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
    output_path : str
        Path to output file.
    include_conf : bool
        Include confidence column if True.
    """
    cols = ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'] if include_conf else ['img', 'cls', 'x1', 'y1', 'x2', 'y2']
    df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].round().astype(int)

    nan_mask = df[cols].isna().any(axis=1)

    if nan_mask.any():
        bad_rows = df.loc[nan_mask]
        print("⚠️ Warning: NaN detected in the following rows")
        print(bad_rows)
              
    df[cols].to_csv(output_path, sep=" ", index=False, header=False)