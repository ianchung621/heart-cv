import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def isotonic_regression(confs: np.ndarray, increasing: bool, eps: float = 1e-4) -> np.ndarray:
    """
    Fit an isotonic regression to confidence values and enforce strict monotonicity.

    Parameters
    ----------
    confs : np.ndarray
        1D array of confidence values sorted by z (earlier -> later).
    increasing : bool
        If True, enforce non-decreasing (tail-like); if False, non-increasing (head-like).
    eps : float, default=1e-4
        Small offset to make the sequence strictly monotonic.

    Returns
    -------
    np.ndarray
        Strictly monotonic confidence array of same length.
    """
    confs = np.asarray(confs, dtype=float)
    z = np.arange(len(confs))
    iso = IsotonicRegression(increasing=increasing)
    y_mono = iso.fit_transform(z, confs)

    # enforce strict monotonicity by small jitter
    if increasing:
        y_strict = y_mono + eps * np.arange(len(y_mono))
    else:
        y_strict = y_mono - eps * np.arange(len(y_mono))
    return y_strict

def apply_isotonic(df_pred: pd.DataFrame, increasing: bool, eps: float = 1e-4) -> pd.DataFrame:
    """
    Group predictions by pid, sort by z, and apply isotonic regression on 'conf'.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must include columns ['pid', 'z', 'conf'].
    increasing : bool
        Direction of monotonicity.
    eps : float
        Jitter strength for isotonic regression.

    Returns
    -------
    pd.DataFrame
        Same columns as input with updated 'conf' values.
    """
    assert {"pid", "z", "conf"} <= set(df_pred.columns)

    df_out_list = []
    for pid, df_pid in df_pred.groupby("pid", sort=False):
        df_pid = df_pid.sort_values("z").copy()
        df_pid["conf"] = isotonic_regression(df_pid["conf"].to_numpy(), increasing, eps)
        df_out_list.append(df_pid)

    return pd.concat(df_out_list, ignore_index=True)