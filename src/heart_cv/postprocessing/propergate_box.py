from typing import Literal

import numpy as np
import pandas as pd

def propagate_box(
    df_pred: pd.DataFrame,
    length: int = 1,
    conf_decay: float = 1.0,
    direction: Literal["up", "down", "both"] = "both",
) -> pd.DataFrame:
    """
    Propagate YOLO boxes along z-axis slices for temporal/spatial continuity.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Columns ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'].
        'img' should have format 'patientXXXX_ZZZZ'.
    length : int
        Number of slices to propagate.
    conf_decay : float
        Confidence decay rate. New conf = conf * exp(-conf_decay * dz)
    direction : {'up', 'down', 'both'}
        Direction of propagation along z index.

    Returns
    -------
    pd.DataFrame
        Original + propagated boxes.
    """
    # Extract patient id and slice index
    df = df_pred.copy()
    df["pid"] = df["img"].str.extract(r"patient(\d+)_")[0].astype(int)
    df["z"] = df["img"].str.extract(r"_(\d+)$")[0].astype(int)

    propagated = [df]

    for dz in range(1, length + 1):
        decay_factor = np.exp(-conf_decay * dz)

        if direction in ("up", "both"):
            df_up = df.copy()
            df_up["z"] = df_up["z"] - dz
            df_up["conf"] = df_up["conf"] * decay_factor
            df_up["img"] = df_up["pid"].apply(lambda p: f"patient{p:04d}_") + df_up["z"].apply(lambda z: f"{z:04d}")
            propagated.append(df_up)

        if direction in ("down", "both"):
            df_down = df.copy()
            df_down["z"] = df_down["z"] + dz
            df_down["conf"] = df_down["conf"] * decay_factor
            df_down["img"] = df_down["pid"].apply(lambda p: f"patient{p:04d}_") + df_down["z"].apply(lambda z: f"{z:04d}")
            propagated.append(df_down)

    df_all = pd.concat(propagated, ignore_index=True)
    return df_all