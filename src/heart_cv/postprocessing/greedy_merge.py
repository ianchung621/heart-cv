import numpy as np
import pandas as pd
from warnings import warn

def merge_box_maximin(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the maximin merged box M(A,B) for inclusion case (B ⊂ A).
    Both A,B are np.array([x1,y1,x2,y2]).
    
    Returns:
        np.array([x1,y1,x2,y2]) for merged box M
        or raises ValueError if invalid (e.g. no overlap or a<=b)
    """
    # ensure correct order
    wA, hA = A[2]-A[0], A[3]-A[1]
    wB, hB = B[2]-B[0], B[3]-B[1]
    if wA <= wB or hA <= hB:
        raise ValueError("Expect B ⊂ A (strictly smaller in both dims).")

    a, b = wA*hA, wB*hB
    m_star = np.sqrt(a*b)

    dW, dH = wA - wB, hA - hB
    Acoef = dW * dH
    Bcoef = wB * dH + hB * dW
    Ccoef = b - m_star

    # quadratic solution (always the '+' root)
    disc = Bcoef**2 - 4*Acoef*Ccoef
    if disc < 0:
        raise ValueError("No real solution (numerical issue).")
    t = (-Bcoef + np.sqrt(disc)) / (2 * Acoef)
    if t > 1 or t < 0:
        raise ValueError("interpolated box out of orignal box")

    # interpolate coordinates
    M = (1 - t) * B + t * A
    return M

def apply_greedy_merge(df_pred: pd.DataFrame, thres: float|int = 1, min_area_ratio = 0.25) -> pd.DataFrame:
    """
    Greedy merge for images with exactly 2 boxes.

    For each image:
      - If there are exactly 2 boxes (A,B)
      - Find which one is larger by area
      - If small box area > thres * (small_box_area inside large box)
        (i.e. small box is mostly contained within large box)
      - Replace both with a merged box M(A,B) via merge_box_maximin()

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must include columns ['img', 'x1','y1','x2','y2'].
    thres : float, default=1
        Containment ratio threshold.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with merged boxes.
    """

    def intersection_area(A, B):
        x1 = max(A[0], B[0])
        y1 = max(A[1], B[1])
        x2 = min(A[2], B[2])
        y2 = min(A[3], B[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def strictly_contain(A, B):
        return (
            B[0] > A[0]
            and B[1] > A[1]
            and B[2] < A[2]
            and B[3] < A[3]
        )

    df_out = []

    for img, g in df_pred.groupby("img", group_keys=False):
        if len(g) != 2:
            df_out.append(g)
            continue

        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy()
        areas = np.array([(b[2]-b[0]) * (b[3]-b[1]) for b in boxes])
        i_large = int(np.argmax(areas))
        i_small = 1 - i_large
        if areas[i_small] < min_area_ratio * areas[i_large]:
            df_out.append(g)
            continue

        A = boxes[i_large]
        B = boxes[i_small]

        if thres == 1:
            if not strictly_contain(A, B):
                df_out.append(g)
                continue
        else:
            inter = intersection_area(A, B)
            contain_ratio = inter / (areas[i_small] + 1e-9)

            if contain_ratio < thres:
                df_out.append(g)
                continue

        try:
            M = merge_box_maximin(A, B)
        except ValueError as e:
            warn(f"[{img}] merge failed, Skipping: {e}")
            df_out.append(g)
            continue

        # build merged row: take the larger box’s metadata
        merged_row = g.iloc[i_large].copy()
        merged_row[["x1", "y1", "x2", "y2"]] = M
        merged_row["conf"] = g["conf"].max()  # optional
        df_out.append(pd.DataFrame([merged_row]))

    df_merged = pd.concat(df_out, ignore_index=True)
    print(f"merged {len(df_pred) - len(df_merged)} boxes")
    return df_merged