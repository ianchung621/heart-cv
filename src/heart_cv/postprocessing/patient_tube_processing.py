import networkx as nx
from typing import Literal

import pandas as pd
import numpy as np

def pruning_small_side_tubes(
    df_tube: pd.DataFrame,
    keeping_tube_length: int = 10,
    conf_thres: float = 0.5,
) -> pd.DataFrame:
    """
    Prune short or side tubes in a tube DataFrame while preserving strong lone boxes.

    Parameters
    ----------
    df_tube : pd.DataFrame
        Must contain ['img','cls','conf','x1','y1','x2','y2','pid','z','tube_id'].
    keeping_tube_length : int, default=10
        Minimum number of slices (tube length) to keep a tube intact.
    conf_thres : float, default=0.5
        Confidence threshold for retaining lone boxes from pruned tubes.

    Returns
    -------
    pd.DataFrame
        Pruned DataFrame with main and retained boxes.
    """

    if df_tube.empty:
        return df_tube.copy()

    # --- compute per-tube stats --- #
    tube_stats = (
        df_tube.groupby("tube_id")
        .agg(
            n_z=("z", "nunique"),
            z_min=("z", "min"),
            z_max=("z", "max"),
        )
        .reset_index()
    )
    tube_stats["z_span"] = tube_stats["z_max"] - tube_stats["z_min"] + 1

    # --- find main tube --- #
    main_tube_id = tube_stats.loc[tube_stats["n_z"].idxmax(), "tube_id"]
    z_min_main = tube_stats.loc[tube_stats["tube_id"] == main_tube_id, "z_min"].item()
    z_max_main = tube_stats.loc[tube_stats["tube_id"] == main_tube_id, "z_max"].item()

    keep_rows = []

    for tube_id, group in df_tube.groupby("tube_id"):
        n_z = group["z"].nunique()
        z_min = group["z"].min()
        z_max = group["z"].max()

        if tube_id == main_tube_id:
            # always keep main tube
            keep_rows.append(group)
        elif n_z > keeping_tube_length:
            # long tubes always kept
            keep_rows.append(group)
        else:
            # check overlap relative to main tube
            fully_inside = (z_min >= z_min_main) and (z_max <= z_max_main)
            if fully_inside:
                # drop fully side tube, but keep strong boxes
                strong_boxes = group[group["conf"] > conf_thres]
                if not strong_boxes.empty:
                    keep_rows.append(strong_boxes)
            else:
                # extends beyond -> keep
                keep_rows.append(group)


    if not keep_rows:
        return pd.DataFrame(columns=df_tube.columns)

    df_pruned = pd.concat(keep_rows, ignore_index=True)
    return df_pruned.sort_values(["tube_id", "z"]).reset_index(drop=True)

def pruning_recursive_side_tubes(
    df_tube: pd.DataFrame,
    keeping_tube_length: int = 10,
    conf_thres: float = 1.0,
    strictly_inside: bool = False,
) -> pd.DataFrame:
    """
    Prune side tubes using strict containment on z-spans (no recursion needed).

    Strictly-inside uses *open* bounds:
        A inside B  <=>  z_min_B < z_min_A  AND  z_max_A < z_max_B
    so boundary-touching counts as "exceeding" and is kept.

    Always keep:
      - main tube (max n_z)
      - tubes with length > keeping_tube_length
      - any tube NOT strictly inside any already-kept tube

    From pruned tubes, retain only boxes with conf > conf_thres.

    Parameters
    ----------
    df_tube : DataFrame with columns:
        ['img','cls','conf','x1','y1','x2','y2','pid','z','tube_id']
    keeping_tube_length : int
        Threshold to mark tubes as protected (always kept).
    conf_thres : float
        Confidence threshold to salvage lone boxes from pruned tubes.

    Returns
    -------
    DataFrame: pruned df_tube with the same columns.
    """
    if df_tube.empty:
        return df_tube.copy()

    # Per-tube stats
    stats = (
        df_tube.groupby("tube_id")
        .agg(n_z=("z", "nunique"), z_min=("z", "min"), z_max=("z", "max"))
        .reset_index()
    )

    # Main tube = max length
    main_tube_id = stats.loc[stats["n_z"].idxmax(), "tube_id"]

    # Protected tubes (always keep)
    protected = set(stats.loc[stats["n_z"] >= keeping_tube_length, "tube_id"].tolist())
    protected.add(main_tube_id)

    # Sort by span (wide first), then by length, so big parents are considered earlier
    order = stats.sort_values(["n_z"], ascending=False).reset_index(drop=True)

    kept = set()
    kept_spans: list[tuple[int, int, int]] = []  # (z_min, z_max, tube_id)

    def a_inside_b(a_min, a_max, b_min, b_max, strictly_inside) -> bool:
        # open bounds => touching boundary is NOT inside, return a inside b
        if strictly_inside:
            return (b_min < a_min) and (a_max < b_max)
        else:
            return (b_min <= a_min) and (a_max <= b_max)

    for _, row in order.iterrows():
        tid = int(row.tube_id)
        a_min, a_max = int(row.z_min), int(row.z_max)

        if tid in protected:
            kept.add(tid)
            kept_spans.append((a_min, a_max, tid))
            continue

        # If strictly inside ANY already-kept span => prunable
        inside_any = any(a_inside_b(a_min, a_max, b_min, b_max, strictly_inside) for b_min, b_max, _ in kept_spans)

        if not inside_any:
            # Not strictly inside any kept span → keep it
            kept.add(tid)
            kept_spans.append((a_min, a_max, tid))
        # else: prunable

    # Build final dataframe: keep full kept tubes; salvage high-conf boxes from pruned tubes
    keep_rows = []
    for tid, grp in df_tube.groupby("tube_id"):
        if tid in kept:
            keep_rows.append(grp)
        else:
            salvage = grp[grp["conf"] >= conf_thres]
            if not salvage.empty:
                keep_rows.append(salvage)

    if not keep_rows:
        return pd.DataFrame(columns=df_tube.columns)

    out = pd.concat(keep_rows, ignore_index=True)
    return out.sort_values(["tube_id", "z"]).reset_index(drop=True)

def trim_competing_tubes(
        df_pid: pd.DataFrame,
        k: int = 3,
        conf_thres: float = 1,
        trim_main = True,
        trim_tail = True
        ) -> pd.DataFrame:
    """
    Trim 'competing box' overlap between main and tail tubes near valve tail transition.

    Logic
    -----
    1. Identify main tube as the one with longest z-span.
    2. Among other tubes, find candidate tail tube:
        - must overlap with main tube in z (within ±2 slices)
        - must have increasing conf trend and eventual conf > main_tube conf
    3. Find z_takeover where tail tube's conf > main tube's conf for the first time.
    4. Trim:
        - main tube after z_takeover + k
        - tail tube before z_takeover - k

    Parameters
    ----------
    df_pid : pd.DataFrame
        Must contain columns ['pid','tube_id','z','conf'].
        Single patient only (pid.nunique()==1).
    k : int, default=2
        Safety margin for trimming window.

    Returns
    -------
    pd.DataFrame
        Trimmed patient-level dataframe.
    """
    if df_pid["pid"].nunique() != 1:
        raise ValueError("df_pid must contain only one patient (pid.nunique()==1).")
    pid = df_pid["pid"].iloc[0]

    # --- identify main tube by longest z span --- #
    tube_spans = (
        df_pid.groupby("tube_id")["z"]
        .agg(["min", "max"])
        .assign(span=lambda d: d["max"] - d["min"])
    )
    main_tube = tube_spans["span"].idxmax()

    # --- candidate tail tubes: overlapping or nearby in z --- #
    zmin_main, zmax_main = tube_spans.loc[main_tube, ["min", "max"]]
    candidate_tubes = tube_spans[
        (tube_spans["min"] < zmax_main)
        & (tube_spans["max"] >= zmax_main)
        & (tube_spans.index != main_tube)
    ].index

    if len(candidate_tubes) == 0:
        return df_pid  # no competing tube

    # --- find takeover --- #
    main_df = df_pid[df_pid.tube_id == main_tube][["z", "conf"]].set_index("z")

    takeover_records = []
    for tail_tube in candidate_tubes:
        tail_df = df_pid[df_pid.tube_id == tail_tube][["z", "conf"]].set_index("z")
        df_merge = (
            main_df.join(tail_df, how="inner", lsuffix="_main", rsuffix="_tail")
            .sort_index()
        )
        if df_merge.empty:
            continue
        mask = df_merge["conf_tail"] > df_merge["conf_main"]
        if mask.any():
            z_takeover = df_merge.index[mask].min()
            takeover_records.append((tail_tube, z_takeover))

    if not takeover_records:
        return df_pid  # no takeover found

    # --- choose best tail tube: earliest takeover --- #
    tail_tube, z_takeover = sorted(takeover_records, key=lambda x: x[1])[0]

    # --- trimming + reporting --- #
    trimmed = []
    trim_summary = []

    for tube_id, dft in df_pid.groupby("tube_id"):
        n_before = len(dft)
        if tube_id == main_tube and trim_main:
            dft = dft[(dft["z"] <= z_takeover + k) | (dft["conf"] >= conf_thres)]
            n_trim = n_before - len(dft)
            if n_trim > 0:
                trim_summary.append(("main", n_trim))
        elif tube_id == tail_tube and trim_tail:
            dft = dft[(dft["z"] >= z_takeover - k) | (dft["conf"] >= conf_thres)]
            n_trim = n_before - len(dft)
            if n_trim > 0:
                trim_summary.append(("tail", n_trim))
        trimmed.append(dft)

    if trim_summary:
        print(f"pid={pid}: ", end="")
        print(", ".join([f"{role} trimmed {n}" for role, n in trim_summary]))

    return pd.concat(trimmed, ignore_index=True)

def select_best_connected_path(
    df_pid: pd.DataFrame,
    lam: float = 0.1,
    edge_mode: Literal["iou", "center"] = "iou",
) -> pd.DataFrame:
    """Patient tube processing function (full z-span, no cuts)."""
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        denom = a_area + b_area - inter
        return inter / denom if denom > 0 else 0.0

    def _center_sim(a, b, sigma: float = 0.002):
        ax, ay = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
        bx, by = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
        return float(np.exp(-sigma * np.hypot(ax - bx, ay - by)))

    def _edge_weight(a, b):
        return _iou_xyxy(a, b) if edge_mode == "iou" else _center_sim(a, b)

    if df_pid.empty:
        return df_pid

    out = []
    for tube_id, df_tube in df_pid.groupby("tube_id"):
        df_tube = df_tube.sort_values("z").reset_index(drop=False)  # keep old index in "index" col
        layers = [g for _, g in df_tube.groupby("z", sort=True)]
        L = len(layers)
        if L == 0:
            continue
        if L == 1:
            i_best = layers[0]["conf"].values.argmax()
            out.append(layers[0].iloc[[i_best]])
            continue

        conf_layers = [g["conf"].to_numpy(float) for g in layers]
        box_layers = [g[["x1", "y1", "x2", "y2"]].to_numpy(float) for g in layers]

        dp = [np.full(len(conf_layers[k]), -np.inf) for k in range(L)]
        prev = [np.full(len(conf_layers[k]), -1, int) for k in range(L)]
        dp[0] = conf_layers[0].copy()

        for k in range(1, L):
            W = np.zeros((len(box_layers[k - 1]), len(box_layers[k])))
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] = _edge_weight(box_layers[k - 1][i], box_layers[k][j])
            for j in range(W.shape[1]):
                scores = dp[k - 1] + lam * W[:, j]
                i_star = int(np.argmax(scores))
                cand = conf_layers[k][j] + scores[i_star]
                if cand > dp[k][j]:
                    dp[k][j] = cand
                    prev[k][j] = i_star

        j_end = int(np.argmax(dp[-1]))
        path_idx = [j_end]
        for k in range(L - 1, 0, -1):
            path_idx.append(int(prev[k][path_idx[-1]]))
        path_idx = path_idx[::-1]

        # safer: use iloc because we're indexing inside df_tube, not df_pid
        selected = [layers[k].iloc[[path_idx[k]]] for k in range(L)]
        out.append(pd.concat(selected, ignore_index=True))

    if not out:
        return pd.DataFrame(columns=df_pid.columns)

    return pd.concat(out, ignore_index=True).sort_values(["pid", "tube_id", "z"]).reset_index(drop=True)