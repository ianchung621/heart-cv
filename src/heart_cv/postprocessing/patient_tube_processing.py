import pandas as pd

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
    conf_thres: float = 0.5,
    strictly_inside: bool = True,
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

def trim_competing_tubes(df_pid: pd.DataFrame, k: int = 2, conf_thres: float = 0.5) -> pd.DataFrame:
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
        if tube_id == main_tube:
            dft = dft[(dft["z"] <= z_takeover + k) | (dft["conf"] >= conf_thres)]
            n_trim = n_before - len(dft)
            if n_trim > 0:
                trim_summary.append(("main", n_trim))
        elif tube_id == tail_tube:
            dft = dft[(dft["z"] >= z_takeover - k) | (dft["conf"] >= conf_thres)]
            n_trim = n_before - len(dft)
            if n_trim > 0:
                trim_summary.append(("tail", n_trim))
        trimmed.append(dft)

    if trim_summary:
        print(f"pid={pid}: ", end="")
        print(", ".join([f"{role} trimmed {n}" for role, n in trim_summary]))

    return pd.concat(trimmed, ignore_index=True)