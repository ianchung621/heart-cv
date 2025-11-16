from typing import Callable
import pandas as pd
import numpy as np

class TubeSectionor:
    """
    Assign each box in a patient's tubes to one of:
        'main_no', 'main_o', 'tail_no', 'tail_o'

    Rules
    -----
    main  = tube with largest z-span
    tail  = tube whose z_start is inside or just after main and z_end > main.z_end
    overlap (o) = slices where z values of main and tail intersect
    """

    def _compute_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate z_start, z_end, and length per tube."""
        assert {"tube_id", "z"} <= set(df.columns), "df must contain tube_id and z"
        return (
            df.groupby("tube_id")["z"]
            .agg(z_start="min", z_end="max")
            .assign(length=lambda x: x.z_end - x.z_start)
            .reset_index()
        )

    def _find_main_tail(self, bounds: pd.DataFrame):
        """Return (main_id, tail_id, main_z0, main_z1)."""
        main_row = bounds.loc[bounds["length"].idxmax()]
        main_id, z0, z1 = main_row.tube_id, main_row.z_start, main_row.z_end

        # tail: start within/after main, end beyond
        mask = (bounds.z_start >= z0) & (bounds.z_start <= z1 + 1) & (bounds.z_end > z1) & (bounds.tube_id != main_id)
        if mask.any():
            tail_id = bounds.loc[mask, "tube_id"].iloc[0]
        else:
            tail_id = None
        return main_id, tail_id, z0, z1

    def assign_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each box as 'main_no', 'main_o', 'tail_no', or 'tail_o'.
        """
        bounds = self._compute_bounds(df)
        main_id, tail_id, z0, z1 = self._find_main_tail(bounds)

        df = df.copy()
        if tail_id is None:
            # only main tube exists
            df["section"] = "main_no"
            return df

        # find z range of tail
        tail_row = bounds.loc[bounds.tube_id == tail_id].iloc[0]
        t0, t1 = tail_row.z_start, tail_row.z_end

        # overlapping z range
        overlap_low, overlap_high = max(z0, t0), min(z1, t1)

        def classify(row):
            if row.tube_id == main_id:
                if overlap_low <= row.z <= overlap_high:
                    return "main_o"
                else:
                    return "main_no"
            elif row.tube_id == tail_id:
                if overlap_low <= row.z <= overlap_high:
                    return "tail_o"
                else:
                    return "tail_no"
            else:
                return "else"

        df["section"] = df.apply(classify, axis=1)
        return df.dropna(subset=["section"])

    def apply_by_section(self, df_tube: pd.DataFrame, func_dict: dict[str, Callable]) -> pd.DataFrame:
        """
        Group by pid, apply section-specific functions, return ALL rows.
        Uses a stable `_rowid` to avoid index mismatches.
        """
        assert {"pid", "tube_id", "z"} <= set(df_tube.columns)
        out_all = []

        for pid, df_pid in df_tube.groupby("pid", sort=False):
            df_sec = self.assign_section(df_pid)        # labels: main_no/main_o/tail_no/tail_o
            df_sec = df_sec.copy()
            df_sec["_rowid"] = np.arange(len(df_sec))   # stable row id within this pid

            processed_parts = []
            processed_rowids: set[int] = set()

            for section, func in func_dict.items():
                df_sub = df_sec[df_sec.section == section].copy()
                if df_sub.empty:
                    continue

                # Ensure _rowid survives through user function
                # (if they drop it, we merge it back by index)
                df_sub_before = df_sub.copy()
                out = func(df_sub)  # user may change order/cols
                if out is None or out.empty:
                    continue

                if "_rowid" not in out.columns:
                    # try to recover mapping via original index alignment
                    out = out.merge(
                        df_sub_before[["_rowid"]],
                        left_index=True, right_index=True, how="left"
                    )

                # track processed rowids
                if "_rowid" in out.columns:
                    processed_rowids.update(out["_rowid"].dropna().astype(int).tolist())

                processed_parts.append(out)

            if processed_parts:
                processed = pd.concat(processed_parts, ignore_index=True, sort=False)
                untouched = df_sec[~df_sec["_rowid"].isin(processed_rowids)]
                df_pid_out = pd.concat([processed, untouched], ignore_index=True, sort=False)
                # restore original per-pid order
                if "_rowid" in df_pid_out.columns:
                    df_pid_out = df_pid_out.sort_values("_rowid").drop(columns=["_rowid"]).reset_index(drop=True)
            else:
                df_pid_out = df_sec.drop(columns=["_rowid"]).reset_index(drop=True)

            out_all.append(df_pid_out)

        return pd.concat(out_all, ignore_index=True, sort=False)